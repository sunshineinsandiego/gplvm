#!/usr/bin/env python3
"""
Extract DINOv3 intermediate-layer feature maps for one strict sequence folder.

Accepted --input content (no nested subdirectories):
- single image file
- single video file
- single DICOM file (single-frame or multi-frame)
- DICOM slice sequence (multiple DICOM files with required InstanceNumber)

One or more frames/slices are processed per run via --frame-index.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sequence_frame_loader import (
    SequenceSource,
    detect_sequence_source,
    load_selected_rgb_frame,
    output_stem,
    parse_frame_index_args,
    parse_bool_flag,
    source_frame_path,
)

from select_keyframes_dpp import (
    build_cosine_matrices,
    greedy_dpp_map_order,
    build_coverage_table,
    save_keyframes_json,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _default_models_dir() -> Path:
    env_models_dir = os.environ.get("PTT_MODELS_DIR")
    if env_models_dir:
        return Path(env_models_dir)

    workspace_models = Path("/workspace/models")
    if workspace_models.exists():
        return workspace_models

    return Path(__file__).resolve().parents[1] / "models"


def _adjust_offsets_for_patch_multiple(
    length: int, off_a: int, off_b: int, patch_size: int, axis: str
) -> tuple[int, int, int]:
    usable = length - off_a - off_b
    if usable <= 0:
        raise ValueError(
            f"Initial offsets remove all pixels on {axis}: "
            f"length={length}, off_a={off_a}, off_b={off_b}"
        )

    remainder = usable / patch_size - usable // patch_size
    if remainder > 0.5:
        add_pixels = round((usable / patch_size), 0) * patch_size - usable
        to_a = round(add_pixels / 2, 0)
        to_b = add_pixels - to_a
        off_a = int(off_a - to_a)
        off_b = int(off_b - to_b)
    else:
        take_pixels = usable - (usable // patch_size) * patch_size
        to_a = round(take_pixels / 2, 0)
        to_b = take_pixels - to_a
        off_a = int(off_a + to_a)
        off_b = int(off_b + to_b)

    new_length = length - off_a - off_b
    if new_length <= 0:
        raise ValueError(
            f"Patch-multiple adjustment removed all pixels on {axis}: "
            f"length={length}, off_a={off_a}, off_b={off_b}"
        )
    if new_length % patch_size != 0:
        raise ValueError(
            f"Adjusted {axis} is not divisible by patch_size={patch_size}: "
            f"adjusted_{axis}={new_length}"
        )
    return off_a, off_b, new_length


def _crop_or_pad_image(
    image_rgb_u8: np.ndarray,
    patch_size: int,
    off_top: int,
    off_bottom: int,
    off_left: int,
    off_right: int,
) -> np.ndarray:
    if off_top < 0 or off_bottom < 0 or off_left < 0 or off_right < 0:
        raise ValueError(
            "User-provided offsets must be >= 0. "
            f"Got top={off_top}, bottom={off_bottom}, left={off_left}, right={off_right}"
        )

    height, width = image_rgb_u8.shape[0], image_rgb_u8.shape[1]
    off_top, off_bottom, adjusted_h = _adjust_offsets_for_patch_multiple(
        length=height,
        off_a=off_top,
        off_b=off_bottom,
        patch_size=patch_size,
        axis="height",
    )
    off_left, off_right, adjusted_w = _adjust_offsets_for_patch_multiple(
        length=width,
        off_a=off_left,
        off_b=off_right,
        patch_size=patch_size,
        axis="width",
    )

    crop_top = max(0, off_top)
    crop_bottom = max(0, off_bottom)
    crop_left = max(0, off_left)
    crop_right = max(0, off_right)

    if crop_top + crop_bottom >= height or crop_left + crop_right >= width:
        raise ValueError(
            "Crop offsets exceed image size after patch adjustment: "
            f"height={height}, width={width}, "
            f"top={off_top}, bottom={off_bottom}, left={off_left}, right={off_right}"
        )

    cropped = image_rgb_u8[
        crop_top : (height - crop_bottom),
        crop_left : (width - crop_right),
        :,
    ]

    pad_top = max(0, -off_top)
    pad_bottom = max(0, -off_bottom)
    pad_left = max(0, -off_left)
    pad_right = max(0, -off_right)

    if pad_top or pad_bottom or pad_left or pad_right:
        cropped = np.pad(
            cropped,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    if cropped.shape[0] != adjusted_h or cropped.shape[1] != adjusted_w:
        raise ValueError(
            "Internal shape mismatch after crop/pad adjustment: "
            f"expected=({adjusted_h},{adjusted_w}), got={cropped.shape[:2]}"
        )
    return cropped


def _preprocess(
    image_rgb_u8: np.ndarray,
    patch_size: int,
    off_top: int,
    off_bottom: int,
    off_left: int,
    off_right: int,
    imagenet: bool,
) -> torch.Tensor:
    adjusted = _crop_or_pad_image(
        image_rgb_u8=image_rgb_u8,
        patch_size=patch_size,
        off_top=off_top,
        off_bottom=off_bottom,
        off_left=off_left,
        off_right=off_right,
    )
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    image_t_u8 = torch.from_numpy(adjusted).permute(2, 0, 1).contiguous()
    image_t = to_float(image_t_u8)
    if imagenet:
        image_t = normalize(image_t)
    return image_t


def _extract_features(
    model: torch.nn.Module,
    image_t: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    image_t = image_t[None].to(device)
    with torch.inference_mode():
        intermediate_layers = [i * (model.n_blocks - 1) // 4 for i in range(1, 5)]
        layers = model.get_intermediate_layers(
            image_t,
            n=intermediate_layers,
            reshape=True,
            norm=False,
            return_class_token=True,
        )
        feats = torch.cat(
            [
                torch.cat(
                    [layer[0], layer[1][:, :, None, None].expand_as(layer[0])],
                    dim=1,
                )
                for layer in layers
            ],
            dim=1,
        )
    return feats.squeeze(0).detach().cpu()


def _preview_rgb_u8_from_encoder_input(
    image_t: torch.Tensor,
) -> np.ndarray:
    preview = image_t.detach().cpu().clone()
    if preview.ndim != 3 or preview.shape[0] < 3:
        raise ValueError(
            f"Expected encoder input tensor shape [C,H,W] with C>=3, got {tuple(preview.shape)}"
        )

    preview = preview[:3]
    # Keep preview in encoder-input space (no de-normalization).
    preview = preview.clamp(0.0, 1.0)
    preview_u8 = (preview.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return preview_u8


def _save_pre_encoder_debug(
    image_t: torch.Tensor,
    debug_dir: Path,
    base_name: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = debug_dir / f"{base_name}_pre_encoder_tensor.npy"
    preview_path = debug_dir / f"{base_name}_pre_encoder_preview.png"

    np.save(tensor_path, image_t.detach().cpu().numpy())
    preview_u8 = _preview_rgb_u8_from_encoder_input(image_t=image_t)
    Image.fromarray(preview_u8, mode="RGB").save(preview_path)

    tensor = image_t.detach().cpu()
    print(
        "Saved DINOv3 pre-encoder debug assets: "
        f"tensor={tensor_path} shape={tuple(tensor.shape)} "
        f"min={float(tensor.min()):.6f} max={float(tensor.max()):.6f}, "
        f"preview={preview_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 features for one strict sequence folder.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Sequence folder with exactly one source "
            "(single image/video/dicom) or a pure DICOM slice set."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output base directory (default: scan_output). "
            "Final output is auto-derived as "
            "<output-base>/<subject>/<sequence>/dinov3/<subset>/."
        ),
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("/app/dinov3"),
        help="Path to local DINOv3 repo inside container.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            os.environ.get(
                "DINOV3_WEIGHTS",
                "/workspace/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            )
        ),
        help="Path to local DINOv3 checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--frame-index",
        nargs="+",
        default=["0"],
        help=(
            "Frame selection: one index (e.g. 0), multiple indices "
            "(e.g. 1 2 3 or 1,2,3), or all."
        ),
    )
    parser.add_argument(
        "--off-top",
        type=int,
        default=0,
        help="Initial top offset (must be >= 0).",
    )
    parser.add_argument(
        "--off-bottom",
        type=int,
        default=0,
        help="Initial bottom offset (must be >= 0).",
    )
    parser.add_argument(
        "--off-left",
        type=int,
        default=0,
        help="Initial left offset (must be >= 0).",
    )
    parser.add_argument(
        "--off-right",
        type=int,
        default=0,
        help="Initial right offset (must be >= 0).",
    )
    parser.add_argument(
        "--overwrite",
        type=parse_bool_flag,
        default=True,
        help="Overwrite outputs if they exist (true/false, default: true).",
    )
    parser.add_argument(
        "--imagenet",
        type=parse_bool_flag,
        default=False,
        help="Apply ImageNet normalization (true/false, default: false).",
    )
    parser.add_argument(
        "--save-pre-encoder",
        type=parse_bool_flag,
        default=False,
        help=(
            "Save the exact tensor fed to the DINOv3 encoder plus a PNG preview "
            "(true/false, default: false)."
        ),
    )
    parser.add_argument(
        "--save-encodings",
        type=parse_bool_flag,
        default=False,
        help="Save extracted DINOv3 feature tensors (.npy) (true/false, default: false).",
    )
    parser.add_argument(
        "--dpp-keyframes",
        type=int,
        default=0,
        help="If >0, perform DPP selection directly in memory and save the JSON manifest.",
    )
    return parser


def _validate_user_offsets(args: argparse.Namespace) -> None:
    offsets = {
        "off_top": args.off_top,
        "off_bottom": args.off_bottom,
        "off_left": args.off_left,
        "off_right": args.off_right,
    }
    negatives = {name: value for name, value in offsets.items() if value < 0}
    if negatives:
        rendered = ", ".join(f"{k}={v}" for k, v in negatives.items())
        raise ValueError(
            f"User-provided offsets must be >= 0. Invalid values: {rendered}"
        )


def _print_source_summary(source: SequenceSource) -> None:
    fps_part = (
        f", frame_rate_fps={source.frame_rate_fps:.6f}"
        if source.frame_rate_fps is not None
        else ""
    )
    print(
        f"Detected source_type={source.source_type}, "
        f"frame_count={source.frame_count}{fps_part}"
    )


def _subset_label(frame_index_args: list[str], frame_indices: list[int]) -> str:
    tokens: list[str] = []
    for raw in frame_index_args:
        for part in str(raw).split(","):
            token = part.strip()
            if token:
                tokens.append(token)

    if len(tokens) == 1 and tokens[0].lower() == "all":
        return "all"
    if len(frame_indices) == 1:
        return f"idx{frame_indices[0]}"
    return "list" + "_".join(str(idx) for idx in frame_indices)


def _resolve_sequence_output_root(input_root: Path, output_root: Path | None) -> Path:
    base = (output_root or Path("scan_output")).expanduser().resolve()
    subject_id = input_root.parent.name
    sequence_id = input_root.name
    return base / subject_id / sequence_id


def main() -> int:
    args = build_parser().parse_args()

    input_root = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    repo_dir = args.repo_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()

    if not repo_dir.exists():
        print(f"Error: DINOv3 repo not found: {repo_dir}", file=sys.stderr)
        return 2
    if not weights.is_file():
        print(f"Error: checkpoint not found: {weights}", file=sys.stderr)
        return 3
    print(f"Using DINOv3 weights: {weights}")

    try:
        _validate_user_offsets(args)
        source = detect_sequence_source(input_root)
        _print_source_summary(source)
        target_frame_indices = parse_frame_index_args(
            values=args.frame_index,
            frame_count=source.frame_count,
        )
    except Exception as exc:
        print(f"Error while validating input={input_root}: {exc}", file=sys.stderr)
        return 4

    subset_label = _subset_label(args.frame_index, target_frame_indices)
    sequence_out_root = _resolve_sequence_output_root(
        input_root=input_root,
        output_root=output_root,
    )
    out_dir = sequence_out_root / "dinov3" / subset_label
    out_dir.mkdir(parents=True, exist_ok=True)
    target_outputs: list[tuple[int, Path, list[Path]]] = []
    for frame_index in target_frame_indices:
        base_name = output_stem(source, frame_index)
        out_path = out_dir / f"{base_name}.npy"
        debug_paths: list[Path] = []
        if args.save_pre_encoder:
            debug_dir = out_path.parent
            debug_paths = [
                debug_dir / f"{base_name}_pre_encoder_tensor.npy",
                debug_dir / f"{base_name}_pre_encoder_preview.png",
            ]
        target_outputs.append((frame_index, out_path, debug_paths))

    if not args.overwrite:
        existing: list[Path] = []
        for _, out_path, debug_paths in target_outputs:
            paths_to_check = [*debug_paths]
            if args.save_encodings:
                paths_to_check.append(out_path)
            for path in paths_to_check:
                if path.exists():
                    existing.append(path)
        if existing:
            rendered = ", ".join(str(path) for path in existing)
            print(
                "Error: overwrite attempted with --overwrite=false. "
                f"Existing paths: {rendered}",
                file=sys.stderr,
            )
            return 5

    device = torch.device(args.device)
    dino = torch.hub.load(
        str(repo_dir),
        "dinov3_vits16",
        source="local",
        weights=str(weights),
    )
    dino.eval().to(device)

    selected_source = "<unresolved>"
    current_frame_index = -1
    in_memory_feats = []
    in_memory_ids = []
    try:
        for frame_index, out_path, _ in target_outputs:
            current_frame_index = frame_index
            base_name = out_path.stem
            selected_source = str(source_frame_path(source, frame_index))
            frame_rgb_u8 = load_selected_rgb_frame(source, frame_index)
            image_t = _preprocess(
                image_rgb_u8=frame_rgb_u8,
                patch_size=16,
                off_top=args.off_top,
                off_bottom=args.off_bottom,
                off_left=args.off_left,
                off_right=args.off_right,
                imagenet=args.imagenet,
            )
            if args.save_pre_encoder:
                _save_pre_encoder_debug(
                    image_t=image_t,
                    debug_dir=out_path.parent,
                    base_name=base_name,
                )
            feats = _extract_features(dino, image_t, device)
            
            if args.dpp_keyframes > 0:
                in_memory_feats.append(feats.numpy())
                in_memory_ids.append(base_name)
                
            if args.save_encodings:
                np.save(out_path, feats.numpy())
    except Exception as exc:
        print(
            "Error while processing source. "
            f"source={selected_source}, source_type={source.source_type}, "
            f"frame_index={current_frame_index}. Cause: {exc}",
            file=sys.stderr,
        )
        return 6

    if args.dpp_keyframes > 0 and len(in_memory_feats) > 0:
        embeddings = np.stack(in_memory_feats, axis=0)
        num_frames = embeddings.shape[0]
        n_select = min(args.dpp_keyframes, num_frames)
        coverage_sim, kernel = build_cosine_matrices(embeddings)
        k_report_max = min(num_frames, n_select + 5)
        order_for_report = greedy_dpp_map_order(kernel, k_report_max)
        selected_indices = order_for_report[:n_select]
        selected_ids = [in_memory_ids[idx] for idx in selected_indices]
        coverage_table = build_coverage_table(order_for_report, in_memory_ids, coverage_sim, kernel)
        
        output_json = out_dir / "selected_keyframes_dpp.json"
        save_keyframes_json(
            output_path=output_json,
            input_dir=out_dir,
            selected_ids=selected_ids,
            n_requested=args.dpp_keyframes,
            num_input_vectors=num_frames,
            coverage_extra=5,
            coverage_table=coverage_table,
        )
        print(f"Performed in-memory DPP selection. Wrote JSON to: {output_json}")

    if args.save_encodings and len(target_outputs) == 1:
        print(f"Done. Wrote features to: {target_outputs[0][1]}")
    elif args.save_encodings:
        print(
            "Done. Wrote DINOv3 features for "
            f"{len(target_outputs)} frames under: {out_dir}"
        )
    else:
        print("Done. Skipped writing DINOv3 feature .npy files (--save-encodings=false).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
