#!/usr/bin/env python3
"""
Extract MedSAM2 image-encoder feature maps for one strict sequence folder.

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
from torchvision.transforms import Compose, Resize, ToTensor

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

from sequence_frame_loader import (
    SequenceSource,
    detect_sequence_source,
    load_selected_rgb_frame,
    output_stem,
    parse_frame_index_args,
    parse_bool_flag,
    source_frame_path,
)

TRUNK_LEVEL_NAMES = ("trunk_s0", "trunk_s1", "trunk_s2", "trunk_s3")
FPN_LEVEL_NAMES = ("fpn_l0", "fpn_l1", "fpn_l2")


def _default_models_dir() -> Path:
    env_models_dir = os.environ.get("PTT_MODELS_DIR")
    if env_models_dir:
        return Path(env_models_dir)

    workspace_models = Path("/workspace/models")
    if workspace_models.exists():
        return workspace_models

    return Path(__file__).resolve().parents[1] / "models"


def _crop_image(
    image_rgb_u8: np.ndarray,
    off_top: int,
    off_bottom: int,
    off_left: int,
    off_right: int,
) -> np.ndarray:
    """Crop image by given offsets. No patch-multiple adjustment."""
    if off_top < 0 or off_bottom < 0 or off_left < 0 or off_right < 0:
        raise ValueError(
            "Offsets must be >= 0. "
            f"Got top={off_top}, bottom={off_bottom}, left={off_left}, right={off_right}"
        )
    height, width = image_rgb_u8.shape[0], image_rgb_u8.shape[1]
    cropped_h = height - off_top - off_bottom
    cropped_w = width - off_left - off_right
    if cropped_h <= 0 or cropped_w <= 0:
        raise ValueError(
            f"Crop offsets remove all pixels: "
            f"height={height}, width={width}, "
            f"off_top={off_top}, off_bottom={off_bottom}, "
            f"off_left={off_left}, off_right={off_right}"
        )
    return image_rgb_u8[
        off_top : (height - off_bottom),
        off_left : (width - off_right),
        :,
    ]


def _build_model(
    config_dir: Path,
    config_file: str,
    ckpt_path: Path,
    device: str,
    use_high_res: bool | None,
) -> torch.nn.Module:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        kwargs = {}
        if use_high_res is not None:
            flag = "true" if use_high_res else "false"
            kwargs["hydra_overrides_extra"] = [f"++model.use_high_res_features_in_sam={flag}"]
        model = build_sam2(
            config_file=config_file,
            ckpt_path=str(ckpt_path),
            device=device,
            mode="eval",
            **kwargs,
        )
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract MedSAM2 image-encoder features for one strict sequence folder.",
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
            "<output-base>/<subject>/<sequence>/medsam2/<subset>/."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(os.environ.get("MEDSAM2_ROOT", "/opt/MedSAM2")) / "sam2",
        help="Path to MedSAM2 config directory (default: /opt/MedSAM2/sam2).",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="Config file path relative to config-dir.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "MEDSAM2_CHECKPOINT",
                str(_default_models_dir() / "MedSAM2_latest.pt"),
            )
        ),
        help="Path to local MedSAM2 checkpoint.",
    )
    parser.add_argument(
        "--use-high-res",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force use_high_res_features_in_sam on load (default: true).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available).",
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
        "--imagenet",
        type=parse_bool_flag,
        default=True,
        help="Apply ImageNet normalization (true/false, default: true).",
    )
    parser.add_argument(
        "--overwrite",
        type=parse_bool_flag,
        default=True,
        help="Overwrite outputs if they exist (true/false, default: true).",
    )
    parser.add_argument(
        "--save-pre-encoder",
        type=parse_bool_flag,
        default=False,
        help=(
            "Save the exact tensor fed to the MedSAM2 encoder plus a PNG preview "
            "(true/false, default: false)."
        ),
    )
    parser.add_argument(
        "--save-encodings",
        type=parse_bool_flag,
        default=False,
        help="Save extracted MedSAM2 feature tensors (.npy) (true/false, default: false).",
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


def _expected_feature_paths(out_dir: Path, base_name: str) -> list[Path]:
    paths: list[Path] = []
    for level_name in TRUNK_LEVEL_NAMES:
        paths.append(out_dir / level_name / f"{base_name}.npy")
    for level_name in FPN_LEVEL_NAMES:
        paths.append(out_dir / level_name / f"{base_name}.npy")
    return paths


def _preview_rgb_u8_from_encoder_input(
    image_t_batched: torch.Tensor,
) -> np.ndarray:
    tensor = image_t_batched.detach().cpu()
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            "Expected encoder input tensor shape [1,C,H,W], "
            f"got {tuple(tensor.shape)}"
        )

    rgb = tensor[0, :3].clone()
    if rgb.shape[0] < 3:
        raise ValueError(
            f"Expected at least 3 channels for RGB preview, got {tuple(tensor.shape)}"
        )

    # Keep preview in encoder-input space (no de-normalization).
    rgb = rgb.clamp(0.0, 1.0)
    preview_u8 = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return preview_u8


def _save_pre_encoder_debug(
    image_t_batched: torch.Tensor,
    debug_dir: Path,
    base_name: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = debug_dir / f"{base_name}_pre_encoder_tensor.npy"
    preview_path = debug_dir / f"{base_name}_pre_encoder_preview.png"

    np.save(tensor_path, image_t_batched.detach().cpu().numpy())
    preview_u8 = _preview_rgb_u8_from_encoder_input(image_t_batched)
    Image.fromarray(preview_u8, mode="RGB").save(preview_path)

    tensor = image_t_batched.detach().cpu()
    print(
        "Saved MedSAM2 pre-encoder debug assets: "
        f"tensor={tensor_path} shape={tuple(tensor.shape)} "
        f"min={float(tensor.min()):.6f} max={float(tensor.max()):.6f}, "
        f"preview={preview_path}"
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
    config_dir = args.config_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    if not config_dir.exists():
        print(f"Error: config dir not found: {config_dir}", file=sys.stderr)
        return 2
    if not checkpoint.is_file():
        print(f"Error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 3
    print(f"Using MedSAM2 checkpoint: {checkpoint}")

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

    model = _build_model(
        config_dir=config_dir,
        config_file=args.config_file,
        ckpt_path=checkpoint,
        device=args.device,
        use_high_res=args.use_high_res,
    )
    model.eval()
    if args.imagenet:
        transform = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)
    else:
        transform = Compose([
            ToTensor(),
            Resize((model.image_size, model.image_size)),
        ])

    subset_label = _subset_label(args.frame_index, target_frame_indices)
    sequence_out_root = _resolve_sequence_output_root(
        input_root=input_root,
        output_root=output_root,
    )
    out_dir = sequence_out_root / "medsam2" / subset_label
    out_dir.mkdir(parents=True, exist_ok=True)
    for level_name in [*TRUNK_LEVEL_NAMES, *FPN_LEVEL_NAMES]:
        (out_dir / level_name).mkdir(parents=True, exist_ok=True)
    target_outputs: list[tuple[int, str, list[Path], list[Path]]] = []
    for frame_index in target_frame_indices:
        base_name = output_stem(source, frame_index)
        expected_feature_paths = _expected_feature_paths(out_dir, base_name)
        debug_paths: list[Path] = []
        if args.save_pre_encoder:
            debug_dir = out_dir / "trunk_s0"
            debug_paths = [
                debug_dir / f"{base_name}_pre_encoder_tensor.npy",
                debug_dir / f"{base_name}_pre_encoder_preview.png",
            ]
        target_outputs.append(
            (frame_index, base_name, expected_feature_paths, debug_paths)
        )

    if not args.overwrite:
        existing: list[Path] = []
        for _, _, expected_feature_paths, debug_paths in target_outputs:
            paths_to_check = [*debug_paths]
            if args.save_encodings:
                paths_to_check.extend(expected_feature_paths)
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

    selected_source = "<unresolved>"
    current_frame_index = -1
    try:
        for frame_index, base_name, _, _ in target_outputs:
            current_frame_index = frame_index
            selected_source = str(source_frame_path(source, frame_index))
            frame_rgb_u8 = load_selected_rgb_frame(source, frame_index)
            if args.off_top or args.off_bottom or args.off_left or args.off_right:
                frame_rgb_u8 = _crop_image(
                    frame_rgb_u8,
                    args.off_top,
                    args.off_bottom,
                    args.off_left,
                    args.off_right,
                )
            image = Image.fromarray(frame_rgb_u8, mode="RGB")
            image_t = transform(image)[None].to(args.device)
            if args.save_pre_encoder:
                _save_pre_encoder_debug(
                    image_t_batched=image_t,
                    debug_dir=out_dir / "trunk_s0",
                    base_name=base_name,
                )

            with torch.inference_mode():
                trunk_feats = list(model.image_encoder.trunk(image_t))
                backbone_out = model.forward_image(image_t)
                fpn_feats = list(backbone_out.get("backbone_fpn", []))

            output_pairs: list[tuple[Path, np.ndarray]] = []
            for level_idx, feat in enumerate(trunk_feats):
                out_path = out_dir / f"trunk_s{level_idx}" / f"{base_name}.npy"
                output_pairs.append((out_path, feat.squeeze(0).detach().cpu().numpy()))

            for level_idx, feat in enumerate(fpn_feats):
                out_path = out_dir / f"fpn_l{level_idx}" / f"{base_name}.npy"
                output_pairs.append((out_path, feat.squeeze(0).detach().cpu().numpy()))

            if args.save_encodings:
                for out_path, arr in output_pairs:
                    np.save(out_path, arr)
    except Exception as exc:
        print(
            "Error while processing source. "
            f"source={selected_source}, source_type={source.source_type}, "
            f"frame_index={current_frame_index}. Cause: {exc}",
            file=sys.stderr,
        )
        return 5

    if args.save_encodings and len(target_outputs) == 1:
        print(f"Done. Wrote features under: {out_dir}")
    elif args.save_encodings:
        print(
            "Done. Wrote MedSAM2 features for "
            f"{len(target_outputs)} frames under: {out_dir}"
        )
    else:
        print("Done. Skipped writing MedSAM2 feature .npy files (--save-encodings=false).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
