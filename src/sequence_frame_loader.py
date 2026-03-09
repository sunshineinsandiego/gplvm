#!/usr/bin/env python3
"""Utilities for strict per-sequence input parsing and frame selection."""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - dependency availability is env specific
    cv2 = None

try:
    import pydicom
except Exception:  # pragma: no cover - dependency availability is env specific
    pydicom = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DICOM_EXTS = {".dcm", ".dicom"}
_MPEG2_DICOM_TSUIDS = {
    "1.2.840.10008.1.2.4.100",  # MPEG2 Main Profile / Main Level
    "1.2.840.10008.1.2.4.101",  # MPEG2 Main Profile / High Level
}
_MPEG4_DICOM_TSUIDS = {
    "1.2.840.10008.1.2.4.102",  # MPEG-4 AVC/H.264 High Profile / Level 4.1
    "1.2.840.10008.1.2.4.103",  # MPEG-4 AVC/H.264 BD-compatible
    "1.2.840.10008.1.2.4.104",  # MPEG-4 AVC/H.264 High Profile / Level 4.2 (2D)
    "1.2.840.10008.1.2.4.105",  # MPEG-4 AVC/H.264 High Profile / Level 4.2 (3D)
    "1.2.840.10008.1.2.4.106",  # MPEG-4 AVC/H.264 Stereo High Profile
}
_HEVC_DICOM_TSUIDS = {
    "1.2.840.10008.1.2.4.107",  # HEVC/H.265 Main Profile / Level 5.1
    "1.2.840.10008.1.2.4.108",  # HEVC/H.265 Main 10 Profile / Level 5.1
}


@dataclass(frozen=True)
class SequenceSource:
    source_type: str
    source_stem: str
    frame_count: int
    frame_rate_fps: float | None
    primary_path: Path | None
    sequence_paths: tuple[Path, ...] = ()


def parse_bool_flag(value: str) -> bool:
    """Parse booleans from CLI values like true/false/1/0."""
    norm = str(value).strip().lower()
    if norm in {"1", "true", "t", "yes", "y"}:
        return True
    if norm in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Use true/false."
    )


def parse_frame_index_args(values: list[str], frame_count: int) -> list[int]:
    """Parse frame-index CLI tokens supporting int/all/comma-separated lists."""
    if not values:
        raise ValueError("frame_index selection is empty.")

    tokens: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            token = part.strip()
            if token:
                tokens.append(token)

    if not tokens:
        raise ValueError("frame_index selection is empty.")

    lowered = [token.lower() for token in tokens]
    if "all" in lowered:
        if len(tokens) != 1:
            raise ValueError(
                "frame_index value 'all' cannot be combined with explicit indices."
            )
        return list(range(frame_count))

    indices: list[int] = []
    duplicates: set[int] = set()
    seen: set[int] = set()
    for token in tokens:
        try:
            frame_index = int(token)
        except Exception as exc:
            raise ValueError(
                "Invalid frame_index token. Use a non-negative integer, "
                f"a comma-separated list, or 'all'. Got: {token!r}"
            ) from exc

        _validate_frame_index(frame_index, frame_count)
        if frame_index in seen:
            duplicates.add(frame_index)
            continue
        seen.add(frame_index)
        indices.append(frame_index)

    if duplicates:
        rendered = ", ".join(str(idx) for idx in sorted(duplicates))
        raise ValueError(f"Duplicate frame_index values are not allowed: {rendered}")

    return indices


def detect_sequence_source(input_root: Path) -> SequenceSource:
    """Detect one strict sequence source inside input_root."""
    files = _list_sequence_files(input_root)

    if len(files) == 1:
        path = files[0]
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            return SequenceSource(
                source_type="image_single",
                source_stem=path.stem,
                frame_count=1,
                frame_rate_fps=None,
                primary_path=path,
            )
        if ext in VIDEO_EXTS:
            frame_count, fps = _video_metadata(path)
            return SequenceSource(
                source_type="video_single",
                source_stem=path.stem,
                frame_count=frame_count,
                frame_rate_fps=fps,
                primary_path=path,
            )
        if ext in DICOM_EXTS:
            frame_count = _dicom_frame_count(path)
            return SequenceSource(
                source_type="dicom_multiframe" if frame_count > 1 else "dicom_single",
                source_stem=path.stem,
                frame_count=frame_count,
                frame_rate_fps=None,
                primary_path=path,
            )
        raise ValueError(
            f"Unsupported file extension in sequence folder: {path.name}"
        )

    exts = {p.suffix.lower() for p in files}
    if exts.issubset(DICOM_EXTS):
        pairs: list[tuple[int, Path]] = []
        for path in files:
            instance_number = _dicom_instance_number(path)
            pairs.append((instance_number, path))

        unique_instances = {instance for instance, _ in pairs}
        if len(unique_instances) != len(pairs):
            pass # We will just sort by filename as a tiebreaker so no frames are dropped
            
        ordered = tuple(path for _, path in sorted(pairs, key=lambda item: (item[0], item[1].name)))
        return SequenceSource(
            source_type="dicom_sequence",
            source_stem=input_root.name,
            frame_count=len(ordered),
            frame_rate_fps=None,
            primary_path=None,
            sequence_paths=ordered,
        )

    raise ValueError(
        "Sequence folder must contain exactly one source file "
        "(single image/video/dicom) or a pure DICOM slice set."
    )


def detect_all_sources(root_dir: Path) -> list[SequenceSource]:
    """
    Recursively scan root_dir for all files and group them into SequenceSources.
    Logic for DICOMs:
    - NumberOfFrames > 1: dicom_multiframe
    - Same SeriesInstanceUID + Unique InstanceNumbers: dicom_sequence
    - Same SeriesInstanceUID + Duplicate InstanceNumbers: independent dicom_single
    - Single files: image_single, video_single, dicom_single
    """
    if not root_dir.exists() or not root_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {root_dir}")

    sources: list[SequenceSource] = []
    
    # Process directory by directory to group files naturally
    dirs_to_check = [root_dir] + [d for d in root_dir.rglob("*") if d.is_dir()]
    
    for d in dirs_to_check:
        files = [f for f in d.iterdir() if f.is_file() and not f.name.startswith(".")]
        if not files:
            continue
            
        dicom_files = []
        
        for f in files:
            ext = f.suffix.lower()
            
            # 1. Standard Images
            if ext in IMAGE_EXTS:
                sources.append(SequenceSource(
                    source_type="image_single",
                    source_stem=f.stem,
                    frame_count=1,
                    frame_rate_fps=None,
                    primary_path=f,
                ))
                continue
                
            # 2. Standard Videos
            if ext in VIDEO_EXTS:
                try:
                    frame_count, fps = _video_metadata(f)
                    sources.append(SequenceSource(
                        source_type="video_single",
                        source_stem=f.stem,
                        frame_count=frame_count,
                        frame_rate_fps=fps,
                        primary_path=f,
                    ))
                except Exception as e:
                    print(f"Failed to process video {f}: {e}")
                continue
                
            # 3. DICOMs (by extension or probing)
            is_dicom = ext in DICOM_EXTS
            if not is_dicom and pydicom is not None:
                # Probe if it might be an extensionless DICOM
                try:
                    pydicom.dcmread(str(f), stop_before_pixels=True)
                    is_dicom = True
                except Exception:
                    pass
                    
            if is_dicom:
                dicom_files.append(f)
                
        # Group DICOMs in this directory
        if dicom_files and pydicom is not None:
            uid_groups = {}
            for f in dicom_files:
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                    uid = str(ds.get("SeriesInstanceUID", "unknown"))
                    
                    # Check if it's explicitly a multiframe
                    frames = ds.get("NumberOfFrames", None)
                    num_frames = getattr(frames, "value", frames) if hasattr(frames, "value") else frames
                    try:
                        num_frames = int(num_frames) if num_frames is not None else 1
                    except (ValueError, TypeError):
                        num_frames = 1
                        
                    if num_frames > 1:
                        sources.append(SequenceSource(
                            source_type="dicom_multiframe",
                            source_stem=f.stem,
                            frame_count=num_frames,
                            frame_rate_fps=None,
                            primary_path=f,
                        ))
                    else:
                        # Single frame, extract instance number for grouping
                        inst = ds.get("InstanceNumber", None)
                        inst_val = getattr(inst, "value", inst) if hasattr(inst, "value") else inst
                        try:
                            inst_num = int(inst_val) if inst_val is not None else -1
                        except (ValueError, TypeError):
                            inst_num = -1
                            
                        uid_groups.setdefault(uid, []).append({"path": f, "inst": inst_num})
                        
                except Exception as e:
                    print(f"Failed to map DICOM {f}: {e}")
                    
            # Resolve UID groups
            for uid, items in uid_groups.items():
                if len(items) == 1:
                    sources.append(SequenceSource(
                        source_type="dicom_single",
                        source_stem=items[0]["path"].stem,
                        frame_count=1,
                        frame_rate_fps=None,
                        primary_path=items[0]["path"],
                    ))
                else:
                    # Check for instance number collisions
                    inst_nums = [item["inst"] for item in items]
                    # If there's a collision (-1 counts as collision if multiple) or dupe numbers
                    has_duplicates = len(inst_nums) != len(set(inst_nums))
                    
                    if has_duplicates:
                        # Treat all as independent dicom_single
                        for item in items:
                            sources.append(SequenceSource(
                                source_type="dicom_single",
                                source_stem=item["path"].stem,
                                frame_count=1,
                                frame_rate_fps=None,
                                primary_path=item["path"],
                            ))
                    else:
                        # True sequence!
                        sorted_items = sorted(items, key=lambda x: x["inst"])
                        ordered_paths = tuple(x["path"] for x in sorted_items)
                        sources.append(SequenceSource(
                            source_type="dicom_sequence",
                            source_stem=f"{d.name}_{uid[-6:]}",
                            frame_count=len(ordered_paths),
                            frame_rate_fps=None,
                            primary_path=None,
                            sequence_paths=ordered_paths,
                        ))

    return sources


def source_frame_path(source: SequenceSource, frame_index: int) -> Path:
    """Resolve the concrete file path used for the selected frame."""
    _validate_frame_index(frame_index, source.frame_count)
    if source.source_type == "dicom_sequence":
        return source.sequence_paths[frame_index]
    if source.primary_path is None:
        raise ValueError("Source path is unavailable for this source type.")
    return source.primary_path


def output_stem(source: SequenceSource, frame_index: int) -> str:
    """Build a stable output stem for the selected frame."""
    _validate_frame_index(frame_index, source.frame_count)
    return f"{source.source_stem}_f{frame_index:04d}"


def load_selected_rgb_frame(source: SequenceSource, frame_index: int) -> np.ndarray:
    """Load the selected frame/slice and return RGB uint8 [H,W,3]."""
    _validate_frame_index(frame_index, source.frame_count)

    if source.source_type == "image_single":
        if source.primary_path is None:
            raise ValueError("Missing source path for image input.")
        frame = np.asarray(Image.open(source.primary_path))
        return _to_rgb_uint8(frame)

    if source.source_type == "video_single":
        if source.primary_path is None:
            raise ValueError("Missing source path for video input.")
        frame = _load_video_frame_rgb(source.primary_path, frame_index)
        return _to_rgb_uint8(frame)

    if source.source_type in {"dicom_single", "dicom_multiframe"}:
        if pydicom is None:
            raise ImportError(
                "pydicom is required for DICOM input. Install with: pip install pydicom"
            )
        if source.primary_path is None:
            raise ValueError("Missing source path for DICOM input.")
        ds = pydicom.dcmread(str(source.primary_path))
        frame = _load_dicom_frame(
            ds=ds,
            source_type=source.source_type,
            frame_index=frame_index,
            path=source.primary_path,
        )
        return _to_rgb_uint8(frame)

    if source.source_type == "dicom_sequence":
        if pydicom is None:
            raise ImportError(
                "pydicom is required for DICOM input. Install with: pip install pydicom"
            )
        selected_path = source.sequence_paths[frame_index]
        ds = pydicom.dcmread(str(selected_path))
        frame = np.asarray(ds.pixel_array)
        return _to_rgb_uint8(frame)

    raise ValueError(f"Unsupported source_type: {source.source_type}")


def _list_sequence_files(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(
            f"Input root must be a sequence directory: {input_root}"
        )

    nested_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    if nested_dirs:
        rendered = ", ".join(p.name for p in nested_dirs)
        raise ValueError(
            "Nested subdirectories are not allowed in input-root. "
            f"Found: {rendered}"
        )

    files = sorted(p for p in input_root.iterdir() if p.is_file())
    if not files:
        raise ValueError(f"No files found under input-root: {input_root}")
    return files


def _validate_frame_index(frame_index: int, frame_count: int) -> None:
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got: {frame_index}")
    if frame_index >= frame_count:
        raise ValueError(
            f"frame_index {frame_index} is out of range for frame_count={frame_count}"
        )


def _dicom_frame_count(path: Path) -> int:
    if pydicom is None:
        raise ImportError(
            "pydicom is required for DICOM input. Install with: pip install pydicom"
        )
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    raw = ds.get("NumberOfFrames", 1)
    try:
        count = int(raw)
    except Exception as exc:
        raise ValueError(f"Invalid NumberOfFrames in {path}: {raw!r}") from exc
    if count < 1:
        raise ValueError(f"Invalid NumberOfFrames in {path}: {count}")
    return count


def _dicom_instance_number(path: Path) -> int:
    if pydicom is None:
        raise ImportError(
            "pydicom is required for DICOM input. Install with: pip install pydicom"
        )
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    raw = ds.get("InstanceNumber", None)
    if raw is None or str(raw).strip() == "":
        raise ValueError(
            f"Missing InstanceNumber in DICOM slice sequence file: {path}"
        )
    try:
        return int(raw)
    except Exception as exc:
        raise ValueError(
            f"Non-integer InstanceNumber in DICOM slice sequence file {path}: {raw!r}"
        ) from exc





def _video_metadata(path: Path) -> tuple[int, float | None]:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open video file: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                frame_count += 1
        cap.release()
        if frame_count <= 0:
            raise ValueError(f"Could not determine frame count for video: {path}")
        return frame_count, (fps if fps > 0 else None)

    return _video_metadata_ffprobe(path)


def _load_video_frame_rgb(path: Path, frame_index: int) -> np.ndarray:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open video file: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok or frame_bgr is None:
            raise RuntimeError(
                f"Failed to read frame_index={frame_index} from video: {path}"
            )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return _load_video_frame_rgb_ffmpeg(path=path, frame_index=frame_index)


def _video_metadata_ffprobe(path: Path) -> tuple[int, float | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed for {path}: {stderr}")

    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path}") from exc

    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video streams for {path}")
    stream = streams[0]

    frame_count = _parse_positive_int(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _parse_positive_int(stream.get("nb_frames"))
    if frame_count is None or frame_count <= 0:
        raise ValueError(f"Could not determine frame count for video: {path}")

    fps = _parse_rate(stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_rate(stream.get("r_frame_rate"))
    return frame_count, fps


def _parse_positive_int(value: object) -> int | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _parse_rate(value: object) -> float | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        text = str(value)
        if "/" in text:
            num_str, den_str = text.split("/", 1)
            num = float(num_str)
            den = float(den_str)
            if den == 0:
                return None
            rate = num / den
        else:
            rate = float(text)
    except Exception:
        return None
    return rate if rate > 0 else None


def _load_video_frame_rgb_ffmpeg(path: Path, frame_index: int) -> np.ndarray:
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got: {frame_index}")

    frame_selector = f"select=eq(n\\,{frame_index})"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        frame_selector,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed to read frame_index={frame_index} from video {path}: {stderr}"
        )
    if not proc.stdout:
        raise RuntimeError(
            f"ffmpeg returned no frame bytes for frame_index={frame_index} from video {path}"
        )
    return _decode_image_bytes_rgb(proc.stdout, context=f"{path} frame_index={frame_index}")


def _decode_image_bytes_rgb(image_bytes: bytes, context: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return np.asarray(img.convert("RGB"))
    except Exception as exc:
        raise RuntimeError(f"Failed to decode image bytes for {context}") from exc


def _load_dicom_frame(
    ds: "pydicom.Dataset",
    source_type: str,
    frame_index: int,
    path: Path,
) -> np.ndarray:
    try:
        pixels = np.asarray(ds.pixel_array)
    except Exception as exc:
        if source_type != "dicom_multiframe":
            raise
        return _load_dicom_multiframe_with_ffmpeg(
            ds=ds,
            frame_index=frame_index,
            path=path,
            base_error=exc,
        )

    if source_type == "dicom_single":
        if frame_index != 0:
            raise ValueError("frame_index must be 0 for single-frame DICOM input.")
        return _single_or_color_dicom_frame(pixels, path)

    return _multiframe_dicom_frame(pixels, frame_index, path)


def _load_dicom_multiframe_with_ffmpeg(
    ds: "pydicom.Dataset",
    frame_index: int,
    path: Path,
    base_error: Exception,
) -> np.ndarray:
    if not hasattr(ds, "PixelData"):
        raise RuntimeError(f"DICOM has no PixelData: {path}") from base_error

    try:
        from pydicom.encaps import generate_frames
    except Exception as exc:
        raise RuntimeError(
            "DICOM fallback decode requires pydicom.encaps.generate_frames."
        ) from exc

    frame_count = _parse_positive_int(ds.get("NumberOfFrames")) or 1
    frame_payloads = list(generate_frames(ds.PixelData, number_of_frames=frame_count))
    if not frame_payloads:
        raise RuntimeError(
            f"Failed to extract encapsulated frame payloads from DICOM: {path}"
        ) from base_error

    # Some DICOM video files embed a full mp4/h264 bitstream as one payload item.
    # Others provide one payload per frame. Try direct per-frame image decode first.
    if len(frame_payloads) == frame_count and len(frame_payloads) > frame_index:
        try:
            return _decode_image_bytes_rgb(
                frame_payloads[frame_index],
                context=f"{path} frame_index={frame_index}",
            )
        except Exception:
            pass

    transfer_syntax_uid = str(
        getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "")
    )
    payload = frame_payloads[0] if len(frame_payloads) == 1 else b"".join(frame_payloads)
    format_hints = _dicom_video_input_formats(transfer_syntax_uid, payload)

    last_error: Exception | None = None
    for input_format in format_hints:
        try:
            return _decode_video_bytes_frame_rgb(
                payload=payload,
                frame_index=frame_index,
                context=f"{path} frame_index={frame_index}",
                input_format=input_format,
            )
        except Exception as exc:
            last_error = exc

    hint_text = ", ".join(fmt if fmt is not None else "auto" for fmt in format_hints)
    raise RuntimeError(
        "Failed DICOM compressed-video fallback decode "
        f"(transfer_syntax_uid={transfer_syntax_uid or 'unknown'}, tried={hint_text})"
    ) from (last_error or base_error)


def _decode_video_bytes_frame_rgb(
    payload: bytes,
    frame_index: int,
    context: str,
    input_format: str | None,
) -> np.ndarray:
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got: {frame_index}")

    frame_selector = f"select=eq(n\\,{frame_index})"
    cmd = ["ffmpeg", "-v", "error"]
    if input_format is not None:
        cmd.extend(["-f", input_format])
    cmd.extend(
        [
            "-i",
            "pipe:0",
            "-vf",
            frame_selector,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]
    )
    proc = subprocess.run(cmd, input=payload, capture_output=True)
    if proc.returncode == 0 and proc.stdout:
        return _decode_image_bytes_rgb(proc.stdout, context=context)

    # Some codecs/containers (especially MP4 payloads) may require seekable input.
    # Fall back to decoding from a temporary file.
    suffix = _video_suffix_for_payload(payload=payload, input_format=input_format)
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(payload)
        tmp.flush()
        file_cmd = ["ffmpeg", "-v", "error"]
        if input_format is not None:
            file_cmd.extend(["-f", input_format])
        file_cmd.extend(
            [
                "-i",
                tmp.name,
                "-vf",
                frame_selector,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ]
        )
        file_proc = subprocess.run(file_cmd, capture_output=True)

    if file_proc.returncode != 0 or not file_proc.stdout:
        stdin_err = proc.stderr.decode("utf-8", errors="replace").strip()
        file_err = file_proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "ffmpeg failed to decode DICOM payload frame "
            f"frame_index={frame_index}, input_format={input_format or 'auto'}, "
            f"stdin_error={stdin_err or '<none>'}, file_error={file_err or '<none>'}"
        )
    return _decode_image_bytes_rgb(file_proc.stdout, context=context)


def _video_suffix_for_payload(payload: bytes, input_format: str | None) -> str:
    if _looks_like_mp4(payload):
        return ".mp4"
    if input_format == "h264":
        return ".h264"
    if input_format == "hevc":
        return ".hevc"
    if input_format == "mpegvideo":
        return ".m2v"
    return ".bin"


def _dicom_video_input_formats(
    transfer_syntax_uid: str,
    payload: bytes,
) -> tuple[str | None, ...]:
    if _looks_like_mp4(payload):
        # Let ffmpeg probe from the MP4 container.
        return (None,)

    if transfer_syntax_uid in _MPEG4_DICOM_TSUIDS:
        return ("h264", None)
    if transfer_syntax_uid in _HEVC_DICOM_TSUIDS:
        return ("hevc", None)
    if transfer_syntax_uid in _MPEG2_DICOM_TSUIDS:
        return ("mpegvideo", None)
    return (None,)


def _looks_like_mp4(payload: bytes) -> bool:
    head = payload[:64]
    return len(head) >= 12 and b"ftyp" in head[:32]


def _single_or_color_dicom_frame(pixels: np.ndarray, path: Path) -> np.ndarray:
    if pixels.ndim == 2:
        return pixels
    if pixels.ndim == 3 and pixels.shape[-1] in {3, 4}:
        return pixels
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        return pixels[0]
    raise ValueError(
        "Single-frame DICOM produced an unexpected pixel array shape "
        f"for {path}: {pixels.shape}"
    )


def _multiframe_dicom_frame(
    pixels: np.ndarray, frame_index: int, path: Path
) -> np.ndarray:
    if pixels.ndim < 3:
        raise ValueError(
            "Multi-frame DICOM produced an invalid pixel array shape "
            f"for {path}: {pixels.shape}"
        )
    if frame_index >= pixels.shape[0]:
        raise ValueError(
            f"frame_index {frame_index} exceeds decoded frame axis for {path}: "
            f"decoded_frames={pixels.shape[0]}"
        )
    return pixels[frame_index]


def _to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)

    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.ndim != 3:
        raise ValueError(
            f"Expected 2D/3D frame array, got shape={arr.shape}"
        )

    channels = arr.shape[-1]
    if channels == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif channels == 3:
        pass
    elif channels >= 4:
        arr = arr[..., :3]
    else:
        raise ValueError(
            f"Unsupported channel count for frame conversion: {channels}"
        )

    if arr.dtype == np.uint8:
        return arr

    arrf = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arrf)
    if not finite.all():
        arrf = np.where(finite, arrf, 0.0)

    mn = float(arrf.min())
    mx = float(arrf.max())
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    arrf = (arrf - mn) / (mx - mn)
    return np.clip(arrf * 255.0, 0.0, 255.0).astype(np.uint8)
