#!/usr/bin/env python3
"""
Deterministic DPP keyframe selection from frame embeddings.

Input:
- --input can be either:
  1) a single sample directory containing top-level *_vector.npy files, or
  2) a root directory (for example /workspace/scan_output/<subject>) that contains
     sequence/model/sample subdirectories.

Behavior:
- For each discovered sample directory, loads and flattens top-level *_vector.npy files.
- L2-normalizes frame vectors.
- Builds cosine kernel K = X X^T and adds diagonal jitter EPSILON * I.
- Selects min(N, num_frames) keyframes using deterministic greedy MAP-DPP
  (maximize log-det of selected submatrix).
- Prints coverage report for k in [0, ..., min(num_frames, selected_n + coverage_extra)].
- Writes selected_keyframes_dpp.json inside each processed sample directory.

JSON output fields:
- selected_frame_ids
- kernel_type
- method
- seed
- n_keyframes_selected
- coverage_table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


SEED = 0
EPSILON = 1e-6
OUTPUT_JSON_NAME = "selected_keyframes_dpp.json"
VECTOR_SUFFIX = "_vector.npy"
TIE_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic DPP keyframe selection with coverage reporting."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input directory. Either a sample directory containing top-level "
            "*_vector.npy files, or a root directory scanned for model/sample folders."
        ),
    )
    parser.add_argument(
        "--n-keyframes",
        type=int,
        default=10,
        help="Fixed number of keyframes to select (default: 10).",
    )
    parser.add_argument(
        "--coverage-extra",
        type=int,
        default=5,
        help=(
            "Print coverage for k in [0..n_keyframes+coverage_extra], "
            "clipped to num_frames (default: 5)."
        ),
    )
    return parser.parse_args()


def load_embeddings(input_dir: Path) -> Tuple[np.ndarray, List[str]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input path not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input must be a directory: {input_dir}")

    top_level_npy = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix == ".npy"],
        key=lambda p: p.name,
    )
    vector_files = [p for p in top_level_npy if p.name.endswith(VECTOR_SUFFIX)]
    invalid_npy = [p.name for p in top_level_npy if not p.name.endswith(VECTOR_SUFFIX)]

    if not vector_files:
        raise FileNotFoundError(
            f"No top-level {VECTOR_SUFFIX} files found in input directory: {input_dir}"
        )
    if invalid_npy:
        vector_names = [p.name for p in vector_files]
        print(
            "Warning: found top-level .npy files that are not *_vector.npy; "
            f"ignoring: {invalid_npy}. Using vector files: {vector_names}"
        )

    vectors: List[np.ndarray] = []
    frame_ids: List[str] = []
    for file_path in vector_files:
        arr = np.load(file_path, allow_pickle=False)
        vec = np.asarray(arr, dtype=np.float64).reshape(-1)
        vectors.append(vec)
        frame_ids.append(file_path.stem)

    lengths = {vec.shape[0] for vec in vectors}
    if len(lengths) != 1:
        raise ValueError(
            "Per-frame vectors must all have same flattened length. "
            f"Found lengths: {sorted(lengths)}"
        )
    embeddings = np.stack(vectors, axis=0)
    return embeddings, frame_ids


def has_top_level_vectors(input_dir: Path) -> bool:
    top_level_npy = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix == ".npy"
    ]
    return any(p.name.endswith(VECTOR_SUFFIX) for p in top_level_npy)


def discover_sample_dirs(input_root: Path) -> List[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input path not found: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"--input must be a directory: {input_root}")

    # If this path is already a sample folder, process only this folder.
    if has_top_level_vectors(input_root):
        return [input_root.resolve()]

    model_names = {"dinov3", "medsam2"}
    discovered: List[Path] = []
    for model_dir in sorted(
        [p for p in input_root.rglob("*") if p.is_dir() and p.name in model_names],
        key=lambda p: str(p),
    ):
        for sample_dir in sorted(
            [p for p in model_dir.iterdir() if p.is_dir()],
            key=lambda p: str(p),
        ):
            if has_top_level_vectors(sample_dir):
                discovered.append(sample_dir.resolve())

    if not discovered:
        raise FileNotFoundError(
            f"No sample directories with top-level {VECTOR_SUFFIX} files found under: {input_root}"
        )
    return discovered


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def build_cosine_matrices(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = l2_normalize_rows(embeddings)
    cosine = x @ x.T
    coverage_sim = np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)
    kernel = cosine.copy()
    np.fill_diagonal(kernel, np.diag(kernel) + EPSILON)
    return coverage_sim, kernel


def logdet_subset(kernel: np.ndarray, subset: Sequence[int]) -> float:
    if len(subset) == 0:
        return 0.0
    block = kernel[np.ix_(subset, subset)]
    sign, value = np.linalg.slogdet(block)
    if sign <= 0:
        return float("-inf")
    return float(value)


def greedy_dpp_map_order(kernel: np.ndarray, k: int) -> List[int]:
    n = kernel.shape[0]
    k = max(0, min(k, n))
    selected: List[int] = []
    remaining = set(range(n))

    rng = np.random.default_rng(SEED)
    tie_order = rng.permutation(n)
    tie_rank = np.empty(n, dtype=np.int64)
    tie_rank[tie_order] = np.arange(n, dtype=np.int64)

    for _ in range(k):
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            score = logdet_subset(kernel, selected + [idx])
            if best_idx is None or score > best_score + TIE_TOL:
                best_idx = idx
                best_score = score
            elif abs(score - best_score) <= TIE_TOL:
                if tie_rank[idx] < tie_rank[best_idx]:
                    best_idx = idx
                    best_score = score
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def coverage_from_subset(coverage_sim: np.ndarray, subset: Sequence[int]) -> float:
    if len(subset) == 0:
        return 0.0
    max_sim = coverage_sim[:, subset].max(axis=1)
    return float(max_sim.mean())


def volume_proxy_from_logdet(diversity_logdet: float, k: int) -> float:
    if k == 0:
        return 0.0
    if not np.isfinite(diversity_logdet):
        return 0.0
    return float(np.exp(0.5 * diversity_logdet))


def build_coverage_table(
    order: Sequence[int],
    frame_ids: Sequence[str],
    coverage_sim: np.ndarray,
    kernel: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prev_cov = 0.0
    prev_div = 0.0
    for k in range(len(order) + 1):
        subset_idx = list(order[:k])
        selected_prefix = [frame_ids[idx] for idx in subset_idx]
        cov = coverage_from_subset(coverage_sim, subset_idx)
        div = logdet_subset(kernel, subset_idx)
        delta_cov = cov - prev_cov if k > 0 else 0.0
        delta_div = div - prev_div if k > 0 else 0.0
        volume_proxy = volume_proxy_from_logdet(div, k)
        rows.append(
            {
                "k": int(k),
                "selected_frame_ids_prefix": selected_prefix,
                "coverage": float(cov),
                "delta_coverage": float(delta_cov),
                "diversity_logdet": float(div),
                "delta_diversity": float(delta_div),
                "volume_proxy": float(volume_proxy),
            }
        )
        prev_cov = cov
        prev_div = div
    return rows


def print_coverage_table(rows: Sequence[Dict[str, Any]]) -> None:
    print("\nCoverage report (k = number of selected frames):")
    print(
        "k\tcoverage\tdelta_coverage\tdiversity_logdet\t"
        "delta_diversity\tvolume_proxy"
    )
    for row in rows:
        print(
            f"{row['k']}\t{row['coverage']:.6f}\t{row['delta_coverage']:.6f}\t"
            f"{row['diversity_logdet']:.6f}\t{row['delta_diversity']:.6f}\t"
            f"{row['volume_proxy']:.6f}"
        )


def save_keyframes_json(
    output_path: Path,
    input_dir: Path,
    selected_ids: Sequence[str],
    n_requested: int,
    num_input_vectors: int,
    coverage_extra: int,
    coverage_table: Sequence[Dict[str, Any]],
) -> None:
    selected_row = next((row for row in coverage_table if row["k"] == len(selected_ids)), None)
    payload = {
        "selected_frame_ids": list(selected_ids),
        "kernel_type": "cosine_l2_normalized_with_epsilon_jitter",
        "method": "dpp_map_greedy_logdet",
        "seed": int(SEED),
        "epsilon": float(EPSILON),
        "n_keyframes_requested": int(n_requested),
        "n_keyframes_selected": int(len(selected_ids)),
        "num_input_vectors": int(num_input_vectors),
        "coverage_extra": int(coverage_extra),
        "input_dir": str(input_dir),
        "selected_set_diversity_logdet": (
            float(selected_row["diversity_logdet"]) if selected_row else None
        ),
        "selected_set_volume_proxy": (
            float(selected_row["volume_proxy"]) if selected_row else None
        ),
        "coverage_table": list(coverage_table),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_keyframes <= 0:
        raise ValueError("--n-keyframes must be > 0")
    if args.coverage_extra < 0:
        raise ValueError("--coverage-extra must be >= 0")
    if EPSILON <= 0:
        raise ValueError("EPSILON constant must be > 0")

    sample_dirs = discover_sample_dirs(args.input)
    print(f"Discovered sample directories: {len(sample_dirs)}")

    for sample_dir in sample_dirs:
        embeddings, frame_ids = load_embeddings(sample_dir)
        num_frames = embeddings.shape[0]
        if num_frames == 0:
            raise ValueError(f"No frame embeddings loaded in {sample_dir}")

        n_select = min(args.n_keyframes, num_frames)
        coverage_sim, kernel = build_cosine_matrices(embeddings)

        k_report_max = min(num_frames, n_select + args.coverage_extra)
        order_for_report = greedy_dpp_map_order(kernel, k_report_max)
        selected_indices = order_for_report[:n_select]
        selected_ids = [frame_ids[idx] for idx in selected_indices]
        coverage_table = build_coverage_table(order_for_report, frame_ids, coverage_sim, kernel)

        print(f"\nSample: {sample_dir}")
        print(f"Loaded embeddings: {num_frames} frames, dim={embeddings.shape[1]}")
        print(
            "Selection method: deterministic greedy MAP-DPP "
            "(maximize logdet of selected cosine-kernel submatrix)"
        )
        print(f"Requested N: {args.n_keyframes}")
        print(f"Selected N: {n_select}")
        print(f"Coverage report range: k=0..{k_report_max}")
        print(f"Seed (tie-break): {SEED}")
        print(f"Epsilon jitter: {EPSILON}")
        print(f"Selected frame ids (N={n_select}): {selected_ids}")

        print_coverage_table(coverage_table)
        output_json = sample_dir / OUTPUT_JSON_NAME
        save_keyframes_json(
            output_path=output_json,
            input_dir=sample_dir,
            selected_ids=selected_ids,
            n_requested=args.n_keyframes,
            num_input_vectors=num_frames,
            coverage_extra=args.coverage_extra,
            coverage_table=coverage_table,
        )
        print(f"\nWrote keyframe JSON: {output_json}")


if __name__ == "__main__":
    main()
