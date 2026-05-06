#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any

from prepare_guard_train_SFT_ground2 import (
    BUILD_STATS_FILE_NAME,
    DATASET_INFO_FILE_NAME,
    STALE_VAL_FILE_NAME,
    SYSTEM_PROMPT,
    TRAIN_FILE_NAME,
    build_dataset_info,
    build_split_stats,
    init_worker_state,
    load_attack_manifest_map,
    load_json_list,
    process_sample_for_dataset,
    write_json,
)


THIS_DIR = Path(__file__).resolve().parent
GUARD_ROOT = THIS_DIR.parents[1]

DEFAULT_ORIGINAL_TRAIN_JSON = THIS_DIR / "data_final" / TRAIN_FILE_NAME
DEFAULT_GENERATED_SOURCE = (
    GUARD_ROOT / "adversarial_attack_data" / "screenshot_malicious_reasoned" / "datasets" / "train_overlayed.guard_attack.malicious_only_reasoned.quarter_balanced_merged.json"
)
DEFAULT_ATTACK_MANIFEST_DIR = GUARD_ROOT / "adversarial_attack_data" / "screenshot_malicious_reasoned" / "manifests"
DEFAULT_OUTPUT_DIR = THIS_DIR / "data_final_plus_generated_only_r2_reasoned_quarter"


def remove_stale_val_artifacts(output_dir: Path) -> None:
    stale_val_path = output_dir / STALE_VAL_FILE_NAME
    if stale_val_path.exists():
        stale_val_path.unlink()


def find_duplicate_sample_ids(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if sample_id in seen:
            duplicates.append(sample_id)
        else:
            seen.add(sample_id)
    return duplicates


def summarize_generated_stats(generated_stats: list[dict[str, Any]]) -> dict[str, Any]:
    max_processed_html_chars = -1
    max_processed_html_sample_id = ""
    max_prompt_html_chars = -1
    max_prompt_html_sample_id = ""
    num_samples_with_attack_manifest = 0
    num_html_attack_rebuilt = 0
    num_attack_prompt_verified = 0

    for stats in generated_stats:
        processed_html_chars = int(stats.get("processed_html_chars", 0))
        prompt_html_chars = int(stats.get("prompt_html_chars", 0))
        sample_id = str(stats.get("sample_id", ""))

        if bool(stats.get("sample_has_attack_manifest")):
            num_samples_with_attack_manifest += 1
        if bool(stats.get("html_attack_applied")):
            num_html_attack_rebuilt += 1
        if bool(stats.get("attack_prompt_verified")):
            num_attack_prompt_verified += 1

        if processed_html_chars > max_processed_html_chars:
            max_processed_html_chars = processed_html_chars
            max_processed_html_sample_id = sample_id
        if prompt_html_chars > max_prompt_html_chars:
            max_prompt_html_chars = prompt_html_chars
            max_prompt_html_sample_id = sample_id

    return {
        "num_samples_with_attack_manifest": num_samples_with_attack_manifest,
        "num_html_attack_rebuilt": num_html_attack_rebuilt,
        "num_attack_prompt_verified": num_attack_prompt_verified,
        "max_generated_processed_html_chars": max_processed_html_chars,
        "sample_id_with_max_generated_processed_html_chars": max_processed_html_sample_id,
        "max_generated_prompt_html_chars": max_prompt_html_chars,
        "sample_id_with_max_generated_prompt_html_chars": max_prompt_html_sample_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the already-prepared data_final/guard_train_sft.json with generated-only Guard attack samples. "
            "The generated-only JSON is converted with the same HTML-attack reconstruction logic used by "
            "prepare_guard_train_SFT_ground2.py before merging."
        )
    )
    parser.add_argument(
        "--original-train-json",
        type=Path,
        default=DEFAULT_ORIGINAL_TRAIN_JSON,
        help=f"Prepared ShareGPT train file to keep as the original split. Default: {DEFAULT_ORIGINAL_TRAIN_JSON}",
    )
    parser.add_argument(
        "--generated-source",
        type=Path,
        default=DEFAULT_GENERATED_SOURCE,
        help=f"Generated-only Guard attack JSON to convert and merge. Default: {DEFAULT_GENERATED_SOURCE}",
    )
    parser.add_argument(
        "--attack-manifest-dir",
        type=Path,
        default=DEFAULT_ATTACK_MANIFEST_DIR,
        help=f"Manifest directory used to rebuild attacked HTML text. Default: {DEFAULT_ATTACK_MANIFEST_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the merged ShareGPT dataset. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-html-chars",
        type=int,
        default=0,
        help="Ignored for compatibility. Generated samples always keep the full reconstructed HTML. Default: 0",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=48,
        help="Worker process count for converting generated samples. Use 1 for sequential mode. Default: 48",
    )
    parser.add_argument(
        "--limit-original",
        type=int,
        default=0,
        help="Optional debug limit for original prepared records. Use 0 to keep all. Default: 0",
    )
    parser.add_argument(
        "--limit-generated",
        type=int,
        default=0,
        help="Optional debug limit for generated-only records. Use 0 to keep all. Default: 0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    original_train_json = args.original_train_json.expanduser().resolve()
    generated_source = args.generated_source.expanduser().resolve()
    attack_manifest_dir = args.attack_manifest_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1.")

    original_records = load_json_list(original_train_json)
    generated_samples = load_json_list(generated_source)

    if args.limit_original > 0:
        original_records = original_records[: args.limit_original]
    if args.limit_generated > 0:
        generated_samples = generated_samples[: args.limit_generated]

    duplicate_original = find_duplicate_sample_ids(original_records)
    if duplicate_original:
        preview = ", ".join(duplicate_original[:5])
        raise ValueError(
            f"Duplicate sample_id values found in original prepared records. count={len(duplicate_original)} preview=[{preview}]"
        )

    attack_manifests_by_sample_id = load_attack_manifest_map(attack_manifest_dir)

    converted_generated_records: list[dict[str, Any]] = []
    generated_stats: list[dict[str, Any]] = []

    executor: ProcessPoolExecutor | None = None
    if args.num_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker_state,
            initargs=(attack_manifests_by_sample_id,),
        )
    else:
        init_worker_state(attack_manifests_by_sample_id)

    try:
        if executor is None:
            results = (
                process_sample_for_dataset(sample=sample, max_html_chars=args.max_html_chars)
                for sample in generated_samples
            )
        else:
            chunksize = max(1, len(generated_samples) // (args.num_workers * 4))
            results = executor.map(
                process_sample_for_dataset,
                generated_samples,
                repeat(args.max_html_chars),
                chunksize=chunksize,
            )

        total_generated = len(generated_samples)
        for index, result in enumerate(results, start=1):
            converted_generated_records.append(result["record"])
            generated_stats.append(result.get("stats", {}))
            if index % 500 == 0 or index == total_generated:
                print(f"[merge-generated] processed {index}/{total_generated} samples")
    finally:
        if executor is not None:
            executor.shutdown()

    duplicate_generated = find_duplicate_sample_ids(converted_generated_records)
    if duplicate_generated:
        preview = ", ".join(duplicate_generated[:5])
        raise ValueError(
            f"Duplicate sample_id values found in converted generated records. count={len(duplicate_generated)} preview=[{preview}]"
        )

    merged_records = [*original_records, *converted_generated_records]
    duplicate_merged = find_duplicate_sample_ids(merged_records)
    if duplicate_merged:
        preview = ", ".join(duplicate_merged[:5])
        raise ValueError(
            f"Duplicate sample_id values found after merging original + generated records. "
            f"count={len(duplicate_merged)} preview=[{preview}]"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_val_artifacts(output_dir)

    dataset_info = build_dataset_info()
    generated_summary = summarize_generated_stats(generated_stats)
    build_stats = {
        "original_prepared": build_split_stats(original_records),
        "generated_only_converted": build_split_stats(converted_generated_records),
        "merged": build_split_stats(merged_records),
        "has_val": False,
        "max_html_chars": args.max_html_chars,
        "max_html_chars_behavior": "ignored_no_truncation",
        "num_workers": args.num_workers,
        "num_original_prepared_samples": len(original_records),
        "num_generated_source_samples": len(generated_samples),
        "num_generated_converted_samples": len(converted_generated_records),
        "num_merged_samples": len(merged_records),
        "original_train_json": str(original_train_json),
        "generated_source": str(generated_source),
        "attack_manifest_dir": str(attack_manifest_dir),
        "num_attack_manifest_records_loaded": len(attack_manifests_by_sample_id),
        **generated_summary,
        "system_prompt_source": "inline_in_prepare_guard_train_SFT_ground2",
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }

    train_output_path = output_dir / TRAIN_FILE_NAME
    dataset_info_output_path = output_dir / DATASET_INFO_FILE_NAME
    build_stats_output_path = output_dir / BUILD_STATS_FILE_NAME

    write_json(train_output_path, merged_records)
    write_json(dataset_info_output_path, dataset_info)
    write_json(build_stats_output_path, build_stats)

    print(f"Saved merged train dataset to: {train_output_path}")
    print(f"Saved dataset_info to: {dataset_info_output_path}")
    print(f"Saved build_stats to: {build_stats_output_path}")
    print(
        "Merged counts: "
        f"original={len(original_records)} generated={len(converted_generated_records)} merged={len(merged_records)}"
    )


if __name__ == "__main__":
    main()
