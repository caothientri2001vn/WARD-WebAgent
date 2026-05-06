#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from guard_attack_utils import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_OPENROUTER_MAX_ATTEMPTS,
    DEFAULT_OPENROUTER_RETRY_BACKOFF_SECONDS,
    DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
    OpenRouterBenignReasoningRewriter,
    SCREENSHOT_PLACEMENTS,
    build_attack_recipe,
    build_attacked_record,
    compact_text,
    normalize_label,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_JSON = ROOT / "train_guard" / "data_new2" / "train_overlayed.json"
DEFAULT_OUTPUT_ROOT = ROOT / "adversarial_attack_data" / "screenshot"
DEFAULT_ATTACK_TARGETS = ("html_only", "screenshot_only", "both")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, min(16, cpu_count))


class JsonArrayWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        self.handle.write("[\n")
        self.is_first = True
        self.closed = False

    def write(self, payload: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError(f"Cannot write to closed JSON array writer: {self.path}")
        if not self.is_first:
            self.handle.write(",\n")
        self.handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
        self.is_first = False

    def close(self) -> None:
        if self.closed:
            return
        if not self.is_first:
            self.handle.write("\n")
        self.handle.write("]\n")
        self.handle.close()
        self.closed = True

    def __enter__(self) -> "JsonArrayWriter":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        self.closed = False

    def write(self, payload: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError(f"Cannot write to closed JSONL writer: {self.path}")
        self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.closed:
            return
        self.handle.close()
        self.closed = True

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()


def process_selected_sample(
    sample: dict[str, Any],
    *,
    attack_target: str,
    output_root: Path,
    attack_seed: int,
    screenshot_placement: str,
    html_min_repeat: int,
    html_max_repeat: int,
    max_html_chars: int,
    benign_reasoning_rewriter: OpenRouterBenignReasoningRewriter | None,
) -> dict[str, Any]:
    sample_id = compact_text(sample.get("sample_id"))
    recipe = None
    try:
        recipe = build_attack_recipe(
            sample,
            attack_target=attack_target,
            attack_seed=attack_seed,
            html_min_repeat=html_min_repeat,
            html_max_repeat=html_max_repeat,
            screenshot_placement=screenshot_placement,
        )
        attacked_record, manifest_record = build_attacked_record(
            sample,
            recipe=recipe,
            output_root=output_root,
            max_html_chars=max_html_chars,
            benign_reasoning_rewriter=benign_reasoning_rewriter,
        )
    except Exception as exc:
        return {
            "ok": False,
            "sample": sample,
            "recipe": recipe,
            "error": {
                "sample_id": sample_id,
                "attack_target": attack_target,
                "source_label": normalize_label(sample.get("label")),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "output_sample_id": recipe.get("output_sample_id") if isinstance(recipe, dict) else None,
                "config_hash": recipe.get("config_hash") if isinstance(recipe, dict) else None,
                "failed_at": now_iso(),
                "traceback": traceback.format_exc(),
            },
        }
    return {
        "ok": True,
        "sample": sample,
        "recipe": recipe,
        "attacked_record": attacked_record,
        "manifest_record": manifest_record,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Guard adversarial dataset from train_overlayed.json by sampling benign and malicious "
            "source records, then materializing html-only, screenshot-only, and both-mode prompt injections."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-count-per-class", type=int, default=21000)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--attack-seed", type=int, default=42)
    parser.add_argument(
        "--attack-targets",
        nargs="+",
        choices=list(DEFAULT_ATTACK_TARGETS),
        default=list(DEFAULT_ATTACK_TARGETS),
        help="One or more attack targets to generate.",
    )
    parser.add_argument(
        "--screenshot-placement",
        choices=list(SCREENSHOT_PLACEMENTS),
        default="random",
        help="Placement strategy for screenshot text injection.",
    )
    parser.add_argument(
        "--max-html-chars",
        type=int,
        default=0,
        help=(
            "Maximum number of processed HTML characters to consider when computing HTML attack metadata. "
            "Use 0 to disable truncation."
        ),
    )
    parser.add_argument("--html-min-repeat", type=int, default=1)
    parser.add_argument("--html-max-repeat", type=int, default=40)
    parser.add_argument(
        "--rewrite-benign-reasoning-via-openrouter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, benign-source samples have their final reasoning rewritten through OpenRouter "
            "so the text no longer claims the webpage is benign or legitimate."
        ),
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default=DEFAULT_OPENROUTER_MODEL,
        help="OpenRouter model used for benign reasoning rewrite.",
    )
    parser.add_argument(
        "--openrouter-timeout-seconds",
        type=float,
        default=DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
        help="Request timeout for the benign reasoning OpenRouter rewrite call.",
    )
    parser.add_argument(
        "--openrouter-cache-path",
        type=Path,
        default=None,
        help="Optional JSON cache path for benign reasoning rewrites.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of samples to process in parallel for each attack target.",
    )
    parser.add_argument("--log-every", type=int, default=250)
    return parser.parse_args()


def load_source_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a top-level JSON list in {path}, got {type(payload).__name__}.")

    records: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            records.append(item)
    return records


def select_source_records(
    records: list[dict[str, Any]],
    *,
    sample_count_per_class: int,
    selection_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    benign_records = [record for record in records if normalize_label(record.get("label")) == "benign"]
    malicious_records = [record for record in records if normalize_label(record.get("label")) == "malicious"]

    if sample_count_per_class > len(benign_records):
        raise ValueError(
            f"Requested {sample_count_per_class} benign samples, but only {len(benign_records)} are available."
        )
    if sample_count_per_class > len(malicious_records):
        raise ValueError(
            f"Requested {sample_count_per_class} malicious samples, but only {len(malicious_records)} are available."
        )

    rng = random.Random(selection_seed)
    selected_benign = rng.sample(benign_records, sample_count_per_class)
    selected_malicious = rng.sample(malicious_records, sample_count_per_class)
    return {
        "benign": selected_benign,
        "malicious": selected_malicious,
    }


def partition_records_evenly(
    records: list[dict[str, Any]],
    *,
    part_count: int,
    label_name: str,
) -> list[list[dict[str, Any]]]:
    if part_count < 1:
        raise ValueError("part_count must be >= 1")
    if len(records) % part_count != 0:
        raise ValueError(
            f"Selected {label_name} sample count {len(records)} is not divisible by the number of attack targets "
            f"({part_count})."
        )

    part_size = len(records) // part_count
    return [records[index * part_size : (index + 1) * part_size] for index in range(part_count)]


def assign_selected_samples_to_targets(
    selected: dict[str, list[dict[str, Any]]],
    *,
    attack_targets: list[str],
) -> dict[str, list[dict[str, Any]]]:
    benign_parts = partition_records_evenly(
        selected["benign"],
        part_count=len(attack_targets),
        label_name="benign",
    )
    malicious_parts = partition_records_evenly(
        selected["malicious"],
        part_count=len(attack_targets),
        label_name="malicious",
    )

    assigned: dict[str, list[dict[str, Any]]] = {}
    for index, attack_target in enumerate(attack_targets):
        assigned[attack_target] = [*benign_parts[index], *malicious_parts[index]]
    return assigned


def dataset_output_path(output_root: Path, dataset_stem: str, attack_target: str) -> Path:
    return output_root / "datasets" / f"{dataset_stem}.guard_attack.{attack_target}.json"


def manifest_output_path(output_root: Path, dataset_stem: str, attack_target: str) -> Path:
    return output_root / "manifests" / f"{dataset_stem}.guard_attack.{attack_target}.jsonl"


def failure_output_path(output_root: Path, dataset_stem: str, attack_target: str) -> Path:
    return output_root / "manifests" / f"{dataset_stem}.guard_attack.{attack_target}.failures.jsonl"


def aggregate_generated_output_path(output_root: Path, dataset_stem: str) -> Path:
    return output_root / "datasets" / f"{dataset_stem}.guard_attack.generated_only.json"


def aggregate_with_source_output_path(output_root: Path, dataset_stem: str) -> Path:
    return output_root / "datasets" / f"{dataset_stem}.guard_attack.with_source.json"


def summarize_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def build_selection_manifest(
    *,
    input_json: Path,
    total_input_samples: int,
    selected: dict[str, list[dict[str, Any]]],
    sample_count_per_class: int,
    selection_seed: int,
) -> dict[str, Any]:
    return {
        "input_json": str(input_json),
        "total_input_samples": total_input_samples,
        "sample_count_per_class": sample_count_per_class,
        "selection_seed": selection_seed,
        "selected_benign_sample_ids": [compact_text(sample.get("sample_id")) for sample in selected["benign"]],
        "selected_malicious_sample_ids": [
            compact_text(sample.get("sample_id")) for sample in selected["malicious"]
        ],
    }


def process_attack_target(
    *,
    attack_target: str,
    selected_samples: list[dict[str, Any]],
    output_root: Path,
    dataset_stem: str,
    attack_seed: int,
    screenshot_placement: str,
    max_html_chars: int,
    html_min_repeat: int,
    html_max_repeat: int,
    log_every: int,
    workers: int,
    benign_reasoning_rewriter: OpenRouterBenignReasoningRewriter | None = None,
    aggregate_generated_writer: JsonArrayWriter | None = None,
    aggregate_with_source_writer: JsonArrayWriter | None = None,
) -> dict[str, Any]:
    dataset_path = dataset_output_path(output_root, dataset_stem, attack_target)
    manifest_path = manifest_output_path(output_root, dataset_stem, attack_target)
    failure_path = failure_output_path(output_root, dataset_stem, attack_target)

    prompt_template_counts: Counter[str] = Counter()
    source_label_counts: Counter[str] = Counter()
    source_location_counts: Counter[str] = Counter()
    final_location_counts: Counter[str] = Counter()
    screenshot_mode_counts: Counter[str] = Counter()
    screenshot_cache_counts: Counter[str] = Counter()
    html_position_counts: Counter[str] = Counter()
    html_repeat_counts: Counter[str] = Counter()
    reasoning_rewrite_counts: Counter[str] = Counter()
    failure_type_counts: Counter[str] = Counter()

    started_at = time.time()
    started_at_iso = now_iso()
    total_samples = len(selected_samples)
    worker_count = max(1, workers)
    written_records = 0
    skipped_records = 0

    with (
        JsonArrayWriter(dataset_path) as dataset_writer,
        JsonlWriter(manifest_path) as manifest_writer,
        JsonlWriter(failure_path) as failure_writer,
    ):
        worker_fn = partial(
            process_selected_sample,
            attack_target=attack_target,
            output_root=output_root,
            attack_seed=attack_seed,
            screenshot_placement=screenshot_placement,
            html_min_repeat=html_min_repeat,
            html_max_repeat=html_max_repeat,
            max_html_chars=max_html_chars,
            benign_reasoning_rewriter=benign_reasoning_rewriter,
        )
        if worker_count == 1:
            result_iter = map(worker_fn, selected_samples)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"guard-{attack_target}")
            result_iter = executor.map(worker_fn, selected_samples)
        try:
            for index, result in enumerate(result_iter, start=1):
                if not result.get("ok", False):
                    skipped_records += 1
                    failure = result["error"]
                    failure_writer.write(failure)
                    failure_type_counts[failure["error_type"]] += 1
                    print(
                        f"[{attack_target}] skipped sample_id={failure['sample_id']} after retries: "
                        f"{failure['error_message']}"
                    )
                    if log_every > 0 and (index == 1 or index % log_every == 0 or index == total_samples):
                        elapsed = time.time() - started_at
                        rate = index / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[{attack_target}] processed {index}/{total_samples} samples "
                            f"({rate:.2f} samples/s, workers={worker_count}, "
                            f"written={written_records}, skipped={skipped_records})"
                        )
                    continue

                sample = result["sample"]
                recipe = result["recipe"]
                attacked_record = result["attacked_record"]
                manifest_record = result["manifest_record"]

                dataset_writer.write(attacked_record)
                manifest_writer.write(manifest_record)
                written_records += 1
                if aggregate_generated_writer is not None:
                    aggregate_generated_writer.write(attacked_record)
                if aggregate_with_source_writer is not None:
                    aggregate_with_source_writer.write(attacked_record)

                prompt_template_counts[recipe["prompt_template_id"]] += 1
                source_label_counts[manifest_record["source_label"]] += 1
                source_location_counts[compact_text(sample.get("injection_location")).lower() or "none"] += 1
                final_location_counts[attacked_record["injection_location"]] += 1

                screenshot_mode = manifest_record["screenshot_attack"].get("mode")
                if screenshot_mode:
                    screenshot_mode_counts[screenshot_mode] += 1
                elif attack_target == "html_only":
                    screenshot_mode_counts["original_copy"] += 1

                if manifest_record["screenshot_attack"].get("cached_before_run") is True:
                    screenshot_cache_counts["reused"] += 1
                else:
                    screenshot_cache_counts["created"] += 1

                if manifest_record["reasoning_rewrite"].get("applied"):
                    if manifest_record["reasoning_rewrite"].get("cached_before_run") is True:
                        reasoning_rewrite_counts["reused"] += 1
                    else:
                        reasoning_rewrite_counts["created"] += 1
                else:
                    reasoning_rewrite_counts["not_applied"] += 1

                if manifest_record["html_attack"].get("applied"):
                    html_position_counts[manifest_record["html_attack"]["requested_position"]] += 1
                    html_repeat_counts[str(manifest_record["html_attack"]["repeat"])] += 1
                else:
                    html_position_counts["not_applied"] += 1
                    html_repeat_counts["not_applied"] += 1

                if log_every > 0 and (index == 1 or index % log_every == 0 or index == total_samples):
                    elapsed = time.time() - started_at
                    rate = index / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{attack_target}] processed {index}/{total_samples} samples "
                        f"({rate:.2f} samples/s, workers={worker_count}, "
                        f"written={written_records}, skipped={skipped_records})"
                    )
        finally:
            if worker_count != 1:
                executor.shutdown(wait=True)

    return {
        "attack_target": attack_target,
        "started_at": started_at_iso,
        "finished_at": now_iso(),
        "num_requested_samples": total_samples,
        "num_records": written_records,
        "num_skipped_samples": skipped_records,
        "dataset_json": str(dataset_path),
        "manifest_jsonl": str(manifest_path),
        "failure_jsonl": str(failure_path),
        "workers": worker_count,
        "source_label_counts": summarize_counter(source_label_counts),
        "source_injection_location_counts": summarize_counter(source_location_counts),
        "final_injection_location_counts": summarize_counter(final_location_counts),
        "prompt_template_counts": summarize_counter(prompt_template_counts),
        "html_position_counts": summarize_counter(html_position_counts),
        "html_repeat_counts": summarize_counter(html_repeat_counts),
        "screenshot_mode_counts": summarize_counter(screenshot_mode_counts),
        "screenshot_cache_counts": summarize_counter(screenshot_cache_counts),
        "reasoning_rewrite_counts": summarize_counter(reasoning_rewrite_counts),
        "failure_type_counts": summarize_counter(failure_type_counts),
    }


def main() -> None:
    args = parse_args()

    input_json = args.input_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.sample_count_per_class < 1:
        raise ValueError("--sample-count-per-class must be >= 1")
    if args.max_html_chars < 0:
        raise ValueError("--max-html-chars must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    started_at_iso = now_iso()
    source_records = load_source_records(input_json)
    selected = select_source_records(
        source_records,
        sample_count_per_class=args.sample_count_per_class,
        selection_seed=args.selection_seed,
    )
    samples_by_target = assign_selected_samples_to_targets(
        selected,
        attack_targets=args.attack_targets,
    )
    selected_samples = [*selected["benign"], *selected["malicious"]]
    dataset_stem = input_json.stem
    benign_reasoning_rewriter = None
    if args.rewrite_benign_reasoning_via_openrouter:
        openrouter_cache_path = args.openrouter_cache_path
        if openrouter_cache_path is None:
            openrouter_cache_path = (
                output_root / "manifests" / f"{dataset_stem}.guard_attack.benign_reasoning_openrouter_cache.json"
            )
        benign_reasoning_rewriter = OpenRouterBenignReasoningRewriter(
            model=args.openrouter_model,
            cache_path=openrouter_cache_path,
            timeout_seconds=args.openrouter_timeout_seconds,
            max_attempts=DEFAULT_OPENROUTER_MAX_ATTEMPTS,
            retry_backoff_seconds=DEFAULT_OPENROUTER_RETRY_BACKOFF_SECONDS,
        )

    selection_manifest = build_selection_manifest(
        input_json=input_json,
        total_input_samples=len(source_records),
        selected=selected,
        sample_count_per_class=args.sample_count_per_class,
        selection_seed=args.selection_seed,
    )
    write_json(output_root / "manifests" / f"{input_json.stem}.guard_attack.selection.json", selection_manifest)

    target_summaries = []
    aggregate_generated_path = aggregate_generated_output_path(output_root, dataset_stem)
    aggregate_with_source_path = aggregate_with_source_output_path(output_root, dataset_stem)
    try:
        with (
            JsonArrayWriter(aggregate_generated_path) as aggregate_generated_writer,
            JsonArrayWriter(aggregate_with_source_path) as aggregate_with_source_writer,
        ):
            for sample in selected_samples:
                aggregate_with_source_writer.write(sample)

            for attack_target in args.attack_targets:
                target_summary = process_attack_target(
                    attack_target=attack_target,
                    selected_samples=samples_by_target[attack_target],
                    output_root=output_root,
                    dataset_stem=dataset_stem,
                    attack_seed=args.attack_seed,
                    screenshot_placement=args.screenshot_placement,
                    max_html_chars=args.max_html_chars,
                    html_min_repeat=args.html_min_repeat,
                    html_max_repeat=args.html_max_repeat,
                    log_every=args.log_every,
                    workers=args.workers,
                    benign_reasoning_rewriter=benign_reasoning_rewriter,
                    aggregate_generated_writer=aggregate_generated_writer,
                    aggregate_with_source_writer=aggregate_with_source_writer,
                )
                target_summaries.append(target_summary)
    finally:
        if benign_reasoning_rewriter is not None:
            benign_reasoning_rewriter.close()

    num_generated_records = sum(summary["num_records"] for summary in target_summaries)
    num_with_source_records = len(selected_samples) + num_generated_records

    build_summary = {
        "started_at": started_at_iso,
        "finished_at": now_iso(),
        "config": {
            "input_json": str(input_json),
            "output_root": str(output_root),
            "sample_count_per_class": args.sample_count_per_class,
            "selection_seed": args.selection_seed,
            "attack_seed": args.attack_seed,
            "attack_targets": args.attack_targets,
            "screenshot_placement": args.screenshot_placement,
            "max_html_chars": args.max_html_chars,
            "html_min_repeat": args.html_min_repeat,
            "html_max_repeat": args.html_max_repeat,
            "workers": args.workers,
            "rewrite_benign_reasoning_via_openrouter": args.rewrite_benign_reasoning_via_openrouter,
            "openrouter_model": args.openrouter_model if args.rewrite_benign_reasoning_via_openrouter else None,
            "openrouter_timeout_seconds": (
                args.openrouter_timeout_seconds if args.rewrite_benign_reasoning_via_openrouter else None
            ),
            "openrouter_max_attempts": (
                DEFAULT_OPENROUTER_MAX_ATTEMPTS if args.rewrite_benign_reasoning_via_openrouter else None
            ),
            "openrouter_retry_backoff_seconds": (
                DEFAULT_OPENROUTER_RETRY_BACKOFF_SECONDS
                if args.rewrite_benign_reasoning_via_openrouter
                else None
            ),
            "openrouter_cache_path": (
                str(args.openrouter_cache_path.expanduser().resolve())
                if args.rewrite_benign_reasoning_via_openrouter and args.openrouter_cache_path is not None
                else None
            ),
        },
        "selection": {
            "total_input_samples": len(source_records),
            "selected_benign": len(selected["benign"]),
            "selected_malicious": len(selected["malicious"]),
            "selected_total": len(selected_samples),
            "per_attack_target": {
                attack_target: {
                    "num_records": len(samples_by_target[attack_target]),
                    "num_benign": sum(
                        1 for sample in samples_by_target[attack_target] if normalize_label(sample.get("label")) == "benign"
                    ),
                    "num_malicious": sum(
                        1
                        for sample in samples_by_target[attack_target]
                        if normalize_label(sample.get("label")) == "malicious"
                    ),
                }
                for attack_target in args.attack_targets
            },
        },
        "aggregate_outputs": {
            "generated_only_json": str(aggregate_generated_path),
            "generated_only_num_records": num_generated_records,
            "with_source_json": str(aggregate_with_source_path),
            "with_source_num_records": num_with_source_records,
        },
        "outputs": target_summaries,
    }
    write_json(output_root / "manifests" / f"{input_json.stem}.guard_attack.summary.json", build_summary)

    print("Finished building Guard attack datasets.")
    print(f"- aggregate generated only: {aggregate_generated_path}")
    print(f"- aggregate with source: {aggregate_with_source_path}")
    for target_summary in target_summaries:
        print(f"- {target_summary['attack_target']}: {target_summary['dataset_json']}")


if __name__ == "__main__":
    main()
