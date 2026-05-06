#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import mimetypes
import os
import random
import re
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import httpx

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: beautifulsoup4. Install it with `pip install beautifulsoup4`."
    ) from exc

from guard_attack_utils import (
    DEFAULT_OPENROUTER_MAX_ATTEMPTS,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_OPENROUTER_RETRY_BACKOFF_SECONDS,
    DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
    OPENROUTER_URL,
    SCREENSHOT_PLACEMENTS,
    build_attack_recipe,
    build_attacked_record,
    compact_text,
    describe_openrouter_error,
    ensure_sentence,
    extract_openrouter_message_text,
    normalize_label,
    preview_for_error,
    read_openrouter_api_key,
    stable_hash,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_JSON = ROOT / "train_guard" / "data_new2" / "train_overlayed.json"
DEFAULT_OUTPUT_ROOT = ROOT / "adversarial_attack_data" / "screenshot_malicious_reasoned"
DEFAULT_ATTACK_TARGETS = ("html_only", "screenshot_only", "both")
BUILDER_VARIANT = "malicious_only_reasoned"

VISIBLE_TEXT_FRAGMENT_PATTERN = re.compile(
    (
        r"<(?P<tag>title|h[1-6]|p|div|span|button|a|li|label|textarea|option|footer|header|main|"
        r"section|article|aside|nav|form)\b[^>]*>(?P<content>.*?)</(?P=tag)>"
    ),
    flags=re.IGNORECASE | re.DOTALL,
)
MAX_RECOVERED_FRAGMENT_CHARS = 1200
MAX_RECOVERED_FRAGMENTS = 20

GROUND_TRUTH_REASONING_SYSTEM_PROMPT = """
You are a security reasoning model for prompt-injection detection in web-agent environments.

You will be given four sources of information:
1. User intended task
2. Preprocessed HTML text extracted from the webpage
3. A screenshot of the webpage
4. Authoritative ground-truth supervision for the final answer: label, injection location, and attack goal

Your task is to write the ground-truth reasoning for the sample.

Instructions:
- Carefully analyze the HTML and the screenshot in relation to the user's intended task.
- Describe the injected or malicious content concretely and explain why it is malicious.
- Explain where the malicious content appears: HTML, screenshot, or both.
- Use the provided ground truth as authoritative. Do not contradict it.
- The reasoning must justify the provided ground truth using evidence from the HTML and screenshot, not merely restate it.
- Return valid JSON only.
- Use exactly the provided ground-truth values for `label`, `injection_location`, and `attack_goal`.

Return exactly this schema:

{
  "reasoning": "detailed, evidence-based explanation",
  "attack_goal": "ground-truth attack goal sentence",
  "injection_location": "html" or "screenshot" or "both" or "none",
  "label": "malicious" or "benign"
}
""".strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, min(16, cpu_count))


def raw_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


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


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/png"
    with image_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def unwrap_prediction_object(obj: dict[str, Any]) -> dict[str, Any]:
    current = obj
    for key in ("prediction", "result", "output", "data"):
        nested = current.get(key)
        if isinstance(nested, dict):
            current = nested
            break
    return current


def pick_first_value(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key not in obj:
            continue
        value = obj.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return ""


def extract_json_from_text(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", candidate)
        candidate = re.sub(r"\n?```$", "", candidate).strip()

    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload

    raise ValueError(f"Cannot parse JSON from model output: {preview_for_error(candidate)}")


def normalize_injection_location(value: Any) -> str:
    location = compact_text(value).lower()
    if location in {"html", "screenshot", "both", "none"}:
        return location
    return "none"


def normalize_model_label(value: Any) -> str:
    label = compact_text(value).lower()
    if label == "malicious":
        return "malicious"
    if label == "benign":
        return "benign"
    return ""


def coerce_reasoning_prediction(obj: dict[str, Any]) -> dict[str, str]:
    obj = unwrap_prediction_object(obj)
    label = compact_text(pick_first_value(obj, ("label", "classification")))
    if not label and "is_malicious" in obj:
        label = "malicious" if bool(obj.get("is_malicious")) else "benign"

    return {
        "reasoning": compact_text(pick_first_value(obj, ("reasoning", "analysis", "explanation"))),
        "attack_goal": ensure_sentence(pick_first_value(obj, ("attack_goal", "goal", "intent"))),
        "injection_location": normalize_injection_location(
            pick_first_value(obj, ("injection_location", "injection_modality", "location", "injected_location"))
        ),
        "label": normalize_model_label(label),
    }


def build_html_soup(raw_html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(raw_html, "lxml")
    except Exception:
        return BeautifulSoup(raw_html, "html.parser")


def normalize_multiline_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def extract_primary_html_text(raw_html: str) -> str:
    soup = build_html_soup(raw_html)
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return normalize_multiline_text(soup.get_text("\n", strip=True))


def should_keep_recovered_fragment(fragment_text: str, primary_text: str, seen_fragments: set[str]) -> bool:
    fragment_compact = compact_text(fragment_text)
    if not fragment_compact:
        return False
    if len(fragment_compact) > MAX_RECOVERED_FRAGMENT_CHARS:
        return False
    if fragment_compact in seen_fragments:
        return False

    letters = sum(char.isalpha() for char in fragment_compact)
    spaces = fragment_compact.count(" ")
    if letters < 20 or spaces < 3:
        return False

    primary_compact = compact_text(primary_text)
    if primary_compact and fragment_compact in primary_compact:
        return False
    return True


def recover_text_fragments_from_malformed_html(raw_html: str, primary_text: str) -> list[str]:
    recovered_fragments: list[str] = []
    seen_fragments = {compact_text(primary_text)} if primary_text else set()

    for match in VISIBLE_TEXT_FRAGMENT_PATTERN.finditer(raw_html):
        fragment_html = match.group(0)
        fragment_text = extract_primary_html_text(fragment_html)
        if not should_keep_recovered_fragment(
            fragment_text=fragment_text,
            primary_text=primary_text,
            seen_fragments=seen_fragments,
        ):
            continue

        seen_fragments.add(compact_text(fragment_text))
        recovered_fragments.append(fragment_text)
        if len(recovered_fragments) >= MAX_RECOVERED_FRAGMENTS:
            break

    return recovered_fragments


def process_html(raw_html: str) -> str:
    primary_text = extract_primary_html_text(raw_html)
    recovered_fragments = recover_text_fragments_from_malformed_html(
        raw_html=raw_html,
        primary_text=primary_text,
    )
    if recovered_fragments:
        return "\n".join([primary_text, *recovered_fragments]).strip()
    return primary_text


def build_html_attack_text(prompt_text: str, repeat: int) -> str:
    if repeat < 1:
        raise ValueError("HTML attack repeat must be >= 1.")
    return "\n\n".join(prompt_text for _ in range(repeat))


def clamp_insert_index(text: str, insert_index: int) -> int:
    return max(0, min(int(insert_index), len(text)))


def build_reasoning_messages(
    *,
    user_task: str,
    processed_html: str,
    screenshot_path: Path,
    ground_truth: dict[str, str],
    top_level_attack_goal: str,
) -> list[dict[str, Any]]:
    text_block = f"""
Below is the available evidence and the authoritative ground truth.

[USER INTENDED TASK]
{user_task or "Not provided"}

[HTML TEXT]
{processed_html or "Not provided"}

[GROUND TRUTH]
label: {ground_truth["label"]}
injection_location: {ground_truth["injection_location"]}
attack_goal: {ground_truth["attack_goal"]}
top_level_attack_goal: {top_level_attack_goal or "Not provided"}

Use the ground truth above as authoritative for the final answer.
Return JSON only.
""".strip()

    content: list[dict[str, Any]] = [{"type": "text", "text": text_block}]
    image_data_url = image_to_data_url(screenshot_path)
    if image_data_url:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    return [
        {"role": "system", "content": GROUND_TRUTH_REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def load_source_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a top-level JSON list in {path}, got {type(payload).__name__}.")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"{path} contains non-dict records.")
    return payload


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
    return {
        "benign": rng.sample(benign_records, sample_count_per_class),
        "malicious": rng.sample(malicious_records, sample_count_per_class),
    }


def partition_records_evenly(
    records: list[dict[str, Any]],
    *,
    part_count: int,
    label_name: str,
) -> list[list[dict[str, Any]]]:
    if part_count < 1:
        raise ValueError("part_count must be >= 1.")
    if len(records) % part_count != 0:
        raise ValueError(
            f"Selected {label_name} sample count {len(records)} is not divisible by the number of attack targets "
            f"({part_count})."
        )
    part_size = len(records) // part_count
    return [records[index * part_size : (index + 1) * part_size] for index in range(part_count)]


def assign_selected_samples_to_targets(
    selected_records: dict[str, list[dict[str, Any]]],
    *,
    attack_targets: list[str],
) -> dict[str, list[dict[str, Any]]]:
    benign_parts = partition_records_evenly(
        selected_records["benign"],
        part_count=len(attack_targets),
        label_name="benign",
    )
    malicious_parts = partition_records_evenly(
        selected_records["malicious"],
        part_count=len(attack_targets),
        label_name="malicious",
    )
    assigned: dict[str, list[dict[str, Any]]] = {}
    for index, attack_target in enumerate(attack_targets):
        assigned[attack_target] = [*benign_parts[index], *malicious_parts[index]]
    return assigned


def selection_manifest_path(output_root: Path, dataset_stem: str) -> Path:
    return output_root / "manifests" / f"{dataset_stem}.guard_attack.{BUILDER_VARIANT}.selection.json"


def summary_output_path(output_root: Path, dataset_stem: str) -> Path:
    return output_root / "manifests" / f"{dataset_stem}.guard_attack.{BUILDER_VARIANT}.summary.json"


def dataset_output_path(output_root: Path, dataset_stem: str, attack_target: str) -> Path:
    return output_root / "datasets" / f"{dataset_stem}.guard_attack.{BUILDER_VARIANT}.{attack_target}.json"


def manifest_output_path(output_root: Path, dataset_stem: str, attack_target: str) -> Path:
    return output_root / "manifests" / f"{dataset_stem}.guard_attack.{BUILDER_VARIANT}.{attack_target}.jsonl"


def failure_output_path(output_root: Path, dataset_stem: str, attack_target: str) -> Path:
    return output_root / "manifests" / f"{dataset_stem}.guard_attack.{BUILDER_VARIANT}.{attack_target}.failures.jsonl"


def aggregate_generated_output_path(output_root: Path, dataset_stem: str) -> Path:
    return output_root / "datasets" / f"{dataset_stem}.guard_attack.{BUILDER_VARIANT}.generated_only.json"


def aggregate_with_source_output_path(output_root: Path, dataset_stem: str) -> Path:
    return output_root / "datasets" / f"{dataset_stem}.guard_attack.{BUILDER_VARIANT}.with_source.json"


def summarize_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def build_selection_manifest(
    *,
    input_json: Path,
    total_input_samples: int,
    selected_records: dict[str, list[dict[str, Any]]],
    samples_by_target: dict[str, list[dict[str, Any]]],
    sample_count_per_class: int,
    selection_seed: int,
) -> dict[str, Any]:
    return {
        "builder_variant": BUILDER_VARIANT,
        "input_json": str(input_json),
        "total_input_samples": total_input_samples,
        "sample_count_per_class": sample_count_per_class,
        "selection_seed": selection_seed,
        "selected_benign_sample_ids": [compact_text(sample.get("sample_id")) for sample in selected_records["benign"]],
        "selected_malicious_sample_ids": [
            compact_text(sample.get("sample_id")) for sample in selected_records["malicious"]
        ],
        "per_attack_target_sample_ids": {
            attack_target: [compact_text(sample.get("sample_id")) for sample in samples]
            for attack_target, samples in samples_by_target.items()
        },
    }


def build_variant_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    variant_recipe = dict(recipe)
    source_sample_id = compact_text(variant_recipe.get("source_sample_id"))
    attack_target = compact_text(variant_recipe.get("attack_target"))
    identity_payload = {
        "builder_variant": BUILDER_VARIANT,
        "source_sample_id": source_sample_id,
        "attack_target": attack_target,
        "prompt_template_id": compact_text(variant_recipe.get("prompt_template_id")),
        "prompt_text_hash": stable_hash(raw_text(variant_recipe.get("prompt_text")), length=24),
        "html_repeat": variant_recipe.get("html_repeat"),
        "html_position": compact_text(variant_recipe.get("html_position")),
        "html_seed": variant_recipe.get("html_seed"),
        "screenshot_mode": compact_text(variant_recipe.get("screenshot_mode")),
        "screenshot_placement": compact_text(variant_recipe.get("screenshot_placement")),
        "screenshot_seed": variant_recipe.get("screenshot_seed"),
        "attack_seed": variant_recipe.get("attack_seed"),
    }
    variant_recipe["builder_variant"] = BUILDER_VARIANT
    variant_recipe["config_hash"] = stable_hash(identity_payload)
    variant_recipe["output_sample_id"] = (
        f"{source_sample_id}__guardattack_{BUILDER_VARIANT}_{attack_target}_{variant_recipe['config_hash']}"
    )
    return variant_recipe


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class OpenRouterGroundTruthReasoner:
    def __init__(
        self,
        *,
        model: str,
        cache_path: Path,
        timeout_seconds: float,
        max_attempts: int,
        retry_backoff_seconds: float,
        max_tokens: int,
        flush_every: int = 25,
    ):
        api_key = read_openrouter_api_key()
        if not api_key:
            raise RuntimeError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY or create "
                "`openrounter_key.txt` / `openrouter_key.txt` in the repo root."
            )

        self.api_key = api_key
        self.model = compact_text(model) or DEFAULT_OPENROUTER_MODEL
        self.cache_path = cache_path.expanduser().resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = max(1, int(max_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.max_tokens = max(256, int(max_tokens))
        self.flush_every = max(1, int(flush_every))

        self.cache = self._load_cache()
        self._cache_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._clients: dict[int, httpx.Client] = {}
        self._serial_retry_condition = threading.Condition()
        self._serial_retry_owner: int | None = None
        self._dirty_updates = 0

    def close(self) -> None:
        self._flush_cache(force=True)
        with self._client_lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.is_file():
            return {}
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object in {self.cache_path}, got {type(payload).__name__}.")
        return payload

    def _snapshot_cache(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self.cache.items()}

    def _flush_cache(self, *, force: bool) -> None:
        snapshot = None
        with self._cache_lock:
            if not force and self._dirty_updates < self.flush_every:
                return
            if not force and self._dirty_updates <= 0:
                return
            if force and self._dirty_updates <= 0 and not self.cache_path.exists():
                return
            snapshot = self._snapshot_cache()
            self._dirty_updates = 0
        if snapshot is not None:
            write_json(self.cache_path, snapshot)

    def _get_client(self) -> httpx.Client:
        thread_id = threading.get_ident()
        with self._client_lock:
            client = self._clients.get(thread_id)
            if client is None:
                client = httpx.Client(
                    timeout=httpx.Timeout(
                        self.timeout_seconds,
                        connect=min(self.timeout_seconds, 15.0),
                        read=self.timeout_seconds,
                        write=self.timeout_seconds,
                        pool=min(self.timeout_seconds, 15.0),
                    )
                )
                self._clients[thread_id] = client
            return client

    def _wait_for_retry_window(self) -> None:
        thread_id = threading.get_ident()
        with self._serial_retry_condition:
            while self._serial_retry_owner is not None and self._serial_retry_owner != thread_id:
                self._serial_retry_condition.wait()

    def _enter_serial_retry_mode(self) -> None:
        thread_id = threading.get_ident()
        with self._serial_retry_condition:
            while self._serial_retry_owner is not None and self._serial_retry_owner != thread_id:
                self._serial_retry_condition.wait()
            self._serial_retry_owner = thread_id

    def _exit_serial_retry_mode(self) -> None:
        thread_id = threading.get_ident()
        with self._serial_retry_condition:
            if self._serial_retry_owner == thread_id:
                self._serial_retry_owner = None
                self._serial_retry_condition.notify_all()

    def _retry_sleep_seconds(self, attempt: int) -> float:
        if attempt <= 1 or self.retry_backoff_seconds <= 0:
            return 0.0
        return min(30.0, self.retry_backoff_seconds * float(attempt - 1))

    def _request_reasoning_once(
        self,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
        sample_id: str,
        attack_target: str,
        attempt: int,
    ) -> dict[str, str]:
        self._wait_for_retry_window()
        try:
            response = self._get_client().post(OPENROUTER_URL, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"OpenRouter request error for sample_id={sample_id}, "
                f"attack_target={attack_target}, attempt={attempt}: {exc}"
            ) from exc

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"OpenRouter returned non-JSON content for sample_id={sample_id}, "
                f"attack_target={attack_target}, attempt={attempt}, status_code={response.status_code}: "
                f"{preview_for_error(response.text)}"
            ) from exc

        if response.status_code >= 400:
            detail = describe_openrouter_error(response_payload) or preview_for_error(response_payload)
            raise RuntimeError(
                f"OpenRouter returned HTTP {response.status_code} for sample_id={sample_id}, "
                f"attack_target={attack_target}, attempt={attempt}: {detail}"
            )

        payload_error = describe_openrouter_error(response_payload)
        if payload_error:
            raise RuntimeError(
                f"OpenRouter returned an error payload for sample_id={sample_id}, "
                f"attack_target={attack_target}, attempt={attempt}: {payload_error}"
            )

        try:
            message_text = extract_openrouter_message_text(response_payload)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc} sample_id={sample_id}, attack_target={attack_target}, attempt={attempt}, "
                f"response_preview={preview_for_error(response_payload)}"
            ) from exc

        parsed = coerce_reasoning_prediction(extract_json_from_text(message_text))
        reasoning = ensure_sentence(parsed.get("reasoning"))
        if not reasoning:
            raise RuntimeError(
                f"OpenRouter returned an empty reasoning field for sample_id={sample_id}, "
                f"attack_target={attack_target}, attempt={attempt}."
            )
        parsed["reasoning"] = reasoning
        return parsed

    def generate(
        self,
        *,
        sample_id: str,
        attack_target: str,
        user_task: str,
        processed_html: str,
        screenshot_path: Path,
        ground_truth: dict[str, str],
        top_level_attack_goal: str,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        processed_html = raw_text(processed_html)
        if not screenshot_path.is_file():
            raise FileNotFoundError(f"Missing screenshot for reasoning generation: {screenshot_path}")

        cache_key = stable_hash(
            {
                "builder_variant": BUILDER_VARIANT,
                "sample_id": sample_id,
                "attack_target": attack_target,
                "model": self.model,
                "user_task_hash": stable_hash(user_task, length=24),
                "processed_html_hash": stable_hash(processed_html, length=24),
                "screenshot": file_fingerprint(screenshot_path),
                "ground_truth": ground_truth,
                "top_level_attack_goal": top_level_attack_goal,
            },
            length=24,
        )
        with self._cache_lock:
            cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            cached_reasoning = compact_text(cached.get("reasoning"))
            if cached_reasoning:
                return {
                    "reasoning": ensure_sentence(cached_reasoning),
                    "attack_goal": ensure_sentence(cached.get("attack_goal")),
                    "injection_location": normalize_injection_location(cached.get("injection_location")),
                    "label": normalize_model_label(cached.get("label")),
                }, {
                    "applied": True,
                    "cached_before_run": True,
                    "model": self.model,
                    "cache_key": cache_key,
                    "cache_path": str(self.cache_path),
                    "attempts_used": 0,
                    "serial_retry_used": False,
                }

        payload = {
            "model": self.model,
            "messages": build_reasoning_messages(
                user_task=user_task,
                processed_html=processed_html,
                screenshot_path=screenshot_path,
                ground_truth=ground_truth,
                top_level_attack_goal=top_level_attack_goal,
            ),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.guard-attack-dataset",
            "X-Title": "Guard reasoning generator",
        }

        serial_retry_used = False
        attempts_used = 0
        last_error: Exception | None = None
        try:
            for attempt in range(1, self.max_attempts + 1):
                attempts_used = attempt
                if attempt > 1:
                    sleep_seconds = self._retry_sleep_seconds(attempt)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                try:
                    prediction = self._request_reasoning_once(
                        payload=payload,
                        headers=headers,
                        sample_id=sample_id,
                        attack_target=attack_target,
                        attempt=attempt,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 1 and self.max_attempts > 1 and not serial_retry_used:
                        serial_retry_used = True
                        self._enter_serial_retry_mode()
                        print(
                            f"[openrouter] sample_id={sample_id} attack_target={attack_target} "
                            f"attempt {attempt}/{self.max_attempts} failed. "
                            f"Switching to serialized retries: {exc}"
                        )
                    else:
                        print(
                            f"[openrouter] sample_id={sample_id} attack_target={attack_target} "
                            f"attempt {attempt}/{self.max_attempts} failed: {exc}"
                        )
                    if attempt >= self.max_attempts:
                        raise RuntimeError(
                            f"OpenRouter reasoning generation failed after {self.max_attempts} attempts "
                            f"for sample_id={sample_id}, attack_target={attack_target}: {exc}"
                        ) from exc
            else:
                raise RuntimeError(
                    f"OpenRouter reasoning generation failed without producing output "
                    f"for sample_id={sample_id}, attack_target={attack_target}."
                )
        finally:
            if serial_retry_used:
                self._exit_serial_retry_mode()

        if last_error is not None and attempts_used > 1:
            print(
                f"[openrouter] sample_id={sample_id} attack_target={attack_target} "
                f"succeeded on attempt {attempts_used}/{self.max_attempts} after serialized retries."
            )

        with self._cache_lock:
            self.cache[cache_key] = {
                "sample_id": sample_id,
                "attack_target": attack_target,
                "model": self.model,
                "reasoning": prediction["reasoning"],
                "attack_goal": prediction["attack_goal"],
                "injection_location": prediction["injection_location"],
                "label": prediction["label"],
                "attempts_used": attempts_used,
                "created_at": now_iso(),
            }
            self._dirty_updates += 1
        self._flush_cache(force=False)
        return prediction, {
            "applied": True,
            "cached_before_run": False,
            "model": self.model,
            "cache_key": cache_key,
            "cache_path": str(self.cache_path),
            "attempts_used": attempts_used,
            "serial_retry_used": serial_retry_used,
        }


def rebuild_attacked_html_for_reasoning(
    *,
    sample_id: str,
    html_path: Path,
    html_attack: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not html_path.is_file():
        raise FileNotFoundError(f"Missing HTML for sample_id={sample_id}: {html_path}")

    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
    processed_html = process_html(raw_html)
    if not isinstance(html_attack, dict) or not html_attack.get("applied"):
        return processed_html, {
            "html_attack_applied_for_reasoning": False,
            "raw_html_chars": len(raw_html),
            "processed_html_chars": len(processed_html),
            "prompt_html_chars": len(processed_html),
            "html_attack_repeat": 0,
            "html_attack_insert_index": 0,
            "html_attack_prompt_text_preview": "",
        }

    prompt_text = raw_text(html_attack.get("prompt_text"))
    repeat = int(html_attack.get("repeat", 0))
    attack_text = build_html_attack_text(prompt_text, repeat)
    insert_index = clamp_insert_index(processed_html, int(html_attack.get("insert_index", 0)))

    stored_original_chars = html_attack.get("original_processed_html_chars")
    if stored_original_chars is not None and int(stored_original_chars) != len(processed_html):
        raise ValueError(
            "Processed HTML length mismatch before reasoning reconstruction for sample "
            f"{sample_id}: current={len(processed_html)} stored={stored_original_chars}"
        )

    prefix = processed_html[:insert_index].strip()
    suffix = processed_html[insert_index:].strip()
    attacked_html = "\n\n".join(part for part in (prefix, attack_text, suffix) if part)

    stored_attacked_chars = html_attack.get("attacked_processed_html_chars")
    if stored_attacked_chars is not None and int(stored_attacked_chars) != len(attacked_html):
        raise ValueError(
            "Attacked HTML length mismatch after reasoning reconstruction for sample "
            f"{sample_id}: current={len(attacked_html)} stored={stored_attacked_chars}"
        )

    return attacked_html, {
        "html_attack_applied_for_reasoning": True,
        "raw_html_chars": len(raw_html),
        "processed_html_chars": len(processed_html),
        "prompt_html_chars": len(attacked_html),
        "html_attack_repeat": repeat,
        "html_attack_insert_index": insert_index,
        "html_attack_prompt_text_preview": compact_text(prompt_text)[:300],
    }


def process_selected_sample(
    sample: dict[str, Any],
    *,
    attack_target: str,
    output_root: Path,
    attack_seed: int,
    screenshot_placement: str,
    html_min_repeat: int,
    html_max_repeat: int,
    reasoning_generator: OpenRouterGroundTruthReasoner,
) -> dict[str, Any]:
    source_sample_id = compact_text(sample.get("sample_id"))
    recipe = None
    try:
        recipe = build_variant_recipe(
            build_attack_recipe(
                sample,
                attack_target=attack_target,
                attack_seed=attack_seed,
                html_min_repeat=html_min_repeat,
                html_max_repeat=html_max_repeat,
                screenshot_placement=screenshot_placement,
            )
        )
        attacked_record, manifest_record = build_attacked_record(
            sample,
            recipe=recipe,
            output_root=output_root,
            max_html_chars=0,
            benign_reasoning_rewriter=None,
        )

        html_path = Path(str(sample.get("html_path", ""))).expanduser().resolve()
        screenshot_path = Path(str(attacked_record.get("screenshot_path", ""))).expanduser().resolve()
        prompt_html, reasoning_html_stats = rebuild_attacked_html_for_reasoning(
            sample_id=recipe["output_sample_id"],
            html_path=html_path,
            html_attack=manifest_record.get("html_attack") if isinstance(manifest_record.get("html_attack"), dict) else {},
        )

        ground_truth_reasoning = attacked_record.get("gt_reasoning") if isinstance(attacked_record.get("gt_reasoning"), dict) else {}
        final_ground_truth = {
            "label": "malicious",
            "injection_location": normalize_injection_location(attacked_record.get("injection_location")),
            "attack_goal": ensure_sentence(ground_truth_reasoning.get("attack_goal")),
        }
        if not final_ground_truth["attack_goal"]:
            raise RuntimeError(
                f"Missing final ground-truth attack goal for sample_id={recipe['output_sample_id']}."
            )

        generated_reasoning, reasoning_generation_info = reasoning_generator.generate(
            sample_id=recipe["output_sample_id"],
            attack_target=attack_target,
            user_task=compact_text(sample.get("user_task")),
            processed_html=prompt_html,
            screenshot_path=screenshot_path,
            ground_truth=final_ground_truth,
            top_level_attack_goal=compact_text(attacked_record.get("attack_goal")),
        )

        attacked_record["gt_reasoning"] = {
            "reasoning": generated_reasoning["reasoning"],
            "attack_goal": final_ground_truth["attack_goal"],
            "injection_location": final_ground_truth["injection_location"],
            "label": final_ground_truth["label"],
        }

        manifest_record["reasoning_rewrite"] = {
            "applied": False,
            "reason": "This builder generates gt_reasoning directly via OpenRouter instead of rewriting text.",
        }
        manifest_record["reasoning_generation"] = {
            **reasoning_generation_info,
            "ground_truth": final_ground_truth,
            "model_output": generated_reasoning,
            "reasoning_input": {
                "user_task_chars": len(compact_text(sample.get("user_task"))),
                "processed_html_chars": int(reasoning_html_stats["processed_html_chars"]),
                "prompt_html_chars": int(reasoning_html_stats["prompt_html_chars"]),
                "screenshot_path": str(screenshot_path),
            },
        }
        manifest_record["reasoning_input_html"] = reasoning_html_stats
    except Exception as exc:
        return {
            "ok": False,
            "sample": sample,
            "recipe": recipe,
            "error": {
                "sample_id": source_sample_id,
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


def process_attack_target(
    *,
    attack_target: str,
    selected_samples: list[dict[str, Any]],
    output_root: Path,
    dataset_stem: str,
    attack_seed: int,
    screenshot_placement: str,
    html_min_repeat: int,
    html_max_repeat: int,
    reasoning_generator: OpenRouterGroundTruthReasoner,
    log_every: int,
    workers: int,
    aggregate_generated_writer: JsonArrayWriter | None = None,
    aggregate_with_source_writer: JsonArrayWriter | None = None,
) -> dict[str, Any]:
    dataset_path = dataset_output_path(output_root, dataset_stem, attack_target)
    manifest_path = manifest_output_path(output_root, dataset_stem, attack_target)
    failure_path = failure_output_path(output_root, dataset_stem, attack_target)

    prompt_template_counts: Counter[str] = Counter()
    source_label_counts: Counter[str] = Counter()
    final_location_counts: Counter[str] = Counter()
    screenshot_mode_counts: Counter[str] = Counter()
    screenshot_cache_counts: Counter[str] = Counter()
    html_position_counts: Counter[str] = Counter()
    html_repeat_counts: Counter[str] = Counter()
    reasoning_cache_counts: Counter[str] = Counter()
    reasoning_output_location_counts: Counter[str] = Counter()
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
            reasoning_generator=reasoning_generator,
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
                source_label_counts[normalize_label(result["sample"].get("label"))] += 1
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

                reasoning_generation = manifest_record.get("reasoning_generation")
                if isinstance(reasoning_generation, dict):
                    if reasoning_generation.get("cached_before_run") is True:
                        reasoning_cache_counts["reused"] += 1
                    else:
                        reasoning_cache_counts["created"] += 1

                    model_output = reasoning_generation.get("model_output")
                    if isinstance(model_output, dict):
                        reasoning_output_location_counts[
                            normalize_injection_location(model_output.get("injection_location"))
                        ] += 1

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
        "prompt_template_counts": summarize_counter(prompt_template_counts),
        "source_label_counts": summarize_counter(source_label_counts),
        "final_injection_location_counts": summarize_counter(final_location_counts),
        "html_position_counts": summarize_counter(html_position_counts),
        "html_repeat_counts": summarize_counter(html_repeat_counts),
        "screenshot_mode_counts": summarize_counter(screenshot_mode_counts),
        "screenshot_cache_counts": summarize_counter(screenshot_cache_counts),
        "reasoning_cache_counts": summarize_counter(reasoning_cache_counts),
        "reasoning_output_location_counts": summarize_counter(reasoning_output_location_counts),
        "failure_type_counts": summarize_counter(failure_type_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Guard adversarial dataset from train_overlayed.json using balanced benign and malicious "
            "source sampling. The script applies one attack target per source sample "
            "and always uses OpenRouter to generate gt_reasoning with ground-truth supervision."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sample-count-per-class",
        type=int,
        default=21000,
        help=(
            "Number of benign source samples and malicious source samples to select. "
            "Each class is split evenly across the requested attack targets."
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
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
    parser.add_argument("--html-min-repeat", type=int, default=1)
    parser.add_argument("--html-max-repeat", type=int, default=40)
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default=DEFAULT_OPENROUTER_MODEL,
        help="OpenRouter model used for reasoning generation.",
    )
    parser.add_argument(
        "--openrouter-timeout-seconds",
        type=float,
        default=DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
        help="Request timeout for the OpenRouter reasoning call.",
    )
    parser.add_argument(
        "--openrouter-max-attempts",
        type=int,
        default=DEFAULT_OPENROUTER_MAX_ATTEMPTS,
        help="Maximum retries for each OpenRouter reasoning request.",
    )
    parser.add_argument(
        "--openrouter-retry-backoff-seconds",
        type=float,
        default=DEFAULT_OPENROUTER_RETRY_BACKOFF_SECONDS,
        help="Linear backoff applied between OpenRouter retries.",
    )
    parser.add_argument(
        "--openrouter-max-tokens",
        type=int,
        default=1400,
        help="max_tokens sent to OpenRouter for each reasoning request.",
    )
    parser.add_argument(
        "--openrouter-cache-path",
        type=Path,
        default=None,
        help="Optional JSON cache path for reasoning generations.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of samples to process in parallel for each attack target.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_json = args.input_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sample_count_per_class = (
        int(args.sample_count)
        if args.sample_count is not None
        else int(args.sample_count_per_class)
    )
    if sample_count_per_class < 1:
        raise ValueError("--sample-count-per-class must be >= 1.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    started_at_iso = now_iso()
    source_records = load_source_records(input_json)
    selected_records = select_source_records(
        source_records,
        sample_count_per_class=sample_count_per_class,
        selection_seed=args.selection_seed,
    )
    samples_by_target = assign_selected_samples_to_targets(
        selected_records,
        attack_targets=args.attack_targets,
    )
    selected_source_records = [*selected_records["benign"], *selected_records["malicious"]]
    dataset_stem = input_json.stem

    openrouter_cache_path = args.openrouter_cache_path
    if openrouter_cache_path is None:
        openrouter_cache_path = (
            output_root / "manifests" / f"{dataset_stem}.guard_attack.{BUILDER_VARIANT}.openrouter_cache.json"
        )

    reasoning_generator = OpenRouterGroundTruthReasoner(
        model=args.openrouter_model,
        cache_path=openrouter_cache_path,
        timeout_seconds=args.openrouter_timeout_seconds,
        max_attempts=args.openrouter_max_attempts,
        retry_backoff_seconds=args.openrouter_retry_backoff_seconds,
        max_tokens=args.openrouter_max_tokens,
    )

    selection_manifest = build_selection_manifest(
        input_json=input_json,
        total_input_samples=len(source_records),
        selected_records=selected_records,
        samples_by_target=samples_by_target,
        sample_count_per_class=sample_count_per_class,
        selection_seed=args.selection_seed,
    )
    write_json(selection_manifest_path(output_root, dataset_stem), selection_manifest)

    target_summaries: list[dict[str, Any]] = []
    aggregate_generated_path = aggregate_generated_output_path(output_root, dataset_stem)
    aggregate_with_source_path = aggregate_with_source_output_path(output_root, dataset_stem)
    try:
        with (
            JsonArrayWriter(aggregate_generated_path) as aggregate_generated_writer,
            JsonArrayWriter(aggregate_with_source_path) as aggregate_with_source_writer,
        ):
            for sample in selected_source_records:
                aggregate_with_source_writer.write(sample)

            for attack_target in args.attack_targets:
                target_summary = process_attack_target(
                    attack_target=attack_target,
                    selected_samples=samples_by_target[attack_target],
                    output_root=output_root,
                    dataset_stem=dataset_stem,
                    attack_seed=args.attack_seed,
                    screenshot_placement=args.screenshot_placement,
                    html_min_repeat=args.html_min_repeat,
                    html_max_repeat=args.html_max_repeat,
                    reasoning_generator=reasoning_generator,
                    log_every=args.log_every,
                    workers=args.workers,
                    aggregate_generated_writer=aggregate_generated_writer,
                    aggregate_with_source_writer=aggregate_with_source_writer,
                )
                target_summaries.append(target_summary)
    finally:
        reasoning_generator.close()

    num_generated_records = sum(summary["num_records"] for summary in target_summaries)
    num_with_source_records = len(selected_source_records) + num_generated_records

    build_summary = {
        "builder_variant": BUILDER_VARIANT,
        "started_at": started_at_iso,
        "finished_at": now_iso(),
        "config": {
            "input_json": str(input_json),
            "output_root": str(output_root),
            "sample_count_per_class": sample_count_per_class,
            "selection_seed": args.selection_seed,
            "attack_seed": args.attack_seed,
            "attack_targets": args.attack_targets,
            "screenshot_placement": args.screenshot_placement,
            "html_min_repeat": args.html_min_repeat,
            "html_max_repeat": args.html_max_repeat,
            "workers": args.workers,
            "html_processing_source": (
                "Copied from train_guard/llamafactory_guard/prepare_guard_train_SFT_ground2.py "
                "to keep reasoning-time HTML reconstruction aligned with prepare/merge."
            ),
            "openrouter_model": args.openrouter_model,
            "openrouter_timeout_seconds": args.openrouter_timeout_seconds,
            "openrouter_max_attempts": args.openrouter_max_attempts,
            "openrouter_retry_backoff_seconds": args.openrouter_retry_backoff_seconds,
            "openrouter_max_tokens": args.openrouter_max_tokens,
            "openrouter_cache_path": str(openrouter_cache_path.expanduser().resolve()),
        },
        "selection": {
            "total_input_samples": len(source_records),
            "selected_benign": len(selected_records["benign"]),
            "selected_malicious": len(selected_records["malicious"]),
            "selected_total": len(selected_source_records),
            "per_attack_target": {
                attack_target: {
                    "num_records": len(samples_by_target[attack_target]),
                    "num_benign": sum(
                        1
                        for sample in samples_by_target[attack_target]
                        if normalize_label(sample.get("label")) == "benign"
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
    write_json(summary_output_path(output_root, dataset_stem), build_summary)

    print("Finished building Guard attack datasets with OpenRouter-generated reasoning.")
    print(f"- aggregate generated only: {aggregate_generated_path}")
    print(f"- aggregate with source: {aggregate_with_source_path}")
    for target_summary in target_summaries:
        print(f"- {target_summary['attack_target']}: {target_summary['dataset_json']}")


if __name__ == "__main__":
    main()
