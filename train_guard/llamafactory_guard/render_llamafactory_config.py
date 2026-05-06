#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "llamafactory_guard" / "data"
DEFAULT_CONFIG_DIR = ROOT / "llamafactory_guard" / "generated_configs"
DEFAULT_OUTPUT_ROOT = ROOT / "llamafactory_guard" / "outputs"
DEFAULT_DEEPSPEED_CONFIG = ROOT / "llamafactory_guard" / "deepspeed" / "ds_z3_config.json"
DEFAULT_TRAIN_DATASET_NAME = "guard_train_sft"
DEFAULT_EVAL_DATASET_NAME = "guard_val_sft"
DATASET_INFO_FILE_NAME = "dataset_info.json"

MODEL_IDS = {
    "thinking": {
        "0.8b": "Qwen/Qwen3.5-0.8B",
        "2b": "Qwen/Qwen3.5-2B",
        "4b": "Qwen/Qwen3.5-4B",
        "9b": "Qwen/Qwen3.5-9B",
    },
    "base": {
        "0.8b": "Qwen/Qwen3.5-0.8B-Base",
        "2b": "Qwen/Qwen3.5-2B-Base",
        "4b": "Qwen/Qwen3.5-4B-Base",
        "9b": "Qwen/Qwen3.5-9B-Base",
    },
}

TRAIN_DEFAULTS = {
    "0.8b": {
        "lora": {"batch_size": 4, "grad_accum": 4},
        "full": {"batch_size": 2, "grad_accum": 8},
    },
    "2b": {
        "lora": {"batch_size": 2, "grad_accum": 8},
        "full": {"batch_size": 1, "grad_accum": 16},
    },
    "4b": {
        "lora": {"batch_size": 1, "grad_accum": 16},
        "full": {"batch_size": 1, "grad_accum": 32},
    },
    "9b": {
        "lora": {"batch_size": 1, "grad_accum": 32},
        "full": {"batch_size": 1, "grad_accum": 64},
    },
}


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def load_dataset_info(dataset_dir: Path) -> dict[str, object]:
    dataset_info_path = dataset_dir / DATASET_INFO_FILE_NAME
    if not dataset_info_path.is_file():
        return {}

    try:
        payload = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {dataset_info_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit(f"{dataset_info_path} must contain a JSON object.")

    return payload


def build_train_yaml(
    *,
    model_name_or_path: str,
    finetuning_type: str,
    dataset_dir: Path,
    train_dataset_name: str,
    eval_dataset_name: str | None,
    output_dir: Path,
    deepspeed_config: Path,
    batch_size: int,
    grad_accum: int,
    cutoff_len: int,
    image_max_pixels: int,
    image_min_pixels: int,
    learning_rate: float,
    num_train_epochs: float,
    precision: str,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    save_total_limit: int,
    preprocessing_num_workers: int,
    dataloader_num_workers: int,
    lora_rank: int,
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "### model",
            f"model_name_or_path: {yaml_quote(model_name_or_path)}",
            f"image_max_pixels: {image_max_pixels}",
            f"image_min_pixels: {image_min_pixels}",
            "video_max_pixels: 16384",
            "trust_remote_code: true",
            "",
            "### method",
            "stage: sft",
            "do_train: true",
            f"do_eval: {'true' if eval_dataset_name is not None else 'false'}",
            f"finetuning_type: {finetuning_type}",
        ]
    )

    if finetuning_type == "lora":
        lines.extend(
            [
                f"lora_rank: {lora_rank}",
                "lora_target: all",
            ]
        )
    else:
        lines.append(f"deepspeed: {yaml_quote(str(deepspeed_config))}")

    lines.extend(
        [
            "",
            "### dataset",
            f"dataset_dir: {yaml_quote(str(dataset_dir))}",
            f"media_dir: {yaml_quote(str(dataset_dir))}",
            f"dataset: {yaml_quote(train_dataset_name)}",
            "template: qwen3_5_nothink",
            "enable_thinking: false",
            f"cutoff_len: {cutoff_len}",
            f"preprocessing_num_workers: {preprocessing_num_workers}",
            f"dataloader_num_workers: {dataloader_num_workers}",
            "",
            "### output",
            f"output_dir: {yaml_quote(str(output_dir))}",
            f"logging_steps: {logging_steps}",
            f"save_steps: {save_steps}",
            "plot_loss: true",
            "overwrite_output_dir: true",
            f"save_total_limit: {save_total_limit}",
            "save_only_model: false",
            "report_to: none",
            "",
            "### train",
            f"per_device_train_batch_size: {batch_size}",
            f"gradient_accumulation_steps: {grad_accum}",
            f"learning_rate: {learning_rate:.1e}",
            f"num_train_epochs: {num_train_epochs}",
            "lr_scheduler_type: cosine",
            "warmup_ratio: 0.05",
            "gradient_checkpointing: true",
            "ddp_timeout: 180000000",
            "resume_from_checkpoint: null",
        ]
    )

    if eval_dataset_name is not None:
        lines[lines.index(f"dataset: {yaml_quote(train_dataset_name)}") + 1 : lines.index("template: qwen3_5_nothink")] = [
            f"eval_dataset: {yaml_quote(eval_dataset_name)}"
        ]
        lines[lines.index(f"save_steps: {save_steps}") + 1 : lines.index("plot_loss: true")] = [
            f"eval_steps: {eval_steps}",
            "eval_strategy: steps",
            "per_device_eval_batch_size: 1",
        ]
    else:
        lines[lines.index(f"save_steps: {save_steps}") + 1 : lines.index("plot_loss: true")] = ['eval_strategy: "no"']

    if precision == "bf16":
        lines.append("bf16: true")
    else:
        lines.append("fp16: true")

    return "\n".join(lines) + "\n"


def build_merge_yaml(
    *,
    model_name_or_path: str,
    adapter_name_or_path: Path,
    export_dir: Path,
) -> str:
    lines = [
        "### Note: DO NOT use quantized model or quantization_bit when merging lora adapters",
        "",
        "### model",
        f"model_name_or_path: {yaml_quote(model_name_or_path)}",
        f"adapter_name_or_path: {yaml_quote(str(adapter_name_or_path))}",
        "template: qwen3_5_nothink",
        "trust_remote_code: true",
        "",
        "### export",
        f"export_dir: {yaml_quote(str(export_dir))}",
        "export_size: 5",
        "export_device: cpu",
        "export_legacy_format: false",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLaMA-Factory YAML configs for Qwen 3.5 guard SFT with LoRA or full fine-tuning."
    )
    parser.add_argument("--model-size", choices=["0.8b", "2b", "4b", "9b"], default="4b")
    parser.add_argument("--model-variant", choices=["thinking", "base"], default="thinking")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional local model path. Overrides model-size/model-variant repo id.")
    parser.add_argument("--finetuning-type", choices=["lora", "full"], default="lora")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--train-dataset-name", type=str, default=DEFAULT_TRAIN_DATASET_NAME)
    parser.add_argument("--eval-dataset-name", type=str, default=DEFAULT_EVAL_DATASET_NAME)
    parser.add_argument("--disable-eval", action="store_true", help="Force train-only config generation without eval_dataset.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--deepspeed-config", type=Path, default=DEFAULT_DEEPSPEED_CONFIG)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--image-max-pixels", type=int, default=2_250_000)
    parser.add_argument("--image-min-pixels", type=int, default=512 * 512)
    parser.add_argument("--learning-rate", type=float, default=0.0, help="Use 0 to keep the method default.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=0, help="Use 0 to keep the size default.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0, help="Use 0 to keep the size default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    deepspeed_config = args.deepspeed_config.expanduser().resolve()
    dataset_info = load_dataset_info(dataset_dir)

    if args.model_path is not None:
        model_name_or_path = str(args.model_path.expanduser().resolve())
    else:
        model_name_or_path = MODEL_IDS[args.model_variant][args.model_size]

    if args.train_dataset_name not in dataset_info:
        raise SystemExit(
            f"Dataset {args.train_dataset_name!r} was not found in {dataset_dir / DATASET_INFO_FILE_NAME}."
        )

    eval_dataset_name: str | None = None
    if args.disable_eval:
        print("Evaluation disabled by --disable-eval; rendering train-only config.")
    elif args.eval_dataset_name in dataset_info:
        eval_dataset_name = args.eval_dataset_name
    else:
        print(
            f"No {args.eval_dataset_name!r} entry found in {dataset_dir / DATASET_INFO_FILE_NAME}; "
            "rendering train-only config."
        )

    size_defaults = TRAIN_DEFAULTS[args.model_size][args.finetuning_type]
    batch_size = args.per_device_train_batch_size or size_defaults["batch_size"]
    grad_accum = args.gradient_accumulation_steps or size_defaults["grad_accum"]

    if args.learning_rate > 0:
        learning_rate = args.learning_rate
    elif args.finetuning_type == "lora":
        learning_rate = 5.0e-5
    else:
        learning_rate = 1.0e-5

    config_stem = f"qwen35_{args.model_variant}_{args.model_size}_{args.finetuning_type}"
    output_dir = output_root / config_stem
    train_config_path = config_dir / f"{config_stem}.yaml"

    train_yaml = build_train_yaml(
        model_name_or_path=model_name_or_path,
        finetuning_type=args.finetuning_type,
        dataset_dir=dataset_dir,
        train_dataset_name=args.train_dataset_name,
        eval_dataset_name=eval_dataset_name,
        output_dir=output_dir,
        deepspeed_config=deepspeed_config,
        batch_size=batch_size,
        grad_accum=grad_accum,
        cutoff_len=args.cutoff_len,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        learning_rate=learning_rate,
        num_train_epochs=args.num_train_epochs,
        precision=args.precision,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        preprocessing_num_workers=args.preprocessing_num_workers,
        dataloader_num_workers=args.dataloader_num_workers,
        lora_rank=args.lora_rank,
    )

    train_config_path.parent.mkdir(parents=True, exist_ok=True)
    train_config_path.write_text(train_yaml, encoding="utf-8")
    print(f"Saved train config to: {train_config_path}")

    if args.finetuning_type == "lora":
        merge_config_path = config_dir / f"{config_stem}_merge.yaml"
        merged_export_dir = output_root / f"{config_stem}_merged"
        merge_yaml = build_merge_yaml(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=output_dir,
            export_dir=merged_export_dir,
        )
        merge_config_path.write_text(merge_yaml, encoding="utf-8")
        print(f"Saved merge config to: {merge_config_path}")


if __name__ == "__main__":
    main()
