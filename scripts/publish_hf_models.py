#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = [
    {
        "name": "WARD-0.8b",
        "source": Path("/home/tri/Guard_new/aaa_rl/exports/split3_grpo__qwen3_5_0_8b_perf__global_step_19__hf"),
        "card": ROOT / "hf_model_cards" / "README.WARD-0.8b.md",
    },
    {
        "name": "WARD-2b",
        "source": Path("/home/tri/Guard_new/aaa_rl/exports/split3_grpo__qwen3_5_2b_perf__global_step_19__hf"),
        "card": ROOT / "hf_model_cards" / "README.WARD-2b.md",
    },
]


def render_card(template_path: Path, namespace: str) -> str:
    return template_path.read_text(encoding="utf-8").replace("{{HF_NAMESPACE}}", namespace)


def detect_namespace(api: HfApi) -> str:
    info = api.whoami()
    if "name" not in info:
        raise RuntimeError("Could not infer Hugging Face namespace from current login.")
    return str(info["name"])


def stage_model(source_dir: Path, readme_text: str, destination_dir: Path) -> None:
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    (destination_dir / "README.md").write_text(readme_text, encoding="utf-8")


def publish_one(api: HfApi, namespace: str, model_name: str, source_dir: Path, card_template: Path, private: bool) -> str:
    repo_id = f"{namespace}/{model_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{model_name}-") as tmp:
        stage_dir = Path(tmp) / model_name
        readme_text = render_card(card_template, namespace)
        stage_model(source_dir=source_dir, readme_text=readme_text, destination_dir=stage_dir)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(stage_dir),
            commit_message=f"Upload {model_name}",
        )
    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish WARD checkpoints to Hugging Face.")
    parser.add_argument("--namespace", help="Target Hugging Face user or org. Defaults to the current login.")
    parser.add_argument("--private", action="store_true", help="Create private model repos.")
    args = parser.parse_args()

    api = HfApi()
    namespace = args.namespace or detect_namespace(api)

    for spec in MODEL_SPECS:
        repo_id = publish_one(
            api=api,
            namespace=namespace,
            model_name=spec["name"],
            source_dir=spec["source"],
            card_template=spec["card"],
            private=args.private,
        )
        print(f"Published {repo_id}")


if __name__ == "__main__":
    main()
