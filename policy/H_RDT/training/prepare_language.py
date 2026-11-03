"""Explicit offline T5-XXL preparation for all 60 exact instruction strings.

This script is intentionally separate: T5-XXL is large and is never loaded by
the CPU data preflight. The user selects the device when invoking this script.
"""
import argparse
import json
from pathlib import Path

from ee58_data import DEFAULT_DATASET, sha256, text_hash, validate_dataset, validate_language


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True, help="Explicitly choose cpu or an allocated cuda:N device")
    args = parser.parse_args()
    spec = validate_dataset(args.dataset)
    if args.output.exists():
        raise FileExistsError("Preserve the existing cache; choose a new output directory")
    if not (args.encoder / "config.json").is_file():
        raise FileNotFoundError("Local T5 config/weights required; downloads are disabled")
    encoder_config = json.loads((args.encoder / "config.json").read_text())
    if encoder_config.get("d_model") != 4096:
        raise ValueError("H-RDT uses T5-v1_1-XXL with 4096-dimensional features")
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True, model_max_length=1024)
    encoder = T5EncoderModel.from_pretrained(args.encoder, local_files_only=True,
                                            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device).eval()
    args.output.mkdir(parents=True)
    records = {}
    for task in spec["tasks"]:
        tokens = tokenizer(task["task"], return_tensors="pt", truncation=True, max_length=1024,
                           padding=False, return_attention_mask=True).to(args.device)
        with torch.inference_mode():
            embeddings = encoder(**tokens).last_hidden_state[0].detach().cpu().contiguous()
        file = args.output / f"task_{task['task_index']:03d}.pt"
        torch.save({"instruction": task["task"], "embeddings": embeddings}, file)
        records[str(task["task_index"])] = {"text_sha256": text_hash(task["task"]),
                                           "file": file.name, "sha256": sha256(file)}
    weight_files = sorted(args.encoder.glob("*.bin")) + sorted(args.encoder.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError("No local T5 weight files were found")
    weight_records = {file.name: {"sha256": sha256(file), "bytes": file.stat().st_size,
                                 "mtime_ns": file.stat().st_mtime_ns} for file in weight_files}
    manifest = {"dataset_tasks_sha256": sha256(spec["root"] / "meta/tasks.jsonl"),
                "encoder_path": str(args.encoder.resolve()), "encoder_config_sha256": sha256(args.encoder / "config.json"),
                "encoder_weights": weight_records, "feature_dim": 4096, "max_length": 1024,
                "texts": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    validate_language(args.output, spec)
    print(f"Verified all {len(records)} T5 instruction embeddings: {args.output}")


if __name__ == "__main__":
    main()
