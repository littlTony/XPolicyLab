"""Sequential finite training job: tokenizer40000 -> policy60000, no daemon."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_ee58 import POLICY, require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text())
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == prepared["gpu_ids"], "GPU IDs differ from preflight")
    # Do not share GPUs with existing training jobs.
    selected = set(prepared["gpu_ids"].split(","))
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    gpu_map = {row.split(",")[0].strip(): row.split(",")[1].strip() for row in rows.splitlines() if row.strip()}
    require(selected <= gpu_map.keys(), "Some selected GPUs are not visible")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    busy = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
    require(not busy.intersection(gpu_map[index] for index in selected), "Selected GPU has existing compute tasks; preserve them")
    output = Path(prepared["output_dir"])
    output.parent.mkdir(exist_ok=True)
    output.mkdir()  # Atomic new-run reservation; never overwrite an old run.
    shutil.copy2(args.prepared, output / "prepared.json")
    shutil.copy2(Path(prepared["workspace"]) / "preflight.json", output / "preflight.json")
    for stage in ("tokenizer", "policy"):
        if stage == "policy":
            gate = json.loads((output / "tokenizer/tokenizer_ready_ee58.json").read_text())
            require(gate["global_step"] == 40000 and gate["contract_id"] == prepared["contract_id"], "Tokenizer gate failed")
        command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=8",
                   "--master_port=" + os.environ.get("UNIT_MASTER_PORT", "29625"),
                   str(POLICY / "training/train_stage.py"), "--prepared", str(args.prepared), "--stage", stage]
        subprocess.run(command, check=True, cwd=POLICY / "UniT", env=dict(os.environ, IS_TORCHRUN="1"))


if __name__ == "__main__":
    main()
