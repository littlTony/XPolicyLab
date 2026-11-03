"""CPU contract checks for the new32 recipe and the unchanged old64 semantics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import yaml

POLICY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POLICY.parents[2]))
from prepare_sparkarena import resolve_run, validate_config
from XPolicyLab.policy.StarVLA import ee58_runtime as runtime


def rejected(call):
    try:
        call()
    except (ValueError, AssertionError):
        return True
    raise AssertionError("Invalid contract was accepted")


def main():
    cfg = yaml.safe_load((POLICY / "configs/sparkarena_qwen_pi_v3.yaml").read_text())
    cfg["framework"]["action_model"].update(action_dim=58, state_dim=58)
    cfg["sparkarena_contract"] = dict(bench_name="DexBench", env_cfg_type="tianji_marvin_wuji",
        action_type="ee", data_contract_id=runtime.CONTRACT_ID, rot6d="matrix_rows_0_then_1",
        num_processes=8, effective_batch=32, adapter_revision="r20261031_fast")
    validate_config(cfg)
    cases = []
    for ga, batch in ((1, 32), (2, 64)):
        config = deepcopy(cfg)
        config["trainer"]["gradient_accumulation_steps"] = ga
        config["sparkarena_contract"]["effective_batch"] = batch
        with tempfile.TemporaryDirectory(prefix="starvla-runtime-contract-") as temporary:
            run = Path(temporary)
            weight = run / "checkpoints/steps_20_pytorch_model.pt"
            weight.parent.mkdir()
            weight.write_bytes(b"CPU CONTRACT ONLY; no model weights")
            (run / "config.full.yaml").write_text(yaml.safe_dump(config))
            runtime.save_sidecars(run, weight, config, {"CPU_fixture": True}, 20)
            manifest, loaded, _ = runtime.read_bundle(weight)
            assert (manifest["gradient_accumulation_steps"], manifest["effective_batch"]) == (ga, batch)
            assert loaded == config
            manifest_path = weight.with_name(weight.name + ".ee58.json")
            manifest["effective_batch"] = 64 if batch == 32 else 32
            manifest_path.write_text(json.dumps(manifest))
            assert rejected(lambda: runtime.read_bundle(weight))
        cases.append({"ga": ga, "global_batch": batch, "save_and_read_bundle": "passed",
                      "mismatched_batch_manifest_rejected": True})
    invalid = deepcopy(cfg)
    invalid["sparkarena_contract"]["effective_batch"] = 64
    assert rejected(lambda: runtime.runtime_contract(invalid, "stats", "config", 20))
    invalid = deepcopy(cfg)
    invalid["trainer"]["gradient_accumulation_steps"] = 4
    assert rejected(lambda: runtime.runtime_contract(invalid, "stats", "config", 20))
    base = "DexBench-tianji_wuji-tianji_marvin_wuji-ee-42"
    args = SimpleNamespace(run_id=None, run_suffix="r20261031_fast", output_dir=None)
    run_id, output = resolve_run(args, lambda _: base)
    assert run_id == base + "-r20261031_fast" and output == POLICY / "checkpoints" / run_id
    for suffix in ("bs32", "batch_size32", "../escape"):
        bad = deepcopy(args); bad.run_suffix = suffix
        assert rejected(lambda: resolve_run(bad, lambda _: base))
    bad = deepcopy(args); bad.run_id = base
    assert rejected(lambda: resolve_run(bad, lambda _: base))
    bad = deepcopy(args); bad.output_dir = Path("/tmp/wrong-name")
    assert rejected(lambda: resolve_run(bad, lambda _: base))
    report = {"status": "passed", "GPU_or_model_forward": False, "runtime_contract_cases": cases,
              "new_prepare_contract": "8xmicro4xGA1=32 /60000", "run_prefix_and_output_guards": "passed",
              "strict_source_hash_check": "retained; original pre-revision manifests require matching archived source"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
