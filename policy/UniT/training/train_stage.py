"""Explicit native UniT stages; config-only init never masquerades as full pretrain."""
import argparse
import json
import os
from pathlib import Path
import sys

from prepare_ee58 import POLICY, UPSTREAM, CONTRACT_ID, sha, require


def tokenizer_gate(path, stats_sha, require_finished=True):
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    contract = cfg.get("ee58_contract", {})
    require(contract.get("id") == CONTRACT_ID and contract.get("real_dim") == 58
            and contract.get("source_stats_sha256") == stats_sha, "Tokenizer is not trained for this EE58/statistics contract")
    require(cfg["action_dim"] == cfg["state_dim"] == 128 and cfg["action_horizon"] == 16,
            "Tokenizer must remain padded128/H16")
    if require_finished:
        state = json.loads((path / "trainer_state.json").read_text())
        require(state["global_step"] == 40000, "Stage2 requires the completed 40000-update EE58 tokenizer")
    from prepare_ee58 import checkpoint_headers
    checkpoint_headers(path)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--stage", choices=("tokenizer", "policy"), required=True)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text())
    require(int(os.environ.get("WORLD_SIZE", "1")) == 8, "Use train_ee58.sh: the fixed recipe requires 8 ranks")
    for rel, digest in prepared["adapter_sha256"].items():
        require(sha(POLICY / rel) == digest, f"Adapter/config changed after preflight: {rel}")
    for name, digest in prepared["source_metadata_sha256"].items():
        require(sha(Path(prepared["source_dataset"]) / "meta" / name) == digest,
                f"Dataset metadata changed after preflight: {name}")
    import torch
    import torch.distributed as dist
    from transformers import TrainingArguments, set_seed
    from ee58_data import EE58DataConfig, EE58Dataset
    from runtime_compat import install, strict_checkpoint_load
    from gr00t.experiment.runner import TrainRunner
    from gr00t.model.gr00t_n1_tokenizer_unit import GR00T_Tokenizer, GR00T_Tokenizer_Config
    from gr00t.model.gr00t_n1_tokenizer_unit_inference import GR00T_Tokenizer as InferenceTokenizer
    from gr00t.model.gr00t_n1_tokenizer_unit_inference import GR00T_Tokenizer_Config as InferenceTokenizerConfig
    from gr00t.model.gr00t_n1_unit import GR00T_N1_5_UniT, GR00T_N1_5_UniT_Config

    install()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    stage, recipe = args.stage, prepared["recipe"]
    hyper = recipe[stage]
    set_seed(recipe["seed"])
    output = Path(prepared["output_dir"]) / stage
    model_dict = json.loads((Path(prepared["workspace"]) / f"{stage}_config.json").read_text())
    tokenizer_path = Path(prepared["output_dir"]) / "tokenizer/checkpoint-40000"
    stats_sha = prepared["source_metadata_sha256"]["stats.json"]
    if stage == "policy":
        tokenizer_config = tokenizer_gate(tokenizer_path, stats_sha)
    data_config = EE58DataConfig(eagle_path=prepared["vlm_path"], use_bridge=stage == "policy",
                                 ignore_lang_prefix=True, num_bridge_tokens=8 if stage == "policy" else None,
                                 tokenizer_only=stage == "tokenizer")
    transforms = data_config.transform()
    dataset = EE58Dataset(dataset_path=prepared["dataset_path"], modality_configs=data_config.modality_config(),
                          transforms=transforms, embodiment_tag="new_embodiment", split="[:]",
                          video_backend="torchvision_av")
    if stage == "tokenizer":
        # Deliberate new tokenizer, with the public pretrained DINO backbone.
        config = GR00T_Tokenizer_Config(**model_dict)
        if prepared.get("tokenizer_warmstart"):
            model = strict_checkpoint_load(GR00T_Tokenizer, prepared["tokenizer_warmstart"], config=config,
                                            local_model_path=prepared["tokenizer_warmstart"])
        else:
            model = GR00T_Tokenizer(config, local_model_path=None)
        model.set_trainable_parameters(tune_vision_model=False, tune_vision_m_former=True,
                                      tune_bridge_projector=True, tune_action_encoder=True,
                                      tune_fusion=True, tune_vq=True, tune_vision_decoder=True,
                                      tune_action_decoder_projector=True, tune_action_decoder_diffusion=True)
    else:
        tokenizer_len = len(transforms.transforms[-1].eagle_processor.tokenizer)
        model_dict["backbone_cfg"].update(tune_llm=True, tune_visual=False, select_layer=36,
                                          tune_bridge_embedding=True, load_bf16=False)
        model_dict["unit_cfg"].update(groot_tokenizer_path=str(tokenizer_path), bridge_type="vision_lang_obs",
                                      compute_bridge_loss=True, bridge_loss_type="ce", use_image_type_embedding=True)
        # Explicit fresh policy projections + pretrained Qwen, matching the Spark recipe.
        model = GR00T_N1_5_UniT(GR00T_N1_5_UniT_Config(**model_dict), local_model_path=None,
                                tokenizer_len=tokenizer_len, bridge_type="vision_lang_obs", select_layer=36,
                                compute_bridge_loss=True, bridge_loss_type="ce", use_image_type_embedding=True,
                                use_vl_mask=True, use_correct_attn_mask=True,
                                groot_tokenizer_path=str(tokenizer_path), detach_vl_for_action=True)
        tokenizer_config["backbone_cfg"]["dinov2_path"] = prepared["dino_path"]
        frozen_tokenizer = strict_checkpoint_load(InferenceTokenizer, tokenizer_path,
                                                  config=InferenceTokenizerConfig(**tokenizer_config),
                                                  local_model_path=str(tokenizer_path))
        frozen_tokenizer.set_trainable_parameters(tune_vision_model=False, tune_vision_m_former=False,
                                                 tune_bridge_projector=False, tune_action_encoder=False,
                                                 tune_fusion=False, tune_vq=False, tune_vision_decoder=False,
                                                 tune_action_decoder_projector=False, tune_action_decoder_diffusion=False)
        frozen_tokenizer.requires_grad_(False).eval()
        model.groot_tokenizer = frozen_tokenizer
        model.set_trainable_parameters(tune_bridge_visual=False, tune_image_type_embedding=True)
        model.action_head.set_trainable_parameters(tune_projector=True, tune_diffusion_model=True)
        model.config.unit_cfg.update(groot_tokenizer_path=str(tokenizer_path), bridge_loss_weight=0.1,
                                      detach_vl_for_action=True, unified_embodiment_id=None)
        # Qwen caches are incompatible with activation checkpointing during training.
        model.backbone.eagle_model.config.use_cache = False
    # Native loss/update implementation, with explicit settings that native CLI hard-codes.
    training_args = TrainingArguments(
        output_dir=str(output), run_name=output.parent.name + "_" + stage, remove_unused_columns=False,
        deepspeed=str(POLICY / "configs/deepspeed_zero2.json") if stage == "policy" else None,
        gradient_checkpointing=stage == "policy", gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=hyper["bf16"], fp16=False, tf32=True,
        per_device_train_batch_size=hyper["micro_batch"], gradient_accumulation_steps=hyper["ga"],
        dataloader_num_workers=4, dataloader_pin_memory=False, dataloader_prefetch_factor=2,
        dataloader_persistent_workers=True, optim="adamw_torch", adam_beta1=0.95, adam_beta2=0.999,
        adam_epsilon=1e-8, learning_rate=hyper["learning_rate"], weight_decay=1e-5,
        warmup_ratio=0.05, lr_scheduler_type="cosine", logging_steps=100 if stage == "tokenizer" else 10,
        max_steps=hyper["max_steps"], save_strategy="steps", save_steps=10000,
        save_total_limit=4 if stage == "tokenizer" else 8, report_to=["tensorboard"],
        seed=recipe["seed"], do_eval=False, ddp_find_unused_parameters=False, ddp_bucket_cap_mb=100,
        max_grad_norm=1.0, save_safetensors=True)
    runner = TrainRunner(model=model, training_args=training_args, train_dataset=dataset,
                         resume_from_checkpoint=False, eagle_path=prepared["vlm_path"],
                         num_bridge_tokens=8 if stage == "policy" else None)
    if stage == "policy":
        from inference_snapshot import InferenceSnapshot
        runner.trainer.add_callback(InferenceSnapshot(prepared))
    require(training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            * training_args.world_size == 64, "Effective batch must be64")
    if training_args.process_index == 0:
        (output / "launch_contract.json").write_text(json.dumps(prepared, indent=2))
    runner.train()
    require(runner.trainer.state.global_step == hyper["max_steps"], "Stage did not finish all optimizer updates")
    if dist.is_initialized():
        dist.barrier()
    if stage == "tokenizer" and training_args.process_index == 0:
        tokenizer_gate(tokenizer_path, stats_sha)
        (output / "tokenizer_ready_ee58.json").write_text(json.dumps({"contract_id": CONTRACT_ID,
             "source_stats_sha256": stats_sha, "global_step": 40000, "checkpoint": str(tokenizer_path)}, indent=2))
    if stage == "policy" and training_args.process_index == 0:
        from inference_snapshot import save_final
        save_final(output, prepared, runner.trainer.state.global_step)


if __name__ == "__main__":
    main()
