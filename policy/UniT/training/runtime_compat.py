"""Parameter-free three-view compatibility; official UniT checkout stays clean."""
import torch
from itertools import islice


class FullBatchSampler(torch.utils.data.Sampler):
    """Trim a freshly shuffled epoch to full64 groups, preserving all-frame eligibility."""
    def __init__(self, sampler):
        self.sampler = sampler
        self.count = len(sampler) // 64 * 64

    def __iter__(self):
        return islice(iter(self.sampler), self.count)

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


def encode_views(vision_model, images, batch_size, hidden_size):
    if images.ndim != 5 or images.shape[:2] != (batch_size, 3):
        raise ValueError(f"Expected one RGB frame per each of three views [B,3,C,H,W], got {images.shape}")
    flat = images.reshape(batch_size * 3, *images.shape[2:])
    features = vision_model(flat)
    if features.ndim != 3 or features.shape[0] != batch_size * 3 or features.shape[-1] != hidden_size:
        raise ValueError(f"Unexpected DINO features {features.shape}")
    return features.reshape(batch_size, 3, features.shape[1], hidden_size).mean(dim=1)


def multiview_forward(self, obs_input, goal_input, batch_size=1):
    if goal_input is None or not torch.is_tensor(obs_input) or not torch.is_tensor(goal_input):
        raise ValueError("EE58 tokenizer requires tensor obs and goal images from all three RGB cameras")
    with torch.no_grad():
        obs_features = encode_views(self.vision_model, obs_input, batch_size, self.hidden_size)
        goal_features = encode_views(self.vision_model, goal_input, batch_size, self.hidden_size)
    outputs = self.m_former(cond_hidden_states=obs_features, target_hidden_states=goal_features)
    return outputs.last_hidden_state[:, :self.query_num], obs_features, goal_features


def install():
    from gr00t.model.tokenizer.vision_branch_encoder import VisionBranchEncoder
    VisionBranchEncoder.forward = multiview_forward
    # Accelerate1.3's DataLoaderShard reaches BatchSamplerShard.set_epoch but
    # that wrapper provides no setter/sampler attribute. Forward explicitly so
    # the random52-frame epoch tail changes along with the native sampler seed.
    from accelerate.data_loader import BatchSamplerShard
    if not hasattr(BatchSamplerShard, "set_epoch"):
        def shard_set_epoch(self, epoch):
            wrapped = self.batch_sampler
            setter = getattr(wrapped, "set_epoch", None)
            if setter is None:
                setter = getattr(getattr(wrapped, "sampler", None), "set_epoch", None)
            if setter is not None:
                setter(epoch)
        BatchSamplerShard.set_epoch = shard_set_epoch
    from gr00t.experiment.trainer import DualBrainTrainer, BaseSampler
    if getattr(DualBrainTrainer, "_ee58_installed", False):
        return
    original_create_optimizer = DualBrainTrainer.create_optimizer

    def create_optimizer(self):
        result = original_create_optimizer(self)
        # The native config has a zero-token future embedding. ZeRO cannot
        # partition empty placeholders; the successful Spark code excludes them.
        for group in self.optimizer.param_groups:
            group["params"] = [parameter for parameter in group["params"] if parameter.numel() > 0]
        return result

    def get_train_sampler(self, *args, **kwargs):
        return FullBatchSampler(BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed))

    DualBrainTrainer.create_optimizer = create_optimizer
    DualBrainTrainer._get_train_sampler = get_train_sampler
    DualBrainTrainer._ee58_installed = True


def strict_checkpoint_load(cls, path, **kwargs):
    """Use the HF loader directly, bypassing UniT's catch-and-random-init fallback."""
    from transformers import PreTrainedModel
    model, report = PreTrainedModel.from_pretrained.__func__(cls, str(path), output_loading_info=True, **kwargs)
    errors = {key: value for key, value in report.items() if value}
    if errors:
        raise RuntimeError(f"Checkpoint is not fully compatible; refusing random fallback: {errors}")
    return model
