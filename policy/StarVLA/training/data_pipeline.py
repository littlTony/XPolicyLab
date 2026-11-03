"""Native replacement sampling, with explicit seed and worker-visible epoch."""
import multiprocessing as mp


def install_dataset_pipeline(seed):
    from starVLA.dataloader import lerobot_datasets as factory
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
    if getattr(factory, "_ee58_seed", None) == int(seed):
        return
    original = getattr(factory, "_ee58_original_get_vla_dataset", factory.get_vla_dataset)

    class SharedEpochMixture(LeRobotMixtureDataset):
        def __init__(self, *args, **kwargs):
            self.shared_epoch = mp.Value("q", 0)
            super().__init__(*args, **kwargs)

        def set_epoch(self, epoch):
            self.shared_epoch.value = int(epoch)
            super().set_epoch(epoch)

        def sample_step(self, index):
            # Persistent DataLoader workers otherwise keep an old ordinary attr.
            self.epoch = int(self.shared_epoch.value)
            return super().sample_step(index)

    def get_vla_dataset(*args, **kwargs):
        kwargs["seed"] = int(seed)
        return original(*args, **kwargs)

    factory.LeRobotMixtureDataset = SharedEpochMixture
    factory._ee58_original_get_vla_dataset = original
    factory.get_vla_dataset = get_vla_dataset
    factory._ee58_seed = int(seed)
