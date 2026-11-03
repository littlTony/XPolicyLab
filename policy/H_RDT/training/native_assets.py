"""One native implementation and one vision preprocessing path for train/serve."""
from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types

POLICY = Path(__file__).resolve().parents[1]


def native_paths():
    upstream = POLICY / "upstream"
    sys.path.insert(0, str(upstream))
    for name in ("models", "datasets", "utils"):
        path = str(upstream / name)
        existing = sys.modules.get(name)
        if existing is not None and path in list(getattr(existing, "__path__", [])):
            continue
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        package = types.ModuleType(name)
        package.__path__ = [path]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = [path]
        sys.modules[name] = package
    if importlib.util.find_spec("imgaug") is None and "utils.image_corrupt" not in sys.modules:
        # The native dataset imports optional corruption augmentation even when
        # disabled. Our reviewed recipe rejects image_aug=True; fail explicitly
        # if anyone attempts that unsupported path instead of fabricating pixels.
        module = types.ModuleType("utils.image_corrupt")
        module.__spec__ = importlib.machinery.ModuleSpec("utils.image_corrupt", loader=None)

        def disabled_corruption(*args, **kwargs):
            raise RuntimeError("imgaug is absent and corruption augmentation is disabled in the EE58 recipe")

        module.image_corrupt = disabled_corruption
        sys.modules["utils.image_corrupt"] = module
        # The native Consumer imports both old backends eagerly. EE58Consumer
        # replaces its constructor, so neither backend can ever be selected.
        # Guard those imports too; an accidental legacy selection fails clearly.
        for module_name, class_name in (("datasets.robotwin2.robotwin_agilex_dataset", "RobotwinAgilexDataset"),
                                        ("datasets.pretrain.egodex_dataset", "EgoDexDataset")):
            if module_name in sys.modules:
                continue
            guarded = types.ModuleType(module_name)
            guarded.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)

            class DisabledLegacyDataset:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Legacy Robotwin/EgoDex backend is disabled in the independent EE58 recipe")

            setattr(guarded, class_name, DisabledLegacyDataset)
            sys.modules[module_name] = guarded
    return upstream


def build_vision(vision_root, *, transform_only=False, image_size=384):
    """Use native letterbox/transforms; transform-only allocates models on meta."""
    native_paths()
    import timm
    import torch
    from unittest.mock import patch
    from models.encoder.dinosiglip_vit import DinoSigLIPViTBackbone
    vision_root = Path(vision_root)
    paths = {
        "vit_large_patch14_reg4_dinov2.lvd142m": vision_root / "vit_large_patch14_reg4_dinov2.lvd142m/pytorch_model.bin",
        "vit_so400m_patch14_siglip_384": vision_root / "vit_so400m_patch14_siglip_384/open_clip_pytorch_model.bin",
    }
    original = timm.create_model

    def create_local(model_name, *args, **kwargs):
        if model_name not in paths or not paths[model_name].is_file():
            raise FileNotFoundError(f"Explicit local DINO/SigLIP weights missing: {model_name}")
        kwargs["pretrained_cfg_overlay"] = dict(file=str(paths[model_name]))
        if transform_only:
            kwargs["pretrained"] = False
            # Limit meta to model parameters. Native torchvision normalization
            # constants must be constructed later on CPU, outside this context.
            with torch.device("meta"):
                return original(model_name, *args, **kwargs)
        return original(model_name, *args, **kwargs)

    with patch.object(timm, "create_model", create_local):
        vision = DinoSigLIPViTBackbone("dino-siglip", "letterbox", default_image_size=image_size)
    return vision.get_image_transform() if transform_only else vision
