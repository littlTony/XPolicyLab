import copy
from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import ActionConfig,ActionFormat,ActionRepresentation,ActionType,ModalityConfig
MODALITIES={
 "video":ModalityConfig(delta_indices=[0],modality_keys=["front","left_wrist","right_wrist"]),
 "state":ModalityConfig(delta_indices=[0],modality_keys=["left_eef","left_hand","right_eef","right_hand"]),
 "action":ModalityConfig(delta_indices=list(range(16)),modality_keys=["left_eef","left_hand","right_eef","right_hand"],action_configs=[
   ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.EEF,format=ActionFormat.XYZ_ROT6D),
   ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.NON_EEF,format=ActionFormat.DEFAULT),
   ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.EEF,format=ActionFormat.XYZ_ROT6D),
   ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.NON_EEF,format=ActionFormat.DEFAULT)]),
 "language":ModalityConfig(delta_indices=[0],modality_keys=["annotation.human.task_description"])
}
register_modality_config(MODALITIES,EmbodimentTag.NEW_EMBODIMENT)
register_modality_config(copy.deepcopy(MODALITIES),EmbodimentTag.GENERAL_EMBODIMENT)
