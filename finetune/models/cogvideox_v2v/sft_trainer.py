from ..cogvideox_v2v.lora_trainer import CogVideoXV2VLoraTrainer
from ..utils import register


class CogVideoXV2VSftTrainer(CogVideoXV2VLoraTrainer):
    pass


register("cogvideox-v2v", "sft", CogVideoXV2VSftTrainer)
