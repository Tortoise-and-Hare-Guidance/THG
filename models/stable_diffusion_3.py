from diffusers import StableDiffusion3Pipeline

import torch

from models.base import DiffusionModel


class StableDiffusion3Model(DiffusionModel):
    def __init__(self, device, model_key):
        super().__init__(device)

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_key, torch_dtype=torch.bfloat16, variant="fp16"
        ).to(device)

        self.pipe.set_progress_bar_config(disable=True)
