from dataclasses import dataclass
from typing import Optional

import torch
from PIL.Image import Image
from torchvision.transforms import GaussianBlur

from models.base import DiffusionModel


@dataclass
class SolverOutput:
    images: list[Image]
    denoising_output: Optional[torch.Tensor] = None


class Solver:
    """
    Base class for all solvers. This class is not meant to be used directly.

    Derived classes should implement the step method.
    self.prompt_embeds and self.negative_prompt_embeds are set in the generate method.
    """

    def __init__(self, net: DiffusionModel):
        self.net: DiffusionModel = net

    def generate(
        self,
        prompts: list[str],
        seed: int,
        num_inference_steps: int,
    ) -> SolverOutput:
        """
        Generate images from the given prompts and latents.
        This method should be overridden by derived classes.
        """
        raise NotImplementedError


class FFTMixin:
    def __init__(self, **kwargs):
        self.radius: float = kwargs.pop("radius")
        self.boost_factor: float = kwargs.pop("boost_factor")
        self.i_0: int = kwargs.pop("i_0")
        super().__init__(**kwargs)

    def enhance_guidance(self, guidance, i):
        dtype = guidance.dtype
        guidance = guidance.to(torch.float32)
        tensor_fft = torch.fft.fft2(guidance)
        tensor_fft = torch.fft.fftshift(tensor_fft)
        B, C, H, W = guidance.shape
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
        center_x, center_y = W // 2, H // 2
        mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= self.radius**2
        low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(guidance.device)
        high_freq_mask = ~low_freq_mask

        low_freq_fft = (
            tensor_fft * low_freq_mask * (self.boost_factor if i < self.i_0 else 1)
        )
        high_freq_fft = (
            tensor_fft * high_freq_mask * (self.boost_factor if i >= self.i_0 else 1)
        )

        combined_fft = torch.fft.ifftshift(low_freq_fft + high_freq_fft)
        result = torch.fft.ifft2(combined_fft).real
        return result.to(dtype)


class GaussianMixin:
    def __init__(self, **kwargs):
        self.radius: float = kwargs.pop("radius")
        self.boost_factor: float = kwargs.pop("boost_factor")
        self.i_0: int = kwargs.pop("i_0")
        super().__init__(**kwargs)

    def enhance_guidance(self, guidance, i):
        low_pass_filter = GaussianBlur(9, self.radius)
        high_pass_filter = lambda x: x - low_pass_filter(x)

        lf = self.boost_factor if i < self.i_0 else 1
        hf = self.boost_factor if i >= self.i_0 else 1
        return lf * low_pass_filter(guidance) + hf * high_pass_filter(guidance)


class BoostMixin:
    def __init__(self, **kwargs):
        self.boost_factor: float = kwargs.pop("boost_factor")
        super().__init__(**kwargs)

    def enhance_guidance(self, guidance, i):
        return guidance * self.boost_factor
