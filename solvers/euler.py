import torch
from diffusers import EulerDiscreteScheduler

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline

from models.base import DiffusionModel
from solvers.base import Solver, SolverOutput


accelerator = Accelerator()


class EulerSolver(Solver):
    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
    ):
        super().__init__(net)
        self.guidance_scale = guidance_scale

        self.net.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.net.pipe.scheduler.config,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        seed: int,
        num_inference_steps: int,
    ) -> SolverOutput:
        pipe: StableDiffusionPipeline = self.net.pipe

        images = pipe(
            prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).images

        return SolverOutput(images=images)
