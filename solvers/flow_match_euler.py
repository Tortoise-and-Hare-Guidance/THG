import torch
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline

from models.base import DiffusionModel
from solvers.base import Solver, SolverOutput, FFTMixin, GaussianMixin, BoostMixin


accelerator = Accelerator()


class FlowMatchEulerSolver(Solver):

    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
    ):
        super().__init__(net)
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        seed: int,
        num_inference_steps: int,
    ) -> SolverOutput:

        pipe: StableDiffusion3Pipeline = self.net.pipe

        height = width = pipe.default_sample_size * pipe.vae_scale_factor
        batch_size = len(prompts)
        num_images_per_prompt = 1
        device = accelerator.device

        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=None,
            device=device,
        )

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            pipe.transformer.config.in_channels,
            height,
            width,
            self.prompt_embeds.dtype,
            device,
            torch.Generator().manual_seed(seed),
            None,
        )

        pipe.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps, device=device
        )
        timesteps = pipe.scheduler.timesteps

        decoder_input, denoising_output = self.denoising_loop(
            num_inference_steps=num_inference_steps,
            latents=latents,
            timesteps=timesteps,
        )
        decoder_input = (
            decoder_input / pipe.vae.config.scaling_factor
        ) + pipe.vae.config.shift_factor
        images = pipe.vae.decode(decoder_input, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images)

        pipe.maybe_free_model_hooks()
        output = SolverOutput(images=images, denoising_output=denoising_output)
        return output

    def ec(self, latents, t):
        return self.net.pipe.transformer(
            hidden_states=latents,
            timestep=t.expand(latents.shape[0]),
            encoder_hidden_states=self.prompt_embeds,
            pooled_projections=self.pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    def eo(self, latents, t):
        return self.net.pipe.transformer(
            hidden_states=latents,
            timestep=t.expand(latents.shape[0]),
            encoder_hidden_states=self.negative_prompt_embeds,
            pooled_projections=self.negative_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents

        pipe: StableDiffusion3Pipeline = self.net.pipe
        for i, t in enumerate(timesteps):
            noise_pred_uncond = self.eo(x[i], t)
            noise_pred_text = self.ec(x[i], t)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            dt = pipe.scheduler.sigmas[i + 1] - pipe.scheduler.sigmas[i]
            x[i + 1] = FlowMatchEulerSolver.step(x[i], dt, noise_pred)

        return x[-1], None

    @staticmethod
    def step(x, dt, v):
        dtype = x.dtype
        x = x.to(torch.float32)
        return (x + dt * v).to(dtype)


class LimitedFlowMatchEulerSolver(FlowMatchEulerSolver):
    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
        t_lo: int,
        t_hi: int,
    ):
        super().__init__(net, guidance_scale)
        assert t_lo < t_hi
        self.t_lo = t_lo
        self.t_hi = t_hi

    @torch.no_grad()
    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents

        pipe: StableDiffusion3Pipeline = self.net.pipe
        for i, t in enumerate(timesteps):
            noise_pred_text = self.ec(x[i], t)
            if i < self.t_lo or i >= self.t_hi:
                noise_pred = noise_pred_text
            else:
                noise_pred_uncond = self.eo(x[i], t)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            dt = pipe.scheduler.sigmas[i + 1] - pipe.scheduler.sigmas[i]
            x[i + 1] = FlowMatchEulerSolver.step(x[i], dt, noise_pred)

        return x[-1], None


class FlowMatchEulerSolverRichardson(FlowMatchEulerSolver):
    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
    ):
        super().__init__(net, guidance_scale)

    @torch.no_grad()
    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents
        xg = [None] * (num_inference_steps + 1)
        xg[0] = torch.zeros_like(latents)

        pipe: StableDiffusion3Pipeline = self.net.pipe
        c_x = []
        c_xg = []

        for i, t in enumerate(timesteps):
            noise_pred_uncond = self.eo(x[i] + xg[i], t)
            noise_pred_text = self.ec(x[i] + xg[i], t)
            dt = pipe.scheduler.sigmas[i + 1] - pipe.scheduler.sigmas[i]
            x[i + 1] = FlowMatchEulerSolver.step(x[i], dt, noise_pred_text)
            xg[i + 1] = FlowMatchEulerSolver.step(
                xg[i],
                dt,
                (self.guidance_scale - 1) * (noise_pred_text - noise_pred_uncond),
            )

            # get step doubling result
            x_r, xg_r = x[i], xg[i]
            sigma_m = (pipe.scheduler.sigmas[i] + pipe.scheduler.sigmas[i + 1]) / 2
            x_r = FlowMatchEulerSolver.step(
                x_r, sigma_m - pipe.scheduler.sigmas[i], noise_pred_text
            )
            xg_r = FlowMatchEulerSolver.step(
                xg_r,
                sigma_m - pipe.scheduler.sigmas[i],
                (self.guidance_scale - 1) * (noise_pred_text - noise_pred_uncond),
            )
            noise_pred_uncond = self.eo(x_r + xg_r, pipe.scheduler._sigma_to_t(sigma_m))
            noise_pred_text = self.ec(x_r + xg_r, pipe.scheduler._sigma_to_t(sigma_m))
            x_r = FlowMatchEulerSolver.step(
                x_r, pipe.scheduler.sigmas[i + 1] - sigma_m, noise_pred_text
            )
            xg_r = FlowMatchEulerSolver.step(
                xg_r,
                pipe.scheduler.sigmas[i + 1] - sigma_m,
                (self.guidance_scale - 1) * (noise_pred_text - noise_pred_uncond),
            )

            # Richardson extrapolation
            denominator = (0.5 * dt * dt).item()
            c_x.append(
                (x[i + 1] - x_r).flatten(start_dim=1).norm(dim=1, dtype=torch.float32)
                / denominator
            )
            c_xg.append(
                (xg[i + 1] - xg_r).flatten(start_dim=1).norm(dim=1, dtype=torch.float32)
                / denominator
            )

        c_x = torch.stack(c_x, dim=-1)
        c_xg = torch.stack(c_xg, dim=-1)
        return x[-1] + xg[-1], torch.stack([c_x, c_xg], dim=-1)  # [batch, num_steps, 2]


class CachingFlowMatchEulerSolver(FlowMatchEulerSolver):
    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
        caching_interval: int,
        caching_start: int,
    ):
        super().__init__(net, guidance_scale)
        self.caching_interval = caching_interval
        self.caching_start = caching_start

    def use_cache(self, i):
        return (i >= self.caching_start) and (
            (i - self.caching_start) % self.caching_interval != 0
        )

    def enhance_guidance(self, guidance, i):
        return guidance

    @torch.no_grad()
    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents

        pipe: StableDiffusion3Pipeline = self.net.pipe
        cache = None

        for i, t in enumerate(timesteps):
            noise_pred_text = self.ec(x[i], t)

            if self.use_cache(i):
                guidance = self.enhance_guidance(cache, i)
            else:
                noise_pred_uncond = self.eo(x[i], t)
                guidance = noise_pred_text - noise_pred_uncond
                cache = guidance

            dt = pipe.scheduler.sigmas[i + 1] - pipe.scheduler.sigmas[i]
            x[i + 1] = FlowMatchEulerSolver.step(
                x[i], dt, noise_pred_text + (self.guidance_scale - 1) * guidance
            )

        return x[-1], None


class FFTCachingFlowMatchEulerSolver(FFTMixin, CachingFlowMatchEulerSolver):
    pass


class GaussianCachingFlowMatchEulerSolver(GaussianMixin, CachingFlowMatchEulerSolver):
    pass


class BoostCachingFlowMatchEulerSolver(BoostMixin, CachingFlowMatchEulerSolver):
    pass


class MultirateFlowMatchEulerSolver(FlowMatchEulerSolver):
    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
        macro_steps: list[int],
    ):
        super().__init__(net, guidance_scale)
        self.macro_steps = macro_steps

    def enhance_guidance(self, guidance, i):
        return guidance

    @torch.no_grad()
    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents
        xg = [None] * (num_inference_steps + 1)
        xg[0] = torch.zeros_like(latents)

        pipe: StableDiffusion3Pipeline = self.net.pipe
        for i, t in enumerate(timesteps):
            noise_pred_text = self.ec(x[i] + xg[i], t)
            dt = pipe.scheduler.sigmas[i + 1] - pipe.scheduler.sigmas[i]
            x[i + 1] = FlowMatchEulerSolver.step(x[i], dt, noise_pred_text)

            if i in self.macro_steps:
                j = i + 1
                noise_pred_uncond = self.eo(x[i] + xg[i], t)
                guidance = noise_pred_text - noise_pred_uncond
                while True:
                    dt = pipe.scheduler.sigmas[j] - pipe.scheduler.sigmas[i]
                    xg[j] = FlowMatchEulerSolver.step(
                        xg[i],
                        dt,
                        (self.guidance_scale - 1) * guidance,
                    )
                    guidance = self.enhance_guidance(guidance, i)
                    if j in self.macro_steps or j >= num_inference_steps:
                        break
                    j += 1

        return x[-1] + xg[-1], None


class LimitedMultirateFlowMatchEulerSolver(MultirateFlowMatchEulerSolver):
    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
        macro_steps: list[int],
        t_lo: int,
        t_hi: int,
    ):
        super().__init__(net, guidance_scale, macro_steps)
        self.t_lo = t_lo
        self.t_hi = t_hi

    @torch.no_grad()
    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents
        xg = [None] * (num_inference_steps + 1)
        xg[0] = torch.zeros_like(latents)

        pipe: StableDiffusion3Pipeline = self.net.pipe
        for i, t in enumerate(timesteps):
            noise_pred_text = self.ec(x[i] + xg[i], t)
            dt = pipe.scheduler.sigmas[i + 1] - pipe.scheduler.sigmas[i]
            x[i + 1] = FlowMatchEulerSolver.step(x[i], dt, noise_pred_text)

            if i in self.macro_steps:
                j = i + 1
                if i < self.t_lo or i >= self.t_hi:
                    guidance = torch.zeros_like(noise_pred_text)
                else:
                    guidance = noise_pred_text - self.eo(x[i] + xg[i], t)
                while True:
                    dt = pipe.scheduler.sigmas[j] - pipe.scheduler.sigmas[i]
                    xg[j] = FlowMatchEulerSolver.step(
                        xg[i],
                        dt,
                        (self.guidance_scale - 1) * guidance,
                    )
                    guidance = self.enhance_guidance(guidance, i)
                    if j in self.macro_steps or j >= num_inference_steps:
                        break
                    j += 1

        return x[-1] + xg[-1], None


class LimitedBoostMultirateFlowMatchEulerSolver(
    BoostMixin, LimitedMultirateFlowMatchEulerSolver
):
    pass
