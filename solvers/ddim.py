import torch
import torchvision.transforms as T


from models.base import DiffusionModel
from solvers.base import Solver, SolverOutput, FFTMixin, GaussianMixin, BoostMixin


def ddim_step(
    scheduler,
    model_output,
    timestep,
    sample,
    prev_timestep=None,
    renoise_model_output=None,
):
    dtype = sample.dtype
    sample = sample.to(torch.float32)
    model_output = model_output.to(torch.float32)

    if renoise_model_output is None:
        renoise_model_output = model_output

    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t

    if scheduler.config.prediction_type == "epsilon":
        pred_orig = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_epsilon = renoise_model_output
    elif scheduler.config.prediction_type == "sample":
        pred_orig = model_output
        pred_epsilon = (
            sample - alpha_prod_t**0.5 * renoise_model_output
        ) / beta_prod_t**0.5
    elif scheduler.config.prediction_type == "v_prediction":
        pred_orig = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * renoise_model_output + (
            beta_prod_t**0.5
        ) * sample
    else:
        raise ValueError("Unknown prediction type")

    if prev_timestep is None:
        prev_timestep = (
            timestep
            - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        )
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )

    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
    pred_sample = alpha_prod_t_prev**0.5 * pred_orig + pred_sample_direction
    return pred_sample.to(dtype)


class DDIMSolver(Solver):

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
        self.prompt_embeds = torch.cat([self.net.get_text_embed([p]) for p in prompts])
        self.negative_prompt_embeds = self.net.get_text_embed([""]).expand(
            len(prompts), -1, -1
        )
        self.net.get_scheduler().set_timesteps(num_inference_steps=num_inference_steps)

        latents = torch.randn(
            (
                len(prompts),
                self.net.pipe.unet.config.in_channels,
                self.net.pipe.unet.config.sample_size,
                self.net.pipe.unet.config.sample_size,
            ),
            dtype=self.prompt_embeds.dtype,
            generator=torch.Generator().manual_seed(seed),
        ).to(self.net.device)
        timesteps = self.net.get_scheduler().timesteps
        decoder_input, denoising_output = self.denoising_loop(
            num_inference_steps=num_inference_steps,
            latents=latents,
            timesteps=timesteps,
        )

        images = [
            T.ToPILImage()(x)
            for x in self.net.decode_latents(decoder_input).to(torch.float32)
        ]
        return SolverOutput(images=images, denoising_output=denoising_output)

    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents

        for i, t in enumerate(timesteps):
            noise_pred_cond = self.ec(x[i], t)
            noise_pred_uncond = self.eo(x[i], t)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            x[i + 1] = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=noise_pred,
                timestep=t,
                sample=x[i],
            )

        return x[-1], None

    @torch.no_grad()
    def eo(self, latents, t):
        return self.net.run_unet(latents, self.negative_prompt_embeds, t)

    @torch.no_grad()
    def ec(self, latents, t):
        return self.net.run_unet(latents, self.prompt_embeds, t)


class DDIMSolverRichardson(DDIMSolver):
    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
    ):
        super().__init__(net, guidance_scale)

    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents
        xg = [None] * (num_inference_steps + 1)
        xg[0] = torch.zeros_like(latents)

        c_x = []
        c_xg = []

        for i, t in enumerate(timesteps):
            noise_pred_cond = self.ec(x[i] + xg[i], t)
            noise_pred_uncond = self.eo(x[i] + xg[i], t)
            x[i + 1] = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=noise_pred_cond,
                timestep=t,
                sample=x[i],
            )
            xg[i + 1] = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=(self.guidance_scale - 1)
                * (noise_pred_cond - noise_pred_uncond),
                timestep=t,
                sample=xg[i],
            )

            # get step doubling result
            x_r, xg_r = x[i], xg[i]
            t_next = timesteps[i + 1] if i + 1 < num_inference_steps else 0
            t_m = (t + t_next) // 2
            x_r = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=noise_pred_cond,
                timestep=t,
                sample=x_r,
                prev_timestep=t_m,
            )
            xg_r = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=(self.guidance_scale - 1)
                * (noise_pred_cond - noise_pred_uncond),
                timestep=t,
                sample=xg_r,
                prev_timestep=t_m,
            )
            noise_pred_cond = self.ec(x_r + xg_r, t_m)
            noise_pred_uncond = self.eo(x_r + xg_r, t_m)
            x_r = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=noise_pred_cond,
                timestep=t_m,
                sample=x_r,
                prev_timestep=t_next,
            )
            xg_r = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=(self.guidance_scale - 1)
                * (noise_pred_cond - noise_pred_uncond),
                timestep=t_m,
                sample=xg_r,
                prev_timestep=t_next,
            )

            # Richardson extrapolation
            denominator = (0.5 * (t_next - t) * (t_next - t)).item()
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
        return x[-1] + xg[-1], torch.stack([c_x, c_xg], dim=-1)


class LimitedDDIMSolver(DDIMSolver):
    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
        t_lo: int,
        t_hi: int,
    ):
        super().__init__(net, guidance_scale)
        self.t_lo = t_lo
        self.t_hi = t_hi

    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents

        for i, t in enumerate(timesteps):
            noise_pred_cond = self.ec(x[i], t)
            if i < self.t_lo or i >= self.t_hi:
                noise_pred = noise_pred_cond
            else:
                noise_pred_uncond = self.eo(x[i], t)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            x[i + 1] = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=noise_pred,
                timestep=t,
                sample=x[i],
            )

        return x[-1], None


class CFGPPDDIMSolver(DDIMSolver):
    def __init__(
        self,
        net: DiffusionModel,
        guidance_scale: float,
    ):
        super().__init__(net, guidance_scale)

    def denoising_loop(self, num_inference_steps, latents, timesteps):
        x = [None] * (num_inference_steps + 1)
        x[0] = latents

        for i, t in enumerate(timesteps):
            noise_pred_cond = self.ec(x[i], t)
            noise_pred_uncond = self.eo(x[i], t)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            x[i + 1] = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=noise_pred,
                timestep=t,
                sample=x[i],
                renoise_model_output=noise_pred_uncond,
            )

        return x[-1], None


class CachingDDIMSolver(DDIMSolver):
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

        cache = None

        for i, t in enumerate(timesteps):
            noise_pred_cond = self.ec(x[i], t)

            if self.use_cache(i):
                guidance = self.enhance_guidance(cache, i)
            else:
                noise_pred_uncond = self.eo(x[i], t)
                guidance = noise_pred_cond - noise_pred_uncond
                cache = guidance

            noise_pred = noise_pred_cond + (self.guidance_scale - 1) * guidance
            x[i + 1] = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=noise_pred,
                timestep=t,
                sample=x[i],
            )

        return x[-1], None


class FFTCachingDDIMSolver(FFTMixin, CachingDDIMSolver):
    pass


class GaussianCachingDDIMSolver(GaussianMixin, CachingDDIMSolver):
    pass


class BoostCachingDDIMSolver(BoostMixin, CachingDDIMSolver):
    pass


class MultirateDDIMSolver(DDIMSolver):
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

    def denoising_loop(self, num_inference_steps, latents, timesteps):

        x = [None] * (num_inference_steps + 1)
        x[0] = latents
        xg = [None] * (num_inference_steps + 1)
        xg[0] = torch.zeros_like(latents)

        for i, t in enumerate(timesteps):
            noise_pred_cond = self.ec(x[i] + xg[i], t)
            x[i + 1] = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=noise_pred_cond,
                timestep=t,
                sample=x[i],
            )

            if i in self.macro_steps:
                j = i + 1
                noise_pred_uncond = self.eo(x[i] + xg[i], t)
                guidance = noise_pred_cond - noise_pred_uncond
                while True:
                    t_j = timesteps[j] if j < num_inference_steps else 0
                    xg[j] = ddim_step(
                        scheduler=self.net.get_scheduler(),
                        model_output=(self.guidance_scale - 1) * guidance,
                        timestep=t,
                        sample=xg[i],
                        prev_timestep=t_j,
                    )
                    guidance = self.enhance_guidance(guidance, i)
                    if j in self.macro_steps or j >= num_inference_steps:
                        break
                    j += 1

        return x[-1] + xg[-1], None


class LimitedMultirateDDIMSolver(MultirateDDIMSolver):
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

    def denoising_loop(self, num_inference_steps, latents, timesteps):

        x = [None] * (num_inference_steps + 1)
        x[0] = latents
        xg = [None] * (num_inference_steps + 1)
        xg[0] = torch.zeros_like(latents)

        for i, t in enumerate(timesteps):
            noise_pred_cond = self.ec(x[i] + xg[i], t)
            x[i + 1] = ddim_step(
                scheduler=self.net.get_scheduler(),
                model_output=noise_pred_cond,
                timestep=t,
                sample=x[i],
            )

            if i in self.macro_steps:
                j = i + 1
                if i < self.t_lo or i >= self.t_hi:
                    guidance = torch.zeros_like(noise_pred_cond)
                else:
                    guidance = noise_pred_cond - self.eo(x[i] + xg[i], t)
                while True:
                    t_j = timesteps[j] if j < num_inference_steps else 0
                    xg[j] = ddim_step(
                        scheduler=self.net.get_scheduler(),
                        model_output=(self.guidance_scale - 1) * guidance,
                        timestep=t,
                        sample=xg[i],
                        prev_timestep=t_j,
                    )
                    guidance = self.enhance_guidance(guidance, i)
                    if j in self.macro_steps or j >= num_inference_steps:
                        break
                    j += 1

        return x[-1] + xg[-1], None


class LimitedBoostMultirateDDIMSolver(
    BoostMixin,
    LimitedMultirateDDIMSolver,
):
    pass
