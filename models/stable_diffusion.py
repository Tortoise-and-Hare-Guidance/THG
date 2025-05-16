from diffusers import StableDiffusionPipeline
import diffusers
import torch

from models.base import DiffusionModel


class StableDiffusionModel(DiffusionModel):

    def __init__(self, device, model_key, scheduler_name: str):
        super().__init__(device)

        scheduler = getattr(diffusers, scheduler_name).from_pretrained(
            model_key, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_key,
            torch_dtype=torch.bfloat16,
            scheduler=scheduler,
            safety_checker=None,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def get_text_embed(self, prompt, **kwargs):
        text_input = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
            **kwargs
        )
        text_embeddings = self.pipe.text_encoder(
            text_input.input_ids.to(self.device),
        )[0]
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        latents = latents.to(self.pipe.vae.dtype)
        imgs = self.pipe.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def run_unet(self, latent, text_embeds, t):
        return self.pipe.unet(latent, t, encoder_hidden_states=text_embeds).sample

    def get_scheduler(self):
        return self.pipe.scheduler
