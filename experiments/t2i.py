import os
import logging

from accelerate import Accelerator
from datasets import load_dataset
import torch
from tqdm import tqdm

import sacred
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from measures import (
    measure_fid,
    measure_cmmd,
    measure_clip_score,
    measure_imagereward,
    measure_psnr_ssim,
)
import models
import solvers
from utils import chunk


logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
sacred.SETTINGS["CAPTURE_MODE"] = "sys"

accelerator = Accelerator()
t2i = Experiment("t2i")


@t2i.config
def config():
    path = "images/generated"
    dataset = "sayakpaul/coco-30-val-2014"
    num_images = 30000
    batch_size = 30
    assert num_images % batch_size == 0, "num_images must be divisible by batch_size"
    seed = 42

    # Generation config
    solver_name = "Solver"
    solver_kwargs = {}
    model_name = "DiffusionModel"
    model_kwargs = {"model_key": "stable-diffusion-v1-5/stable-diffusion-v1-5"}
    num_inference_steps = 50

    assert torch.cuda.device_count() > 0, "No GPU found"

    # Evaluation config
    cmmd_model_name = "openai/clip-vit-large-patch14-336"
    clip_score_model_name = "openai/clip-vit-large-patch14"
    imagereward_model_name = "ImageReward-v1.0"
    gt_path = "images/cfg"


@t2i.capture
def generate_images(
    path: str,
    dataset: str,
    num_images: int,
    batch_size: int,
    solver_name: str,
    solver_kwargs: dict,
    model_name: str,
    model_kwargs: dict,
    num_inference_steps: int,
    seed: int,
) -> None:

    if os.path.exists(path):
        print(f"Path {path} already exists, using existing images.")
        return

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.mkdir(path)
    accelerator.wait_for_everyone()

    captions = load_dataset(dataset)["train"]["caption"][:num_images]
    indices = list(range(num_images))

    net: models.DiffusionModel = getattr(models, model_name)(
        device=accelerator.device, **model_kwargs
    )
    solver: solvers.Solver = getattr(solvers, solver_name)(net=net, **solver_kwargs)

    with accelerator.split_between_processes(indices) as index_split:
        batches = list(chunk(index_split, batch_size))
        for batch in tqdm(batches) if accelerator.is_main_process else batches:
            prompts = [captions[i] for i in batch]

            imgs = solver.generate(prompts, seed + batch[0], num_inference_steps).images
            for i, img in zip(batch, imgs):
                img.save(os.path.join(path, f"{i:05d}.jpg"))
            accelerator.wait_for_everyone()


@t2i.capture
def measure(
    path: str,
    dataset: str,
    num_images: int,
    batch_size: int,
    cmmd_model_name: str,
    clip_score_model_name: str,
    imagereward_model_name: str,
    gt_path: str,
) -> list[float]:

    captions = load_dataset(dataset)["train"]["caption"][:num_images]

    fid = measure_fid(path, num_images)
    cmmd = measure_cmmd(path, cmmd_model_name, batch_size, num_images)
    clip_score = measure_clip_score(path, captions, clip_score_model_name, batch_size)
    imagereward = measure_imagereward(path, captions, imagereward_model_name)
    psnr, ssim = measure_psnr_ssim(path, gt_path, num_images)

    if accelerator.is_main_process:
        print(f"FID: {fid}")
        print(f"CMMD: {cmmd}")
        print(f"CLIP Score: {clip_score}")
        print(f"ImageReward: {imagereward}")
        print(f"PSNR: {psnr}")
        print(f"SSIM: {ssim}")

    return {
        "FID": fid,
        "CMMD": cmmd,
        "CLIP Score": clip_score,
        "ImageReward": imagereward,
        "PSNR ": psnr,
        "SSIM": ssim,
    }


def assess(_config):
    if accelerator.is_main_process:
        print("Config:")
        print(_config)
    generate_images()
    return measure()


def setup_experiment(ex: Experiment):
    if accelerator.is_main_process:
        ex.observers.append(MongoObserver(url="localhost:27017"))
        ex.captured_out_filter = apply_backspaces_and_linefeeds
    ex.main(assess)


def main():
    setup_experiment(t2i)
    t2i.run_commandline()


if __name__ == "__main__":
    main()
