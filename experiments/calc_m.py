import os
from pprint import pprint
import logging

from accelerate import Accelerator
from datasets import load_dataset
import torch
from tqdm import tqdm

import sacred
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

import models
import solvers
from utils import chunk


logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
sacred.SETTINGS["CAPTURE_MODE"] = "sys"

accelerator = Accelerator()
t2i = Experiment("calc_m")


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
        raise FileExistsError(f"Path {path} already exists, please remove it first.")

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
        results = []
        for batch in tqdm(batches) if accelerator.is_main_process else batches:
            prompts = [captions[i] for i in batch]

            output = solver.generate(prompts, seed + batch[0], num_inference_steps)
            for i, img in zip(batch, output.images):
                img.save(os.path.join(path, f"{i:05d}.jpg"))

            results.append(output.denoising_output)  # [batch, num_steps, 2]
            accelerator.wait_for_everyone()

    gathered = accelerator.gather(torch.cat(results))  # [num_images, num_steps, 2]
    if accelerator.is_main_process:
        torch.save(gathered, os.path.join(path, "denoising_output.pt"))
        print(f"Saved denoising output to {os.path.join(path, 'denoising_output.pt')}")
    accelerator.wait_for_everyone()


def assess(_config):
    if accelerator.is_main_process:
        print("Config:")
        pprint(_config)
    generate_images()


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
