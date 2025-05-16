import os

from accelerate import Accelerator
import torch
from tqdm import tqdm
import numpy as np

from utils import suppress_print

# FID
from cleanfid import fid

# CMMD
from measures.cmmd import distance, embedding, io_util

# CLIP Score
from torchmetrics.functional.multimodal.clip_score import clip_score
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# ImageReward
import ImageReward as RM

# PSNR, SSIM
from torchvision.io import read_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


accelerator = Accelerator()


@torch.no_grad()
def measure_fid(path: str, num_images: int):
    """
    Measure the FID score of the generated images.
    """
    with accelerator.main_process_first():
        name = f"coco30k_{num_images}"
        if not fid.test_stats_exists(name, "clean"):
            assert accelerator.is_main_process
            fid.make_custom_stats(
                name, "images/coco30k", num=num_images, device=accelerator.device
            )

    # TODO: distribute work across processes
    score = fid.compute_fid(
        path,
        dataset_name=name,
        dataset_split="custom",
        num_workers=0,
        device=accelerator.device,
        use_dataparallel=False,
        verbose=accelerator.is_main_process,
    )
    return float(score)


@torch.no_grad()
def measure_cmmd(path: str, model_name: str, batch_size: int, max_count: int):
    """
    Measure the CMMD score of the generated images.
    """
    embedding_model = embedding.ClipEmbeddingModel(model_name)
    name = f"images/coco30k_cmmd_{max_count}.npy"
    if not os.path.exists(name):
        ref_embs = io_util.compute_embeddings_for_dir(
            "images/coco30k",
            embedding_model,
            batch_size=batch_size,
            max_count=max_count,
        )
        np.save(name, ref_embs.cpu().numpy())
    else:
        ref_embs = torch.asarray(np.load(name), device=accelerator.device)

    eval_embs = io_util.compute_embeddings_for_dir(
        path, embedding_model, batch_size=batch_size, max_count=max_count
    )
    return distance.mmd(ref_embs, eval_embs).item()


class CLIPScoreDataset(Dataset):
    def __init__(self, path: str, prompts: list[str]):
        self.path = path
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        image = T.PILToTensor()(
            Image.open(os.path.join(self.path, f"{i:05d}.jpg")).convert("RGB")
        )
        return image, self.prompts[i]


@torch.no_grad()
def measure_clip_score(
    path: str,
    prompts: list[str],
    model_name: str,
    batch_size: int,
):
    """
    Measure the CLIP score of the generated images.
    """
    model = CLIPModel.from_pretrained(model_name).to(accelerator.device)
    processor = CLIPProcessor.from_pretrained(model_name)

    dataset = CLIPScoreDataset(path, prompts)
    dataloader = accelerator.prepare(DataLoader(dataset, batch_size=batch_size))
    if accelerator.is_main_process:
        dataloader = tqdm(dataloader, desc="CLIP Score")

    results = []
    for images, texts in dataloader:
        results.append(
            clip_score(
                images, list(texts), model_name_or_path=(lambda: (model, processor))
            )
        )
    gathered = accelerator.gather(torch.stack(results))
    return gathered.mean().item()


def measure_imagereward(path: str, prompts: list[str], model_name: str):
    """
    Measure the ImageReward score of the generated images.
    """
    with suppress_print():
        model = RM.load(model_name, device=accelerator.device)
    indices = list(range(len(prompts)))

    # apply_padding has a bug: huggingface/accelerator/pull/3490
    with accelerator.split_between_processes(indices) as index_split:
        result = []
        for i in (
            tqdm(index_split, desc="ImageReward")
            if accelerator.is_main_process
            else index_split
        ):
            score = model.score(prompts[i], os.path.join(path, f"{i:05d}.jpg"))
            result.append(score)

    gathered = accelerator.gather(torch.tensor(result, device=accelerator.device))
    return gathered.mean().item()


def measure_psnr_ssim(path: str, gt_path: str, num_images: int):
    """
    Measure the PSNR and SSIM of the generated images.
    """
    psnr = PeakSignalNoiseRatio(data_range=255.0).to(accelerator.device)
    ssim = StructuralSimilarityIndexMeasure(data_range=255.0).to(accelerator.device)

    indices = list(range(num_images))

    with accelerator.split_between_processes(indices) as index_split:
        psnr_results = []
        ssim_results = []

        for i in (
            tqdm(index_split, desc="PSNR/SSIM")
            if accelerator.is_main_process
            else index_split
        ):
            img = read_image(os.path.join(path, f"{i:05d}.jpg")).to(
                dtype=torch.float32, device=accelerator.device
            )[None]
            gt_img = read_image(os.path.join(gt_path, f"{i:05d}.jpg")).to(
                dtype=torch.float32, device=accelerator.device
            )[None]
            psnr_results.append(psnr(img, gt_img))
            ssim_results.append(ssim(img, gt_img))

    psnr_results = accelerator.gather(torch.stack(psnr_results))
    ssim_results = accelerator.gather(torch.stack(ssim_results))
    return psnr_results.mean().item(), ssim_results.mean().item()
