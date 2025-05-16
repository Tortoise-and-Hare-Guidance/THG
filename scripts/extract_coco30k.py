import os
import argparse

from tqdm.auto import tqdm
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="sayakpaul/coco-30-val-2014")
    return parser.parse_args()


def main(args):
    dataset = load_dataset(args.path, split="train")

    os.mkdir("images/coco30k")
    for i, x in enumerate(tqdm(dataset)):
        x["image"].save(f"images/coco30k/{i:05d}.jpg")


if __name__ == "__main__":
    main(parse_args())
