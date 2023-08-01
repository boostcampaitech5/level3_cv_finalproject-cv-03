from io import BytesIO
import random

import pandas as pd
import requests
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset


def collate_fn(examples):
    imgs = [example["imgs"] for example in examples]
    texts = [example["texts"] for example in examples]

    imgs = torch.stack(imgs)
    imgs = imgs.to(memory_format=torch.contiguous_format).float()
    texts = np.stack(texts)

    batch = {
        "imgs": imgs,
        "texts": texts,
    }

    return batch


def valid_prompt(txt_path, max_prompt):
    assert max_prompt <= 16

    text = []
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines[:max_prompt]:
            text.append(line.strip())

    return text


class MelonDatasetSDXL(Dataset):
    def __init__(self, csv_path, transform, tokenizers, unet_added_cond_kwargs):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.tokenizer_one = tokenizers[0]
        self.tokenizer_two = tokenizers[1]
        self.unet_added_cond_kwargs = unet_added_cond_kwargs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        # Image
        img_url = data["img_url"]
        res = requests.get(img_url)
        img = Image.open(BytesIO(res.content)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        texts = data["text"].split("|")

        example = {}
        example["imgs"] = img
        example["texts"] = random.choice(texts)

        return example
