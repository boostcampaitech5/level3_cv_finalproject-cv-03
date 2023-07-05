from io import BytesIO
import random

import numpy as np
import pandas as pd
import requests
from PIL import Image

import torch
from torch.utils.data import Dataset


def valid_prompt(txt_path, max_prompt):
    assert max_prompt <= 16

    text = []
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines[:max_prompt]:
            text.append(line.strip())

    return text


class MelonDataset(Dataset):
    def __init__(self, csv_path, transform, tokenizer):
        super().__init__()
        self.transform = transform
        self.tokenizer = tokenizer
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        # Image
        img_url = data["img_url"]
        res = requests.get(img_url)
        img = Image.open(BytesIO(res.content))
        img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Text
        texts = data["text"].split("|")
        inputs = self.tokenizer(
            [random.choice(texts)],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text = inputs.input_ids

        # Output type
        img.to(memory_format=torch.contiguous_format).float()
        text = text.squeeze()

        return img, text
