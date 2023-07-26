from typing import Dict

import torch


# Computes additional embeddings required by the SDXL UNet.
def compute_additional_embeddings(height, width, weight_dtype):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])

    add_time_ids = add_time_ids.to(dtype=weight_dtype)
    unet_added_cond_kwargs = {"time_ids": add_time_ids}

    return unet_added_cond_kwargs
