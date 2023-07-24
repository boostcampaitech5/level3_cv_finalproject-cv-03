from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
import torch
import os
from pathlib import Path


trained_model = "/opt/ml/stable-diffusion-xl/weights/qwer/num400/checkpoint-1000/pytorch_lora_weights.bin"

exp_name = trained_model.split("/")[6]
token_name = trained_model.split("/")[5]

save_dir = f"/opt/ml/stable-diffusion-xl/data/results/{token_name}/{exp_name}"
save_dir = Path(save_dir)
if not save_dir.exists():
    save_dir.mkdir(parents=True)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, force_upcast=False
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# load LoRA weight
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.unet.load_attn_procs(trained_model, use_safetensors=False)
pipe.load_lora_weights(trained_model, use_safetensors=False)

n_steps = 100
prompt = "A photo of qwer woman in ocean, blue sky, palm trees"

latent_image = pipe(
    prompt=prompt, num_inference_steps=n_steps, num_images_per_prompt=4
).images
for i in range(len(latent_image)):
    latent_image[i].save(f"{save_dir}/{exp_name}_{i}.png")

del latent_image
del pipe
torch.cuda.empty_cache()
