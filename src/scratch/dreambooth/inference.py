from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
import torch
import os
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the training script with config.yaml settings."
    )
    parser.add_argument("--token", type=str, help="user's token.")
    parser.add_argument("--prompt", type=str, help="inference prompt")
    parser.add_argument("--user-gender", type=str, help="A gender of user")
    args = parser.parse_args()

    token = args.token

    class_name = args.user_gender

    trained_model = str(
        os.path.join(
            "src/dreambooth/weights",
            token,
            "pytorch_lora_weights.bin",
        )
    )

    save_dir = f"src/dreambooth/data/results/{token}"

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
    pipe.load_lora_weights(trained_model, use_safetensors=False)

    n_steps = 75
    prompt = args.prompt
    prompt = prompt.replace(class_name, f"{token} {class_name}")

    latent_image = pipe(
        prompt=prompt, num_inference_steps=n_steps, num_images_per_prompt=4
    ).images
    for i in range(len(latent_image)):
        latent_image[i].save(f"{save_dir}/results_{i}.png")

    del latent_image
    del pipe
    torch.cuda.empty_cache()

    print(prompt)
