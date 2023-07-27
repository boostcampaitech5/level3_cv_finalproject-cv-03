import torch

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)


pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# TODO
ckpt = "~/src/stable_diffusion_xl/experiments/[tag]ExpName/checkpoints/checkpoint-1"
pipe.load_lora_weights(ckpt, use_safetensors=False)

prompt = "Music Album Cover of 'Nothing Better' by 'IU'"
generator = torch.Generator(device="cuda").manual_seed(3)
image = pipe(
    prompt=prompt,
    prompt_2=prompt,
    height=1024,
    width=1024,
    num_inference_steps=100,
    generator=generator,
).images[0]

image.save("test.png")
