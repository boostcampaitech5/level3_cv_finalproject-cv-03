import os
import shutil

from tqdm import tqdm
import wandb

import torch
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline

from utils.training import compute_snr
from utils.plot import make_image_grid


def train(
    args,
    accelerator,
    weight_dtype,
    noise_scheduler,
    tokenizer,
    text_encoder,
    vae,
    unet,
    ema_unet,
    train_dataloader,
    optimizer,
    lr_scheduler,
    global_step,
    first_epoch,
    resume_step,
):
    print(f"Running training: {first_epoch}(start) >> {args.max_epochs}(finish)")

    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.max_epochs):
        unet.train()
        train_loss = 0.0
        for step, (imgs, texts) in enumerate(train_dataloader):
            # If you use resume, update the progress to resume.
            if args.resume and epoch == first_epoch and step < resume_step:
                if step % args.grad_accum == 0:
                    progress_bar.update(1)
                continue

            # Using accelerator
            with accelerator.accumulate(unet):
                # Image -> latent
                latents = vae.encode(imgs.to(weight_dtype)).latent_dist.sample()
                latents *= vae.config.scaling_factor

                # Prepare noise for forward process(diffusion process)
                noise = torch.randn_like(latents)
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                # Change how you put noise(lightness VS Darkness)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                batch_size = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Change how you put noise(Speed up while maintaining FID score)
                if args.input_perturbation:
                    noise = noise + args.input_perturbation * torch.rand_like(noise)

                # Forward process(diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(texts)[0]

                # Predict the noise residual and compute loss
                pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Using  Min-SNR Weighting Strategy effectively balances the collisions between time steps by adjusting the loss weights of time steps based on the clamped signal-to-noise ratio.
                # Get faster convergence
                if args.snr_gamma is None:
                    loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(pred.float(), noise.float(), reduction="none")
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Compute train loss
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.grad_accum

                # Backpropagation
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # EMA(Exponential Moving Average) : Better results and more reliable training
            if args.use_ema:
                ema_unet.step(unet)
            progress_bar.update(1)
            global_step += 1
            train_loss = 0.0

            # Save chekpoint (MAX :args.ckpt_max)
            if global_step % args.ckpt_step == 0:
                if args.ckpt_max is not None:
                    any_files = os.listdir(args.output_dir)
                    ckpts = [
                        any_file
                        for any_file in any_files
                        if any_file.startswith("checkpoint")
                    ]
                    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))

                    if len(ckpts) >= args.ckpt_max:
                        n_remove_ckpt = len(ckpts) - args.ckpt_max + 1
                        remove_ckpts = ckpts[:n_remove_ckpt]

                        for remove_ckpt in remove_ckpts:
                            remove_ckpt = os.path.join(args.output_dir, remove_ckpt)
                            shutil.rmtree(remove_ckpt)
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            logs = {
                "Step Loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if epoch % args.valid_epoch == 0:
            valid(
                args,
                accelerator,
                tokenizer,
                text_encoder,
                vae,
                unet,
                weight_dtype,
                epoch,
            )


def valid(args, accelerator, tokenizer, text_encoder, vae, unet, weight_dtype, epoch):
    print("Running validation")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # https://facebookresearch.github.io/xformers/what_is_xformers.html
    if args.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.valid_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(
                prompt=args.valid_prompts[i],
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[
                0
            ]  # Default: 50
        images.append(image)

    if args.save_img_path is not None:
        make_image_grid(args, epoch, images, args.grid_size[0], args.grid_size[1])

    if args.use_wandb:
        wandb.log(
            {
                "Validation": [
                    wandb.Image(image, caption=f"{i}: {args.valid_prompts[i]}")
                    for i, image in enumerate(images)
                ]
            }
        )

    # Memory issues
    del pipeline
    torch.cuda.empty_cache()

    return images
