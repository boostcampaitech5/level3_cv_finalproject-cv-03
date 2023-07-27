import os
import shutil

from tqdm import tqdm
import wandb

import torch
import torch.nn.functional as F

from .utils.plot import make_image_grid


def training(
    args,
    accelerator,
    weight_dtype,
    model,
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

    save_path = ""

    for epoch in range(first_epoch, args.max_epochs):
        model.unet.train()
        model.text_encoder.train()
        model.text_encoder_2.train()
        for step, batch in enumerate(train_dataloader):
            imgs, texts = batch["imgs"], list(batch["texts"])

            # If you use resume, update the progress to resume.
            if args.resume and epoch == first_epoch and step < resume_step:
                if step % args.grad_accum == 0:
                    progress_bar.update(1)
                continue

            # Using accelerator
            with accelerator.accumulate(model):
                # Image -> latent
                latents = model.vae.encode(imgs).latent_dist.sample()
                latents *= model.vae.config.scaling_factor
                latents = latents.to(weight_dtype)

                # Prepare noise for forward process(diffusion process)
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    model.scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Forward process(diffusion process)
                noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                prompt_embeds, _, pooled_prompt_embeds, _ = model.encode_prompt(
                    prompt=texts
                )
                time_ids = model._get_add_time_ids(
                    (args.resolution[0], args.resolution[1]),
                    (0, 0),
                    (args.resolution[0], args.resolution[1]),
                    dtype=weight_dtype,
                )

                time_ids = time_ids.repeat(batch_size, 1)

                pred = model.unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={
                        "time_ids": time_ids.to(device=latents.device),
                        "text_embeds": pooled_prompt_embeds,
                    },
                ).sample

                loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

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
            valid(args, accelerator, model, epoch)


def valid(args, accelerator, model, epoch):
    print("Running validation")

    images = []
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    for i in range(len(args.valid_prompts)):
        image = model(
            prompt=args.valid_prompts[i],
            prompt_2=args.valid_prompts[i],
            height=args.resolution[0],
            width=args.resolution[0],
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            output_type="pil",
        ).images[0]
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
