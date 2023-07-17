import os

from PIL import Image


def make_image_grid(args, epoch, imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    save_img_path = os.path.join(
        args.base_path, args.exp_path, args.exp_name, "results", args.save_img_path
    )
    os.makedirs(save_img_path, exist_ok=True)
    grid.save(os.path.join(save_img_path, f"{epoch}.png"))
