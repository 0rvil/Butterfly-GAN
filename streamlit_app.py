import streamlit as st
import torch
from torchvision.utils import make_grid
import numpy as np
import os
import glob
import imageio
from PIL import Image

# Import generator classes
from Models.vanilla_gan import Generator as VanillaGenerator
from Models.dcgan import DCGenerator
from Models.stylegan_lite import StyleGenerator

def denormalize(tensor):
    return (tensor + 1) / 2  # [-1, 1] → [0, 1]

# Define the available models and metadata
MODEL_REGISTRY = {
    "Vanilla_GAN": {
        "class": VanillaGenerator,
        "weights": "vanilla_gan_gen.pth",
        "max_images": 64
    },
    "DCGAN": {
        "class": DCGenerator,
        "weights": "dcgan_gen.pth",
        "max_images": 64
    },
    "StyleGAN_Lite": {
        "class": StyleGenerator,
        "weights": "stylegan_lite_gen.pth",
        "max_images": 32
    }
}

def load_generator(model_key, noise_dim, device):
    entry = MODEL_REGISTRY[model_key]
    if entry["class"] is None:
        st.warning(f"{model_key} is not yet available.")
        return None

    gen = entry["class"](noise_dim).to(device)
    try:
        gen.load_state_dict(torch.load(entry["weights"], map_location=device))
        gen.eval()
        return gen
    except FileNotFoundError:
        st.error(f"Model weights not found: {entry['weights']}")
        return None

def main():
    st.set_page_config(page_title="Butterfly GAN Viewer")
    st.title("Butterfly GAN Image Generator")

    noise_dim = 100
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Select model
    model_key = st.selectbox("Choose GAN model", list(MODEL_REGISTRY.keys()))
    model_meta = MODEL_REGISTRY[model_key]

    # Set image count slider
    num_images = st.slider("How many butterflies to generate?", 1, model_meta["max_images"], min(8, model_meta["max_images"]))

    if st.button("Generate"):
        gen = load_generator(model_key, noise_dim, device)
        if gen:
            with torch.no_grad():
                z = torch.randn(num_images, noise_dim, device=device)
                fake_imgs = gen(z).cpu()
                fake_imgs = denormalize(fake_imgs)

                grid = make_grid(fake_imgs, nrow=min(num_images, 8), padding=2)
                npimg = grid.permute(1, 2, 0).numpy()
                st.image(npimg, caption=f"Generated Butterflies ({model_key})", use_container_width=True)

    st.header("Training Progress GIF")

    @st.cache_data(show_spinner=True)
    def get_or_create_training_gif(model_key):
        folder = f"Outputs/{model_key}"
        gif_path = os.path.join(folder, "training_progress.gif")
        print(gif_path)
        # Check if GIF already exists
        if os.path.exists(gif_path):
            return gif_path

        # Otherwise, try to build from images
        # image_paths = sorted(glob.glob(f"{folder}/fake_epoch_*.png"), key=os.path.getmtime)
        # if not image_paths:
        #     return None

        # images = [Image.open(path) for path in image_paths]
        # imageio.mimsave(gif_path, images, duration=0.5, loop=0)
        return gif_path

    show_gif = st.checkbox("Show Training Progress")

    if show_gif:
        gif_path = get_or_create_training_gif(model_key)
        if gif_path is None:
            st.warning(f"No training output images or GIF found in 'Outputs/{model_key}/'.")
        else:
            st.image(gif_path, caption=f"Training Progress Over Epochs ({model_key})", use_container_width=True)


