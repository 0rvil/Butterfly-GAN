import streamlit as st
import torch
from torchvision.utils import make_grid
import numpy as np
import os
import glob
import sys
sys.path.append(os.path.join(os.getcwd(), 'Models'))
import dnnlib
import legacy
import imageio
from PIL import Image

# Import generator classes
from Models.vanilla_gan import Generator as VanillaGenerator
from Models.dcgan import DCGenerator
from Models.stylegan_lite import StyleGANGenerator

def denormalize(tensor):
    return torch.clamp((tensor + 1) / 2, min=0.0, max=1.0)  # [-1, 1] → [0, 1]

# Define the available models and metadata
MODEL_REGISTRY = {
    "Vanilla_GAN": {
        "class": VanillaGenerator,
        "loader": "load_generator_legacy",
        "weights": "vanilla_gan_gen.pth",
        "max_images": 64,
        'folder': "vanilla_gan"
    },
    "DCGAN": {
        "class": DCGenerator,
        "loader": "load_generator_legacy",
        "weights": "dcgan_gen.pth",
        "max_images": 64,
        'folder': "dcgan"
    },
    "StyleGAN_Lite": {
        "class": StyleGANGenerator,
        "loader": "load_generator_legacy",
        "weights": "stylegan_lite_gen.pth",
        "max_images": 32,
        'folder': "stylegan_lite"
    },
    "StyleGAN_Fine_Tuning": {
        "class": None,
        "loader": "load_generator_pro",
        "weights": "stylegan-finetuning-000500.pkl",
        "max_images": 64,
        "folder": "StyleGAN_Fine_Tuning"
    }
}

def load_generator_legacy(model_key, noise_dim, device):
    entry = MODEL_REGISTRY[model_key]
    gen = entry["class"](noise_dim).to(device)
    try:
        checkpoint = torch.load(entry["weights"], map_location=device)
        if isinstance(checkpoint, dict) and 'gen' in checkpoint:
            gen.load_state_dict(checkpoint['gen'])
        else:
            gen.load_state_dict(checkpoint)
            
        gen.eval()
        st.toast(f"Successfully loaded {model_key}!")
        return gen
    except FileNotFoundError:
        st.error(f"Model weights not found: {entry['weights']}")
        return None
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None
    
# Generator for Transfer Learning, Fine Tuned StyleGAN model
def load_generator_pro(model_key, device):
    entry = MODEL_REGISTRY[model_key]
    try:
        
        with dnnlib.util.open_url(entry['weights']) as f:
            # 'G_ema' is the generator with "Exponential Moving Average" weights, which are the highest quality.
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
        
        G.eval()
        st.toast(f"Successfully loaded {model_key}!")
        return G
    except FileNotFoundError:
        st.error(f"Model weights not found: {entry['weights']}")
        return None
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

def main():
    st.set_page_config(page_title="Butterfly GAN Viewer")
    st.title("Butterfly GAN Image Generator")

    default_noise_dim = 100
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Select model
    model_key = st.selectbox("Choose GAN model", list(MODEL_REGISTRY.keys()))
    model_meta = MODEL_REGISTRY[model_key]

    # Set image count slider
    num_images = st.slider("How many butterflies to generate?", 1, model_meta["max_images"], min(8, model_meta["max_images"]))

    if st.button("Generate"):
        gen = None
        if model_meta["loader"] == "load_generator_pro":
            gen = load_generator_pro(model_key, device)
        elif model_meta["loader"] == "load_generator_legacy":
            gen = load_generator_legacy(model_key, default_noise_dim, device)
        

        if gen:
                with torch.no_grad():
                    
                    # This block checks *which* model is loaded and uses
                    # the correct z_dim and function call for it.
                    
                    if model_key == "StyleGAN_Fine_Tuning":
                        # Fine Tuned model uses z_dim of 512, which is stored in gen.z_dim
                        z = torch.randn(num_images, gen.z_dim, device=device)
                        # Fine Tuned model requires two arguments (z, labels), we pass None for labels
                        fake_imgs = gen(z, None).cpu()
                    else:
                        # All other models use noise_dim=100
                        z = torch.randn(num_images, default_noise_dim, device=device)
                        # And are called with just z
                        fake_imgs = gen(z).cpu()

                    fake_imgs = denormalize(fake_imgs)
                    grid = make_grid(fake_imgs, nrow=min(num_images, 8), padding=2)
                    npimg = grid.permute(1, 2, 0).numpy()
                    st.image(npimg, caption=f"Generated Butterflies ({model_key})", use_container_width=True)

    st.header("Training Progress GIF")

    @st.cache_data(show_spinner=True)
    def get_or_create_training_gif(model_key):
        folder = f"Outputs/{MODEL_REGISTRY[model_key]['folder']}"
        gif_path = os.path.join(folder, "training_progress.gif")
        print(gif_path)
        # Check if GIF already exists
        if os.path.exists(gif_path):
            return gif_path
        else:
            # Otherwise, try to build from images
            # Use glob to find all fake epoch images
            image_paths = sorted(glob.glob(f"{folder}/fake_epoch_*.png"), key=os.path.getmtime)
            
            # Check for the transfer learning model's fakes
            # (The transfer learning model names fakes 'fakes000000.png', etc.)
            if not image_paths:
                image_paths = sorted(glob.glob(f"{folder}/fakes000*.png"), key=os.path.getmtime)

            if not image_paths:
                return None

            try:
                images = [Image.open(path) for path in image_paths]
                imageio.mimsave(gif_path, images, duration=0.5, loop=0)
            except Exception as e:
                st.warning(f"Could not create GIF. Error: {e}")
                return None
        return gif_path

    show_gif = st.checkbox("Show Training Progress")

    if show_gif:
        gif_path = get_or_create_training_gif(model_key)
        if gif_path is None:
            st.warning(f"No training output images or GIF found in 'Outputs/{model_key}/'.")
        else:
            st.image(gif_path, caption=f"Training Progress Over Epochs ({model_key})", use_container_width=True)


