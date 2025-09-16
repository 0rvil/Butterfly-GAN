# Butterfly GAN Project

Interactive Streamlit demo of three GANs trained to generate butterfly images:
Vanilla GAN, DCGAN, and StyleGAN Lite.

## How to run (local)
1. Create a virtual environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (mac/linux) or `.\.venv\Scripts\activate` (Windows)
2. Install requirements:
   - `pip install -r requirements.txt`
3. Run the app:
   - `streamlit run streamlit_app.py`

## Files
- `streamlit_app.py` — Streamlit frontend
- `vanilla_gan_gen.pth`, `dcgan_gen.pth`, `stylegan_lite_gen.pth` — model weights
- `Outputs/` — training images and gifs assets