# main.py

import argparse
from Training import train_vanilla_gan, train_dcgan, train_stylegan
import streamlit_app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train-vanilla", "train-dcgan", "train-stylegan-lite", "ui"], default="ui")
    args = parser.parse_args()

    if args.mode == "train-vanilla":
        train_vanilla_gan.main()  # Train the Vanilla GAN
    elif args.mode == "train-dcgan":
        train_dcgan.main() # Train the DCGAN
    elif args.mode == "train-stylegan-lite":
        train_stylegan.main() # Train the StyleGAN
    elif args.mode == "ui":
        streamlit_app.main() # Render the UI

if __name__ == "__main__":
    main()
