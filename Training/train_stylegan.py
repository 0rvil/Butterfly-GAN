import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datetime import datetime

from Datasets.butterfly_dataset import ButterflyDataset
from Models.stylegan_lite import StyleGenerator, StyleDiscriminator
from Models.utils import weights_init

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "StyleGAN_Lite"

    checkpoint_dir = os.path.join("Checkpoints", model_name)
    output_dir = os.path.join("Outputs", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    noise_dim = 100
    lr = 2e-4
    beta1 = 0.0  # StyleGAN often uses 0.0 for beta1
    beta2 = 0.99
    batch_size = 64
    epochs = 100

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = ButterflyDataset(
        csv_file='Data/butterfly_image_dataset/Training_set.csv',
        root_dir='Data/butterfly_image_dataset/train',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    gen = StyleGenerator(noise_dim).to(device)
    disc = StyleDiscriminator().to(device)
    gen.apply(weights_init)
    disc.apply(weights_init)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))
    criterion = nn.BCELoss()

    start_time = datetime.now()

    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            bs = real.size(0)

            real_labels = torch.ones(bs, device=device)
            fake_labels = torch.zeros(bs, device=device)

            # Train Discriminator
            noise = torch.randn(bs, noise_dim, device=device)
            fake = gen(noise)

            disc_real = disc(real)
            loss_real = criterion(disc_real, real_labels)

            disc_fake = disc(fake.detach())
            loss_fake = criterion(disc_fake, fake_labels)

            loss_disc = (loss_real + loss_fake) / 2
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake)
            loss_gen = criterion(output, real_labels)
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                elapsed = datetime.now() - start_time
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Elapsed: {elapsed} | "
                      f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} "
                      f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

        # Save sample images
        with torch.no_grad():
            fake = gen(torch.randn(64, noise_dim, device=device)).detach().cpu()
            save_image(fake, os.path.join(output_dir, f"fake_epoch_{model_name}_{epoch}.png"), normalize=True, nrow=8)

        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(gen.state_dict(), os.path.join(checkpoint_dir, f"gen_epoch_{model_name}_{epoch + 1}.pth"))

    # Final generator save
    torch.save(gen.state_dict(), os.path.join(checkpoint_dir, f"{model_name}_gen.pth"))