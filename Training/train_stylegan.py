import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from datetime import datetime

from Datasets.butterfly_dataset import ButterflyDataset
from Models.stylegan_lite import StyleGANGenerator, StyleGANDiscriminator
from Models.utils import weights_init
from Models.diff_aug import DiffAugment

def discriminator_hinge_loss(real_pred, fake_pred):
    loss_real = torch.mean(F.relu(1.0 - real_pred))
    loss_fake = torch.mean(F.relu(1.0 + fake_pred))
    return loss_real + loss_fake

def generator_hinge_loss(fake_pred):
    return -torch.mean(fake_pred)

def g_path_regularization(fake_img, latents, mean_path_length, decay=0.01):
    #Gradients of image pixel values w.r.t w latents
    noise = torch.randn_like(fake_img) / fake_img.shape[2]
    grad = torch.autograd.grad(
        outputs=[(fake_img * noise).sum()],
        inputs = latents,
        create_graph=True
    )[0]

    #Path lenghts
    path_lengths = torch.sqrt(grad.pow(2).sum(dim=1))

    #Penalty
    path_penalty = (path_lengths - mean_path_length).pow(2).mean()

    #Update moving average of path lengths
    mean_path_lengths = (1-decay) * mean_path_length + decay * path_lengths.mean().detach()

    return path_penalty, mean_path_lengths

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "StyleGAN_Lite"

    checkpoint_dir = os.path.join("Checkpoints", model_name)
    output_dir = os.path.join("Outputs", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    noise_dim = 100
    lr_g = 1e-5
    lr_d = 1e-5
    beta1 = 0.0  
    beta2 = 0.99
    batch_size = 64
    epochs = 300
    g_every = 4
    path_reg_gamma = 2.0
    mean_path_length = torch.tensor(0.0, device=device)
    last_g_loss = 0.0
    aug_policy = 'color,translation'

    CONTINUE_CHECKPOINT = os.path.join(checkpoint_dir, "gen_epoch_stylegan_lite_100.pth")
    FULL_CHECKPOINT_PATH = os.path.join(checkpoint_dir, f"{model_name}_full.pth")
    GOOD_CHECKPOINT = os.path.join(checkpoint_dir, "gen_epoch_StyleGAN_Lite_160.pth")

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

    gen = StyleGANGenerator(z_dim=noise_dim, w_dim=512).to(device)
    disc = StyleGANDiscriminator().to(device)


    opt_gen = optim.Adam(gen.parameters(), lr=lr_g, betas=(beta1, beta2))
    opt_disc = optim.Adam(disc.parameters(), lr=lr_d, betas=(beta1, beta2))

    start_epoch = 0
    mean_path_length = torch.tensor(0.0, device=device)

    if os.path.exists(GOOD_CHECKPOINT):
        print(f"--- ROLLING BACK to good checkpoint from {GOOD_CHECKPOINT} ---")
        checkpoint = torch.load(GOOD_CHECKPOINT, map_location=device)
        gen.load_state_dict(checkpoint['gen'])
        disc.load_state_dict(checkpoint['disc'])
        opt_gen.load_state_dict(checkpoint['opt_gen'])
        opt_disc.load_state_dict(checkpoint['opt_disc'])
        mean_path_length = checkpoint['mean_path_length']
        start_epoch = checkpoint['epoch']

        # CRITICAL: We must update the optimizers' learning rates
        # to our new, lower value
        for param_group in opt_gen.param_groups:
            param_group['lr'] = lr_g
        for param_group in opt_disc.param_groups:
            param_group['lr'] = lr_d
            
        # IMPORTANT: Overwrite the corrupted "latest" file
        # with this good checkpoint to secure our state.
        torch.save(checkpoint, FULL_CHECKPOINT_PATH)
        print(f"--- Successfully rolled back. State saved to {FULL_CHECKPOINT_PATH} ---")

    elif os.path.exists(FULL_CHECKPOINT_PATH):
        print(f"--- Loading FULL checkpoint from {FULL_CHECKPOINT_PATH} ---")
        checkpoint = torch.load(FULL_CHECKPOINT_PATH, map_location=device)
        gen.load_state_dict(checkpoint['gen'])
        disc.load_state_dict(checkpoint['disc'])
        opt_gen.load_state_dict(checkpoint['opt_gen'])
        opt_disc.load_state_dict(checkpoint['opt_disc'])
        mean_path_length = checkpoint['mean_path_length']
        start_epoch = checkpoint['epoch']
    elif os.path.exists(CONTINUE_CHECKPOINT):
        print(f"--- Loading *generator-only* checkpoint from {CONTINUE_CHECKPOINT} ---")
        gen.load_state_dict(torch.load(CONTINUE_CHECKPOINT, map_location=device))
        start_epoch = 100 
    else:
        print("--- No checkpoint found, starting from scratch ---")
        gen.apply(weights_init)
        disc.apply(weights_init)

    start_time = datetime.now()
    print(f"--- Starting training from epoch {start_epoch} ---")

    for epoch in range(start_epoch, epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            bs = real.size(0)

            # Train Discriminator
            noise = torch.randn(bs, noise_dim, device=device)

            w = gen.mapping(noise)
            fake = gen.synthesis(w)


            real_aug = DiffAugment(real, policy=aug_policy)
            fake_aug = DiffAugment(fake.detach(), policy=aug_policy)


            real_aug.requires_grad_()
            disc_real = disc(real_aug)
            disc_fake = disc(fake_aug)
            loss_disc = discriminator_hinge_loss(disc_real, disc_fake)

            # R1 Regularization
            grad_real = torch.autograd.grad(
                outputs = disc_real.sum(),
                inputs=real_aug,
                create_graph=True
            )[0]

            r1_gamma = 10.0
            r1_penalty = (r1_gamma / 2) * grad_real.pow(2).view(grad_real.size(0), -1).sum(1).mean()
            loss_disc = loss_disc + r1_penalty


            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator

            # Path-Length Regularization
            if(batch_idx + 1) % g_every == 0:
                noise = torch.randn(bs, noise_dim, device=device)
                w = gen.mapping(noise)
                w.requires_grad_()

                fake = gen.synthesis(w)

                fake_for_g_aug = DiffAugment(fake, policy=aug_policy)

                disc_fake_for_g = disc(fake_for_g_aug)

                loss_gen = generator_hinge_loss(disc_fake_for_g)

                path_penalty, mean_path_length = g_path_regularization(fake, w, mean_path_length)

                loss_gen = loss_gen + path_penalty * path_reg_gamma

                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
                
                last_g_loss = loss_gen.item()



            if batch_idx % 100 == 0:
                elapsed = datetime.now() - start_time
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Elapsed: {elapsed} | "
                      f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} "
                      f"Loss D: {loss_disc.item():.4f}, Loss G: {last_g_loss:.4f}")

        # Save sample images
        with torch.no_grad():
            fake = gen(torch.randn(64, noise_dim, device=device)).detach().cpu()
            save_image(fake, os.path.join(output_dir, f"fake_epoch_{model_name}_{epoch}.png"), normalize=True, nrow=8)

        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'gen': gen.state_dict(),
                'disc': disc.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_disc': opt_disc.state_dict(),
                'mean_path_length': mean_path_length,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"gen_epoch_{model_name}_{epoch + 1}.pth"))
            torch.save(checkpoint, FULL_CHECKPOINT_PATH) # <-- Save to the "latest" file

    # Final generator save
    final_checkpoint = {
        'epoch': epochs,
        'gen': gen.state_dict(),
        'disc': disc.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'opt_disc': opt_disc.state_dict(),
        'mean_path_length': mean_path_length,
    }
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, f"{model_name}_gen.pth"))
    torch.save(final_checkpoint, FULL_CHECKPOINT_PATH)