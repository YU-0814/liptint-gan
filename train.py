"""Train pix2pix: GAN + 100*L1 + 10*perceptual.

    python train.py data/dataset --epochs 200
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from pix2pix import PatchDiscriminator, PerceptualLoss, UNetGenerator


class LipPairs(Dataset):
    def __init__(self, root, split):
        self.input_dir, self.target_dir = Path(root) / split / "input", Path(root) / split / "target"
        self.names = sorted(os.listdir(self.input_dir))
        self.to_input = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 4, (0.5,) * 4)])
        self.to_target = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        return (self.to_input(Image.open(self.input_dir / name).convert("RGBA")),
                self.to_target(Image.open(self.target_dir / name).convert("RGB")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-l1", type=float, default=100)
    p.add_argument("--lambda-perceptual", type=float, default=10)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl = DataLoader(LipPairs(a.dataset, "train"), batch_size=a.batch, shuffle=True, num_workers=4)
    val_dl = DataLoader(LipPairs(a.dataset, "val"), batch_size=a.batch, num_workers=4)
    G, D = UNetGenerator().to(device), PatchDiscriminator().to(device)
    gan_loss, l1_loss, perceptual = nn.MSELoss(), nn.L1Loss(), PerceptualLoss().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.5, 0.999))
    decay = lambda epoch: 1.0 - max(0, epoch - a.epochs // 2) / (a.epochs // 2)  # linear decay in the second half
    sched_g, sched_d = (torch.optim.lr_scheduler.LambdaLR(o, decay) for o in (opt_g, opt_d))

    log = {k: [] for k in ("G_GAN", "D", "L1", "perceptual", "val_L1", "val_perceptual")}
    for d in ("checkpoints", "results", "figures"):
        os.makedirs(d, exist_ok=True)

    for epoch in range(1, a.epochs + 1):
        G.train()
        D.train()
        sums = dict.fromkeys(("G_GAN", "D", "L1", "perceptual"), 0.0)
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            real = torch.ones(x.size(0), 1, 30, 30, device=device)
            fake = torch.zeros_like(real)

            opt_g.zero_grad()
            y_hat = G(x)
            g_gan = gan_loss(D(x, y_hat), real)
            g_l1 = l1_loss(y_hat, y) * a.lambda_l1
            g_per = perceptual(y_hat, y) * a.lambda_perceptual
            (g_gan + g_l1 + g_per).backward()
            opt_g.step()

            opt_d.zero_grad()
            d_loss = 0.5 * (gan_loss(D(x, y), real) + gan_loss(D(x, y_hat.detach()), fake))
            d_loss.backward()
            opt_d.step()
            for k, v in zip(sums, (g_gan, d_loss, g_l1, g_per), strict=True):
                sums[k] += v.item()
        sched_g.step()
        sched_d.step()
        for k in sums:
            log[k].append(sums[k] / len(train_dl))

        G.eval()
        val_l1 = val_per = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                y_hat = G(x)
                val_l1 += l1_loss(y_hat, y).item() * a.lambda_l1
                val_per += perceptual(y_hat, y).item() * a.lambda_perceptual
        log["val_L1"].append(val_l1 / len(val_dl))
        log["val_perceptual"].append(val_per / len(val_dl))
        print(f"epoch {epoch:3d}  " + "  ".join(f"{k} {v[-1]:.3f}" for k, v in log.items()), flush=True)

        if epoch % 10 == 0:
            x, y = next(iter(val_dl))
            with torch.no_grad():
                save_image(G(x.to(device)) * 0.5 + 0.5, f"results/epoch{epoch:03d}_fake.png")
            save_image(y * 0.5 + 0.5, f"results/epoch{epoch:03d}_real.png")
        if epoch % 50 == 0:
            torch.save(G.state_dict(), f"checkpoints/G_epoch{epoch}.pth")
            torch.save(D.state_dict(), f"checkpoints/D_epoch{epoch}.pth")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in ("L1", "perceptual"):
        ax1.plot(log[k], label=f"train {k}")
        ax1.plot(log[f"val_{k}"], "--", label=f"val {k}")
    ax1.set_title("Reconstruction terms (weighted)")
    for k in ("G_GAN", "D"):
        ax2.plot(log[k], label=k)
    ax2.set_title("Adversarial terms")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig("figures/loss_curves.png")


if __name__ == "__main__":
    main()
