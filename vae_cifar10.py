import os
import ssl
import certifi

# fix macOS Python SSL cert issue
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# ── Config ──────────────────────────────────────────────────────────────────
LATENT_DIM = 128
BATCH_SIZE  = 256
EPOCHS      = 100
LR          = 1e-3
SAVE_EVERY  = 10          # save checkpoint every N epochs
OUT_DIR     = "outputs"
DATA_DIR    = "data"

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Device: {device}")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# ── Model ────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  32, 4, 2, 1), nn.ReLU(),   # 16x16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),   #  8x8
            nn.Conv2d(64,128, 4, 2, 1), nn.ReLU(),   #  4x4
            nn.Conv2d(128,256,4, 2, 1), nn.ReLU(),   #  2x2
        )
        self.fc_mu  = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_var = nn.Linear(256 * 2 * 2, latent_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),   #  4x4
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(),   #  8x8
            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ReLU(),   # 16x16
            nn.ConvTranspose2d( 32,   3, 4, 2, 1), nn.Sigmoid(),# 32x32
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 2, 2)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var


def vae_loss(recon, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss

# ── Training ─────────────────────────────────────────────────────────────────
model     = VAE(LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# save fixed test batch for consistent reconstruction grids
fixed_x, _ = next(iter(test_loader))
fixed_x = fixed_x[:64].to(device)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for x, _ in train_loader:
        x = x.to(device)
        recon, mu, log_var = model(x)
        loss = vae_loss(recon, x, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_dataset)

    # ── Validation ────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            recon, mu, log_var = model(x)
            val_loss += vae_loss(recon, x, mu, log_var).item()
    avg_val = val_loss / len(test_dataset)

    print(f"Epoch [{epoch:3d}/{EPOCHS}]  train={avg_loss:.4f}  val={avg_val:.4f}")

    # ── Save outputs every SAVE_EVERY epochs and at final epoch ───────────
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        with torch.no_grad():
            recon_fixed, _, _ = model(fixed_x)
        # interleave original / reconstruction
        comparison = torch.cat([fixed_x[:8], recon_fixed[:8]])
        save_image(comparison.cpu(), f"{OUT_DIR}/recon_epoch{epoch:03d}.png", nrow=8)

        # random samples from prior
        z = torch.randn(64, LATENT_DIM).to(device)
        samples = model.decoder(z)
        save_image(samples.cpu(), f"{OUT_DIR}/sample_epoch{epoch:03d}.png", nrow=8)

        # checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_loss": avg_val,
        }, f"{OUT_DIR}/checkpoint_epoch{epoch:03d}.pt")
        print(f"  → saved recon / sample images and checkpoint")

print("Training complete.")
