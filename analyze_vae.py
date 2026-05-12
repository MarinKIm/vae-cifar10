"""
VAE CIFAR-10 Performance Analysis
- Loss curve (train / val across checkpoints)
- Per-sample reconstruction loss & KL divergence on full test set
- SSIM / PSNR per image
- Latent space 2-D PCA / t-SNE (coloured by class)
- Class-conditional reconstruction grid
"""

import os, glob, ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── paths ─────────────────────────────────────────────────────────────────
OUT_DIR  = "outputs"
ANA_DIR  = os.path.join(OUT_DIR, "analysis")
DATA_DIR = "data"
os.makedirs(ANA_DIR, exist_ok=True)

LATENT_DIM = 128
BATCH_SIZE = 256
CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
print(f"Device: {device}")

# ── model (copy from training script) ─────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, ld):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  32, 4,2,1), nn.ReLU(),
            nn.Conv2d(32, 64, 4,2,1), nn.ReLU(),
            nn.Conv2d(64,128, 4,2,1), nn.ReLU(),
            nn.Conv2d(128,256,4,2,1), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(256*2*2, ld)
        self.fc_var = nn.Linear(256*2*2, ld)
    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_var(h)

class Decoder(nn.Module):
    def __init__(self, ld):
        super().__init__()
        self.fc = nn.Linear(ld, 256*2*2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d( 64, 32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d( 32,  3,4,2,1), nn.Sigmoid(),
        )
    def forward(self, z):
        return self.net(self.fc(z).view(-1,256,2,2))

class VAE(nn.Module):
    def __init__(self, ld):
        super().__init__()
        self.encoder = Encoder(ld)
        self.decoder = Decoder(ld)
    def reparameterize(self, mu, lv):
        return mu + torch.randn_like(mu) * (0.5*lv).exp()
    def forward(self, x):
        mu, lv = self.encoder(x)
        return self.decoder(self.reparameterize(mu, lv)), mu, lv

# ── load final model ───────────────────────────────────────────────────────
model = VAE(LATENT_DIM).to(device)
ck = torch.load(f"{OUT_DIR}/checkpoint_epoch100.pt", map_location=device)
model.load_state_dict(ck["model_state"])
model.eval()
print("Loaded checkpoint epoch 100")

# ── 1. Loss curve from all checkpoints ────────────────────────────────────
print("\n[1/5] Loss curve …")
epochs_ck, train_losses, val_losses = [], [], []
for pt in sorted(glob.glob(f"{OUT_DIR}/checkpoint_epoch*.pt")):
    c = torch.load(pt, map_location="cpu")
    epochs_ck.append(c["epoch"])
    train_losses.append(c["train_loss"])
    val_losses.append(c["val_loss"])

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(epochs_ck, train_losses, "o-", label="train")
ax.plot(epochs_ck, val_losses,   "s-", label="val")
ax.set_xlabel("Epoch"); ax.set_ylabel("ELBO loss (per sample)")
ax.set_title("VAE CIFAR-10 — Training Curve")
ax.legend(); ax.grid(True, alpha=.4)
fig.tight_layout()
fig.savefig(f"{ANA_DIR}/loss_curve.png", dpi=150)
plt.close(fig)
print(f"  train epoch1={train_losses[0]:.1f}  epoch100={train_losses[-1]:.1f}  Δ={train_losses[0]-train_losses[-1]:.1f}")
print(f"  val   epoch1={val_losses[0]:.1f}  epoch100={val_losses[-1]:.1f}")

# ── dataset ────────────────────────────────────────────────────────────────
tf = transforms.ToTensor()
test_ds  = datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=tf)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 2. Per-sample recon loss & KL on test set ──────────────────────────────
print("\n[2/5] Per-sample recon / KL metrics …")
recon_losses, kl_losses, all_mu, all_lv, all_labels = [], [], [], [], []

with torch.no_grad():
    for x, y in test_dl:
        x = x.to(device)
        mu, lv = model.encoder(x)
        z  = model.reparameterize(mu, lv)
        rx = model.decoder(z)

        # per-image BCE (sum over pixels, mean over batch)
        rl = F.binary_cross_entropy(rx, x, reduction="none").sum(dim=[1,2,3])
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(dim=1)

        recon_losses.append(rl.cpu())
        kl_losses.append(kl.cpu())
        all_mu.append(mu.cpu())
        all_lv.append(lv.cpu())
        all_labels.append(y)

recon_losses = torch.cat(recon_losses).numpy()
kl_losses    = torch.cat(kl_losses).numpy()
all_mu       = torch.cat(all_mu).numpy()
all_labels   = torch.cat(all_labels).numpy()

print(f"  Recon loss — mean:{recon_losses.mean():.1f}  std:{recon_losses.std():.1f}  "
      f"min:{recon_losses.min():.1f}  max:{recon_losses.max():.1f}")
print(f"  KL loss   — mean:{kl_losses.mean():.1f}  std:{kl_losses.std():.1f}  "
      f"min:{kl_losses.min():.1f}  max:{kl_losses.max():.1f}")
print(f"  ELBO      — mean:{(recon_losses+kl_losses).mean():.1f}")

# per-class stats
print("\n  Per-class mean ELBO:")
for i, name in enumerate(CLASSES):
    idx = all_labels == i
    elbo = (recon_losses[idx] + kl_losses[idx]).mean()
    print(f"    {name:12s}: {elbo:.1f}")

# ── 3. PSNR on test set ────────────────────────────────────────────────────
print("\n[3/5] PSNR …")
psnr_all = []
with torch.no_grad():
    for x, _ in test_dl:
        x = x.to(device)
        rx, _, _ = model(x)
        mse = F.mse_loss(rx, x, reduction="none").mean(dim=[1,2,3])
        psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-8))
        psnr_all.append(psnr.cpu())
psnr_all = torch.cat(psnr_all).numpy()
print(f"  PSNR — mean:{psnr_all.mean():.2f} dB  std:{psnr_all.std():.2f}  "
      f"min:{psnr_all.min():.2f}  max:{psnr_all.max():.2f}")

# histogram
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(psnr_all, bins=60, color="steelblue", edgecolor="white", linewidth=.4)
ax.axvline(psnr_all.mean(), color="red", linestyle="--", label=f"mean {psnr_all.mean():.2f} dB")
ax.set_xlabel("PSNR (dB)"); ax.set_ylabel("Count")
ax.set_title("Reconstruction PSNR — Test Set")
ax.legend(); ax.grid(True, alpha=.4)
fig.tight_layout()
fig.savefig(f"{ANA_DIR}/psnr_histogram.png", dpi=150)
plt.close(fig)

# ── 4. Latent space PCA (fast) + t-SNE ────────────────────────────────────
print("\n[4/5] Latent space PCA + t-SNE …")
# use 2000 samples for speed
idx2k = np.random.choice(len(all_mu), 2000, replace=False)
mu2k  = all_mu[idx2k]
lb2k  = all_labels[idx2k]

# PCA
pca   = PCA(n_components=2).fit_transform(mu2k)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc = axes[0].scatter(pca[:,0], pca[:,1], c=lb2k, cmap="tab10", s=8, alpha=.7)
axes[0].set_title("Latent PCA (2D)"); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
plt.colorbar(sc, ax=axes[0], ticks=range(10), label="class")

# t-SNE
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000).fit_transform(mu2k)
sc2  = axes[1].scatter(tsne[:,0], tsne[:,1], c=lb2k, cmap="tab10", s=8, alpha=.7)
axes[1].set_title("Latent t-SNE (2D)"); axes[1].set_xlabel("dim1"); axes[1].set_ylabel("dim2")
cbar = plt.colorbar(sc2, ax=axes[1], ticks=range(10))
cbar.set_ticklabels(CLASSES)

fig.suptitle("VAE Latent Space — CIFAR-10 Test Set (n=2000)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{ANA_DIR}/latent_space.png", dpi=150)
plt.close(fig)
print("  saved latent_space.png")

# ── 5. Class-conditional reconstruction grid ───────────────────────────────
print("\n[5/5] Class reconstruction grid …")
# pick 8 images per class from test set
imgs_by_class = {i: [] for i in range(10)}
for x, y in DataLoader(test_ds, batch_size=1, shuffle=True):
    c = y.item()
    if len(imgs_by_class[c]) < 8:
        imgs_by_class[c].append(x.squeeze(0))
    if all(len(v) == 8 for v in imgs_by_class.values()):
        break

rows_orig, rows_recon = [], []
with torch.no_grad():
    for c in range(10):
        batch = torch.stack(imgs_by_class[c]).to(device)
        recon, _, _ = model(batch)
        rows_orig.append(batch.cpu())
        rows_recon.append(recon.cpu())

# interleave orig/recon rows
grid_rows = []
for c in range(10):
    grid_rows.append(rows_orig[c])
    grid_rows.append(rows_recon[c])

grid_tensor = torch.cat(grid_rows, dim=0)  # 160 x 3 x 32 x 32
grid_img = make_grid(grid_tensor, nrow=8, padding=2, normalize=False)
save_image(grid_img, f"{ANA_DIR}/class_recon_grid.png")

# annotated version with class labels
fig, ax = plt.subplots(figsize=(14, 22))
npimg = grid_img.permute(1,2,0).numpy()
ax.imshow(npimg)
ax.axis("off")
img_h = 34  # 32px + 2px padding
for c in range(10):
    row_orig  = c * 2
    row_recon = c * 2 + 1
    y_orig  = row_orig  * img_h + img_h // 2
    y_recon = row_recon * img_h + img_h // 2
    ax.text(-2, y_orig,  f"{CLASSES[c]}\norig",  ha="right", va="center", fontsize=7, color="navy")
    ax.text(-2, y_recon, f"{CLASSES[c]}\nrecon", ha="right", va="center", fontsize=7, color="darkred")
fig.tight_layout(pad=0)
fig.savefig(f"{ANA_DIR}/class_recon_grid_annotated.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved class_recon_grid.png + annotated version")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("PERFORMANCE SUMMARY")
print("="*55)
print(f"  Final ELBO (test)    : {(recon_losses+kl_losses).mean():.2f}")
print(f"  Recon loss (test)    : {recon_losses.mean():.2f}")
print(f"  KL divergence (test) : {kl_losses.mean():.2f}")
print(f"  PSNR (test)          : {psnr_all.mean():.2f} dB")
print(f"  Train Δloss (1→100)  : {train_losses[0]-train_losses[-1]:.1f}")
print("="*55)
print(f"\nAll analysis files saved to: {ANA_DIR}/")
print("  loss_curve.png")
print("  psnr_histogram.png")
print("  latent_space.png")
print("  class_recon_grid.png")
print("  class_recon_grid_annotated.png")
