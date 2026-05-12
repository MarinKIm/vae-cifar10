"""
Build: Bayesian Probability → VAE CIFAR-10 Report PDF
Renders key textbook pages as images and composes a polished academic report.
"""

import os, ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import fitz                          # PyMuPDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table,
    TableStyle, HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String
import io

# ── Paths ─────────────────────────────────────────────────────────────────
SRC_PDF  = "/Users/marinkim/Downloads/Prob___Stat (1300).pdf"
OUT_PDF  = "/Users/marinkim/HAI-Project/bayesian_vae_report.pdf"
TMP_DIR  = "/Users/marinkim/HAI-Project/outputs/report_tmp"
ANA_DIR  = "/Users/marinkim/HAI-Project/outputs/analysis"
os.makedirs(TMP_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#1a2744")
BLUE    = colors.HexColor("#2563eb")
LBLUE   = colors.HexColor("#dbeafe")
TEAL    = colors.HexColor("#0d9488")
LTEAL   = colors.HexColor("#ccfbf1")
GRAY    = colors.HexColor("#64748b")
LGRAY   = colors.HexColor("#f1f5f9")
WHITE   = colors.white
BLACK   = colors.HexColor("#0f172a")

W, H = A4   # 595 x 842 pt
MARGIN = 2.2 * cm

# ── Render textbook pages → PNG images ───────────────────────────────────
def render_page(src_doc, page_no, zoom=1.8, crop_pct=None):
    """Render a PDF page to PNG bytes. crop_pct=(top,bottom) fraction to crop."""
    page = src_doc[page_no - 1]
    mat  = fitz.Matrix(zoom, zoom)
    clip = None
    if crop_pct:
        r = page.rect
        top_y  = r.y0 + r.height * crop_pct[0]
        bot_y  = r.y0 + r.height * crop_pct[1]
        clip   = fitz.Rect(r.x0, top_y, r.x1, bot_y)
    pix  = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
    return pix.tobytes("png")

print("Rendering textbook pages …")
src = fitz.open(SRC_PDF)

captures = {
    "bayes_theorem":  render_page(src, 287, zoom=2.0, crop_pct=(0.0, 0.62)),
    "bayes_vs_freq":  render_page(src, 279, zoom=2.0, crop_pct=(0.0, 0.65)),
    "mle":            render_page(src, 290, zoom=2.0, crop_pct=(0.0, 0.72)),
    "mle_gaussian":   render_page(src, 296, zoom=2.0, crop_pct=(0.0, 0.72)),
    "map":            render_page(src, 301, zoom=2.0, crop_pct=(0.0, 0.68)),
    "map_vs_ml":      render_page(src, 314, zoom=2.0, crop_pct=(0.0, 0.80)),
    "kl_div":         render_page(src, 357, zoom=2.0, crop_pct=(0.0, 0.62)),
}

# Save to tmp files
img_paths = {}
for k, png in captures.items():
    p = f"{TMP_DIR}/{k}.png"
    with open(p, "wb") as f:
        f.write(png)
    img_paths[k] = p
    print(f"  saved {k}.png")

src.close()

# ── ReportLab styles ──────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "cover_title": S("cover_title",
            fontName="Helvetica-Bold", fontSize=28, textColor=WHITE,
            leading=36, alignment=TA_CENTER, spaceAfter=6),
        "cover_sub": S("cover_sub",
            fontName="Helvetica", fontSize=13, textColor=colors.HexColor("#bfdbfe"),
            leading=20, alignment=TA_CENTER, spaceAfter=4),
        "cover_author": S("cover_author",
            fontName="Helvetica-Bold", fontSize=11, textColor=WHITE,
            leading=16, alignment=TA_CENTER),
        "h1": S("h1",
            fontName="Helvetica-Bold", fontSize=18, textColor=NAVY,
            leading=24, spaceBefore=18, spaceAfter=8,
            borderPadding=(0,0,4,0)),
        "h2": S("h2",
            fontName="Helvetica-Bold", fontSize=13, textColor=BLUE,
            leading=18, spaceBefore=12, spaceAfter=6),
        "h3": S("h3",
            fontName="Helvetica-Bold", fontSize=11, textColor=TEAL,
            leading=16, spaceBefore=8, spaceAfter=4),
        "body": S("body",
            fontName="Helvetica", fontSize=10, textColor=BLACK,
            leading=15, spaceAfter=6, alignment=TA_JUSTIFY),
        "caption": S("caption",
            fontName="Helvetica-Oblique", fontSize=8.5, textColor=GRAY,
            leading=12, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10),
        "formula": S("formula",
            fontName="Helvetica-Bold", fontSize=10.5, textColor=NAVY,
            leading=16, alignment=TA_CENTER, spaceBefore=6, spaceAfter=6,
            backColor=LBLUE, borderPadding=8),
        "code": S("code",
            fontName="Courier", fontSize=8.5, textColor=colors.HexColor("#1e293b"),
            leading=13, backColor=LGRAY, borderPadding=6, spaceAfter=8),
        "highlight": S("highlight",
            fontName="Helvetica", fontSize=10, textColor=BLACK,
            leading=15, backColor=LTEAL, borderPadding=8,
            spaceAfter=8, alignment=TA_JUSTIFY),
        "toc_entry": S("toc_entry",
            fontName="Helvetica", fontSize=10.5, textColor=NAVY,
            leading=18),
    }

ST = make_styles()

def img_fw(path, width=None, height=None):
    """Full-width image flowable."""
    w = width or (W - 2*MARGIN)
    i = Image(path, width=w, height=height)
    i.hAlign = "CENTER"
    return i

def divider(color=BLUE, thickness=1):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=6, spaceBefore=6)

def section_header(text, style="h1"):
    return [divider(NAVY, 1.5), Paragraph(text, ST[style]), divider(BLUE, 0.5)]

def callout(text, style="highlight"):
    return Paragraph(text, ST[style])

def formula_box(text):
    return Paragraph(text, ST["formula"])

def two_col(left, right, left_w=0.55):
    """Two-column layout via Table."""
    cw = W - 2*MARGIN
    tbl = Table([[left, right]],
                colWidths=[cw*left_w, cw*(1-left_w)],
                hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
        ("INNERGRID", (0,0),(-1,-1), 0, colors.white),
    ]))
    return tbl

# ── Cover page ────────────────────────────────────────────────────────────
def cover_page(story):
    # dark background via a coloured table
    cover_data = [[Paragraph(
        "<br/><br/><br/><br/>",
        ST["cover_title"])]]
    tbl = Table(cover_data, colWidths=[W - 2*MARGIN])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), NAVY),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(Spacer(1, 1.5*cm))

    # Title block
    title_data = [
        [Paragraph("Bayesian Probability Theory", ST["cover_title"])],
        [Paragraph("and Variational Autoencoders", ST["cover_title"])],
        [Spacer(1, 0.3*cm)],
        [Paragraph("From Probabilistic Foundations to CIFAR-10 Generation", ST["cover_sub"])],
        [Spacer(1, 1.2*cm)],
        [Paragraph("MarinKIm · HAI Project · 2026", ST["cover_author"])],
        [Spacer(1, 2*cm)],
    ]
    bg = Table(title_data, colWidths=[W - 2*MARGIN])
    bg.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 20),
        ("RIGHTPADDING",  (0,0),(-1,-1), 20),
    ]))
    story.append(bg)
    story.append(Spacer(1, 0.8*cm))

    # Metrics bar
    metrics = [
        ["CIFAR-10", "latent_dim=128", "PSNR 19.44 dB", "100 Epochs"],
        ["Dataset", "Architecture", "Performance", "Training"],
    ]
    mt = Table(metrics, colWidths=[(W-2*MARGIN)/4]*4)
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), BLUE),
        ("BACKGROUND",    (0,1),(-1,1), LBLUE),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("TEXTCOLOR",     (0,1),(-1,1), NAVY),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",      (0,1),(-1,1), "Helvetica"),
        ("FONTSIZE",      (0,0),(-1,-1), 9.5),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
    ]))
    story.append(mt)
    story.append(PageBreak())

# ── Table of Contents ─────────────────────────────────────────────────────
def toc_page(story):
    story += section_header("Table of Contents", "h1")
    story.append(Spacer(1, 0.3*cm))
    toc = [
        ("1.", "Bayesian Framework: The Foundation of Probabilistic Generative Models"),
        ("2.", "Bayes' Theorem → VAE's Core Probabilistic Identity"),
        ("3.", "Prior Distribution p(z) → Latent Space Design"),
        ("4.", "Likelihood p(x|z) → Decoder and Reconstruction Loss"),
        ("5.", "Posterior Inference → Encoder and Variational Approximation"),
        ("6.", "KL Divergence → ELBO Loss Function"),
        ("7.", "MLE and MAP → Regularization through a Bayesian Lens"),
        ("8.", "Experimental Results on CIFAR-10"),
        ("9.", "Conclusion"),
    ]
    for num, title in toc:
        row = Table([[Paragraph(num, ST["body"]),
                      Paragraph(title, ST["body"])]],
                    colWidths=[1*cm, W-2*MARGIN-1*cm])
        row.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),0),
            ("RIGHTPADDING",(0,0),(-1,-1),0),
            ("TOPPADDING",(0,0),(-1,-1),2),
            ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ]))
        story.append(row)
    story.append(PageBreak())

# ── Section 1: Bayesian Framework ─────────────────────────────────────────
def section_bayesian_framework(story):
    story += section_header("1. Bayesian Framework: The Foundation of Probabilistic Generative Models", "h1")

    story.append(Paragraph(
        "Traditional deep learning models learn deterministic mappings from input to output. "
        "Probabilistic generative models, by contrast, learn the <b>underlying distribution</b> of data — "
        "enabling not just recognition, but generation. "
        "The Bayesian framework provides the mathematical language for this endeavour.",
        ST["body"]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Bayesian vs. Frequentist Probability", ST["h2"]))
    story.append(Paragraph(
        "The course (Prob &amp; Stat, Sungyoon Lee) distinguishes two interpretations of probability. "
        "The <b>Bayesian</b> view treats probability as a <i>degree of belief</i> in a hypothesis — "
        "a belief updated as evidence arrives. "
        "The <b>Frequentist</b> view treats probability as the long-run frequency of an event. "
        "Generative models are inherently Bayesian: we maintain a belief over latent variables "
        "and update that belief given observed data.",
        ST["body"]))

    story.append(img_fw(img_paths["bayes_vs_freq"], width=12*cm))
    story.append(Paragraph(
        "Figure 1. Bayesian vs. Frequentist interpretation (Prob &amp; Stat, p.178). "
        "In VAE we adopt the Bayesian view — the latent code z is a random variable "
        "over which we maintain a prior belief p(z).",
        ST["caption"]))

    story.append(callout(
        "Key insight: A VAE is a Bayesian model. The encoder learns a posterior belief "
        "over latent variables; the decoder samples from that belief to reconstruct data. "
        "Every component of the VAE loss has a direct Bayesian interpretation."))

    story.append(PageBreak())

# ── Section 2: Bayes' Theorem ─────────────────────────────────────────────
def section_bayes_theorem(story):
    story += section_header("2. Bayes' Theorem → VAE's Core Probabilistic Identity", "h1")

    story.append(Paragraph(
        "Bayes' Theorem is the central equation of probabilistic inference. "
        "It tells us how to update a prior belief P(H) after observing evidence E:",
        ST["body"]))

    story.append(img_fw(img_paths["bayes_theorem"], width=13*cm))
    story.append(Paragraph(
        "Figure 2. Bayes' Theorem as stated in the course (Prob &amp; Stat, p.180). "
        "The four quantities — likelihood, prior, posterior, and evidence — "
        "map directly onto VAE components.",
        ST["caption"]))

    story.append(Paragraph("Mapping to VAE", ST["h2"]))
    story.append(Paragraph(
        "In the context of a Variational Autoencoder, we wish to model the distribution "
        "of high-dimensional data x (CIFAR-10 images) via a lower-dimensional latent variable z. "
        "Applying Bayes' Theorem:",
        ST["body"]))

    formula_rows = [
        ["Bayesian term", "VAE interpretation", "In our code"],
        ["H (hypothesis)", "Latent code z ∈ R¹²⁸", "128-dim encoder output"],
        ["E (evidence/data)", "Image x ∈ R³ˣ³²ˣ³²", "CIFAR-10 pixels"],
        ["P(H) — Prior", "p(z) = N(0, I)", "reparameterize: eps~N(0,I)"],
        ["P(E|H) — Likelihood", "p(x|z) via decoder", "Sigmoid output + BCE loss"],
        ["P(H|E) — Posterior", "p(z|x) — intractable!", "Approximated by q_phi(z|x)"],
        ["P(E) — Evidence", "p(x) = integral p(x|z)p(z)dz", "Intractable; use ELBO"],
    ]
    ft = Table(formula_rows,
               colWidths=[(W-2*MARGIN)*f for f in [0.26, 0.38, 0.36]])
    ft.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGRAY]),
        ("ALIGN",         (0,0),(-1,-1), "LEFT"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
    ]))
    story.append(ft)
    story.append(Spacer(1, 0.3*cm))

    story.append(formula_box(
        "p(z | x)  =  p(x | z) · p(z)  /  p(x)"
        "        →        q_φ(z|x) ≈ p(z|x)"))

    story.append(Paragraph(
        "The true posterior p(z|x) is <b>intractable</b> because p(x) requires integrating "
        "over all possible latent codes — an integral with no closed form for neural networks. "
        "This is precisely why VAE uses <b>variational inference</b>: we introduce an "
        "approximate posterior q_φ(z|x), parameterised by the encoder network, and optimise "
        "it to be as close to the true posterior as possible.",
        ST["body"]))

    story.append(PageBreak())

# ── Section 3: Prior ──────────────────────────────────────────────────────
def section_prior(story):
    story += section_header("3. Prior Distribution p(z) → Latent Space Design", "h1")

    story.append(Paragraph(
        "In Bayesian inference, the <b>prior P(H)</b> encodes our belief about the hypothesis "
        "<i>before</i> observing any data. In a VAE, the prior is placed over the latent variable z "
        "and shapes the structure of the learned latent space.",
        ST["body"]))

    story.append(Paragraph("Course Foundation", ST["h2"]))
    story.append(Paragraph(
        "The course introduces the multivariate Gaussian N(y; μ, Σ) and shows how uncorrelated "
        "Gaussian components are also independent. This property is critical: by choosing "
        "p(z) = N(0, I) as our prior, we encourage the encoder to produce <i>disentangled</i> "
        "latent dimensions with no cross-correlations.",
        ST["body"]))

    story.append(Paragraph("Implementation in vae_cifar10.py", ST["h2"]))
    story.append(Paragraph(
        "Our encoder outputs two vectors — μ (mu) and log σ² (log_var) — one per latent dimension. "
        "During training, the KL term of the ELBO loss pushes the approximate posterior "
        "q_φ(z|x) = N(μ_φ(x), diag(σ²_φ(x))) toward the standard Gaussian prior N(0, I):",
        ST["body"]))

    story.append(Paragraph(
        "class Encoder(nn.Module):<br/>"
        "    self.fc_mu  = nn.Linear(256*2*2, 128)   # mean vector<br/>"
        "    self.fc_var = nn.Linear(256*2*2, 128)   # log-variance vector<br/>"
        "<br/>"
        "def reparameterize(self, mu, log_var):<br/>"
        "    std = torch.exp(0.5 * log_var)<br/>"
        "    eps = torch.randn_like(std)   # eps ~ N(0, I) — prior samples<br/>"
        "    return mu + eps * std",
        ST["code"]))

    story.append(callout(
        "The choice p(z) = N(0, I) as prior is not arbitrary. It is the simplest distribution "
        "that (1) has full support over R¹²⁸, (2) allows closed-form KL computation against "
        "a diagonal Gaussian posterior, and (3) makes the latent space easy to sample from "
        "at inference time — draw eps ~ N(0,I) and pass through the decoder."))

    story.append(Paragraph("Effect on Latent Space", ST["h2"]))
    story.append(Paragraph(
        "Our PCA and t-SNE analysis of 2,000 test-set latent codes reveals that the prior "
        "regularisation successfully organises the latent space: semantically similar classes "
        "(automobile/truck, deer/horse) cluster together, while visually distinct classes "
        "(airplane, frog) form well-separated clusters.",
        ST["body"]))

    story.append(img_fw(f"{ANA_DIR}/latent_space.png", width=13*cm, height=5*cm))
    story.append(Paragraph(
        "Figure 3. PCA (left) and t-SNE (right) of the 128-dim latent space for 2,000 test images. "
        "The Gaussian prior encourages smooth, continuous structure rather than a collapsed or fragmented space.",
        ST["caption"]))

    story.append(PageBreak())

# ── Section 4: Likelihood ─────────────────────────────────────────────────
def section_likelihood(story):
    story += section_header("4. Likelihood p(x|z) → Decoder and Reconstruction Loss", "h1")

    story.append(Paragraph(
        "The <b>likelihood P(E|H)</b> in Bayes' Theorem quantifies how probable the observed data is "
        "under a given hypothesis. In a VAE the likelihood is p(x|z): given a latent code z, "
        "how probable is the image x? Maximising this likelihood is exactly what the reconstruction "
        "loss does.",
        ST["body"]))

    story.append(Paragraph("Maximum Likelihood Estimation — Course Derivation", ST["h2"]))
    story.append(Paragraph(
        "The course derives Maximum Likelihood Estimation (MLE) for both Bernoulli and Gaussian models. "
        "For a Bernoulli distribution, the log-likelihood is:",
        ST["body"]))

    story.append(img_fw(img_paths["mle"], width=13*cm))
    story.append(Paragraph(
        "Figure 4. MLE derivation for Bernoulli data (Prob &amp; Stat, p.182). "
        "Each pixel in a CIFAR-10 image is modelled as independent Bernoulli — "
        "leading directly to Binary Cross-Entropy as the reconstruction loss.",
        ST["caption"]))

    story.append(Paragraph("Connection to Binary Cross-Entropy", ST["h2"]))
    story.append(Paragraph(
        "We model each pixel channel value x_i ∈ [0,1] as drawn from a Bernoulli distribution "
        "parameterised by the decoder output r_i = σ(decoder(z)_i). "
        "The log-likelihood is:",
        ST["body"]))

    story.append(formula_box(
        "log p(x|z) = Σ_i [ x_i log r_i + (1 - x_i) log(1 - r_i) ]  =  -BCE(r, x)"))

    story.append(Paragraph(
        "This is exactly the binary cross-entropy loss used in our training loop. "
        "Maximising log p(x|z) is identical to minimising BCE. "
        "The decoder's final Sigmoid activation ensures outputs lie in [0,1], "
        "consistent with Bernoulli pixel probabilities.",
        ST["body"]))

    story.append(Paragraph(
        "def vae_loss(recon, x, mu, log_var):<br/>"
        "    # Reconstruction: -E[log p(x|z)] = BCE summed over pixels<br/>"
        "    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')<br/>"
        "    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())<br/>"
        "    return recon_loss + kl_loss",
        ST["code"]))

    story.append(Paragraph("Gaussian Likelihood (course p.184)", ST["h2"]))
    story.append(img_fw(img_paths["mle_gaussian"], width=13*cm))
    story.append(Paragraph(
        "Figure 5. MLE under Gaussian noise assumption (Prob &amp; Stat, p.184). "
        "This shows that minimising MSE is equivalent to maximising Gaussian log-likelihood — "
        "motivating the use of BCE (Bernoulli) rather than MSE (Gaussian) for image pixel values in [0,1].",
        ST["caption"]))

    story.append(PageBreak())

# ── Section 5: Posterior Inference ────────────────────────────────────────
def section_posterior(story):
    story += section_header("5. Posterior Inference → Encoder and Variational Approximation", "h1")

    story.append(Paragraph(
        "The posterior P(H|E) is our updated belief in the hypothesis after observing data. "
        "In a VAE, computing the true posterior p(z|x) is intractable. "
        "The VAE solves this through <b>variational inference</b>: "
        "introduce a tractable family of distributions q_φ(z|x) and minimise the "
        "KL divergence from it to the true posterior.",
        ST["body"]))

    story.append(Paragraph("From MAP to Variational Inference", ST["h2"]))
    story.append(img_fw(img_paths["map"], width=13*cm))
    story.append(Paragraph(
        "Figure 6. MAP estimation (Prob &amp; Stat, p.187). "
        "MAP finds the single most probable hypothesis — a point estimate. "
        "Variational inference goes further: it finds an entire distribution q_φ(z|x) "
        "that approximates the posterior, capturing uncertainty rather than collapsing to a point.",
        ST["caption"]))

    story.append(Paragraph("The Encoder as a Variational Posterior", ST["h2"]))
    story.append(Paragraph(
        "Our encoder network implements q_φ(z|x) = N(z; μ_φ(x), diag(σ²_φ(x))). "
        "For every input image x, the encoder outputs a mean vector and a log-variance vector "
        "that parameterise a Gaussian distribution over z. "
        "Rather than encoding a single point, the encoder encodes an <i>entire distribution</i> "
        "— the hallmark of Bayesian inference.",
        ST["body"]))

    story.append(formula_box(
        "q_φ(z|x) = N(z; μ_φ(x), diag(exp(log_var_φ(x))))"))

    story.append(Paragraph("The Reparameterisation Trick", ST["h2"]))
    story.append(Paragraph(
        "To backpropagate through sampling from q_φ(z|x), we use the reparameterisation trick. "
        "Instead of sampling z ~ N(μ, σ²) directly (non-differentiable), we write:",
        ST["body"]))

    story.append(formula_box("z = μ_φ(x)  +  σ_φ(x) · ε,     ε ~ N(0, I)"))

    story.append(Paragraph(
        "Now ε is drawn from the fixed prior N(0, I), and the only stochastic node "
        "is outside the parameter path. Gradients flow through μ_φ and σ_φ unobstructed. "
        "This trick is what makes end-to-end training of VAE possible.",
        ST["body"]))

    story.append(Paragraph(
        "def reparameterize(self, mu, log_var):<br/>"
        "    std = torch.exp(0.5 * log_var)   # sigma<br/>"
        "    eps = torch.randn_like(std)       # eps ~ N(0, I)  — the prior<br/>"
        "    return mu + eps * std             # z = mu + sigma * eps",
        ST["code"]))

    story.append(PageBreak())

# ── Section 6: KL Divergence ──────────────────────────────────────────────
def section_kl(story):
    story += section_header("6. KL Divergence → ELBO Loss Function", "h1")

    story.append(Paragraph(
        "The course connects KL divergence to MLE and NLL (Negative Log-Likelihood), "
        "showing that minimising KL(p_data || q_model) is equivalent to maximum likelihood. "
        "In VAE, KL divergence appears in two places: "
        "(1) the ELBO objective and (2) the regularisation term in the loss.",
        ST["body"]))

    story.append(img_fw(img_paths["kl_div"], width=13*cm))
    story.append(Paragraph(
        "Figure 7. KL divergence and its connection to NLL (Prob &amp; Stat, p.231). "
        "Minimising KL(p_S || q) is equivalent to maximising log-likelihood under q, "
        "linking information-theoretic and probabilistic views of learning.",
        ST["caption"]))

    story.append(Paragraph("Evidence Lower Bound (ELBO)", ST["h2"]))
    story.append(Paragraph(
        "Since p(x) is intractable, we cannot directly maximise log p(x). "
        "Instead, we derive the Evidence Lower BOund (ELBO) using Jensen's inequality "
        "and the KL divergence between the approximate and true posterior:",
        ST["body"]))

    story.append(formula_box(
        "log p(x)  ≥  E_q[log p(x|z)]  -  KL( q_φ(z|x) || p(z) )  =: ELBO"))

    story.append(Paragraph(
        "Maximising the ELBO is equivalent to simultaneously: "
        "(1) maximising the expected reconstruction quality E_q[log p(x|z)], and "
        "(2) minimising KL(q_φ(z|x) || p(z)), the divergence between the approximate "
        "posterior and the prior.",
        ST["body"]))

    story.append(Paragraph("Closed-form KL for Diagonal Gaussians", ST["h2"]))
    story.append(Paragraph(
        "Because both q_φ(z|x) = N(μ, diag(σ²)) and p(z) = N(0, I) are Gaussian, "
        "the KL divergence has a closed-form expression — no Monte Carlo estimation needed:",
        ST["body"]))

    story.append(formula_box(
        "KL( N(μ, σ²) || N(0,I) )  =  -½ Σ_j [ 1 + log σ²_j - μ²_j - σ²_j ]"))

    story.append(Paragraph(
        "def vae_loss(recon, x, mu, log_var):<br/>"
        "    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')<br/>"
        "    # Closed-form KL: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))<br/>"
        "    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())<br/>"
        "    return recon_loss + kl_loss   # minimise -ELBO",
        ST["code"]))

    story.append(callout(
        "Our experiment: KL divergence converged to 43.01 on the test set — "
        "well above zero, indicating the encoder is not collapsing to the prior "
        "(posterior collapse), and the 128 latent dimensions are actively used."))

    story.append(PageBreak())

# ── Section 7: MLE vs MAP ─────────────────────────────────────────────────
def section_mle_map(story):
    story += section_header("7. MLE and MAP → Regularisation through a Bayesian Lens", "h1")

    story.append(Paragraph(
        "The course provides a detailed comparison of Maximum Likelihood Estimation (MLE) "
        "and Maximum a Posteriori (MAP) estimation, showing that MAP is equivalent to "
        "MLE with a regularisation term imposed by the prior. "
        "The VAE loss directly instantiates this relationship.",
        ST["body"]))

    story.append(img_fw(img_paths["map_vs_ml"], width=14*cm))
    story.append(Paragraph(
        "Figure 8. MLE vs. MAP comparison table (Prob &amp; Stat, p.194). "
        "MAP introduces prior knowledge that acts as a regulariser — "
        "exactly the role of the KL term in the VAE ELBO.",
        ST["caption"]))

    story.append(Paragraph("VAE as Regularised MLE", ST["h2"]))
    story.append(Paragraph(
        "The reconstruction loss term (-E[log p(x|z)]) corresponds to <b>MLE</b> of the decoder. "
        "The KL term (-KL(q||p)) corresponds to the <b>log prior</b> in MAP, "
        "penalising posteriors that deviate from N(0, I). "
        "Together, the ELBO is formally equivalent to MAP estimation of the encoder parameters "
        "under a Gaussian prior:",
        ST["body"]))

    story.append(formula_box(
        "ELBO = E_q[log p(x|z)]  -  KL(q_φ||p)  ≡  log_likelihood  +  log_prior"))

    story.append(Paragraph("Overfitting and the Prior's Role", ST["h2"]))
    story.append(Paragraph(
        "The course demonstrates (p.190) that with n=3 coin tosses all heads, "
        "MLE gives ω_ML=1 — a severe overfit. "
        "MAP with a sensible prior gives ω_MAP=4/5. "
        "Analogously, without the KL regulariser, the encoder could collapse "
        "q_φ(z|x) to a delta function (zero variance), perfectly memorising training images "
        "but producing a degenerate latent space with no generative capability. "
        "The KL term prevents this collapse, acting as the Bayesian prior.",
        ST["body"]))

    story.append(callout(
        "In our CIFAR-10 VAE: removing the KL term would set σ²→0 for all images "
        "(lossless but non-generative compression). "
        "The KL weight enforces σ²>0 so that sampling ε~N(0,I) produces "
        "meaningful new images — the essence of a generative model."))

    story.append(PageBreak())

# ── Section 8: Results ────────────────────────────────────────────────────
def section_results(story):
    story += section_header("8. Experimental Results on CIFAR-10", "h1")

    story.append(Paragraph(
        "We trained the convolutional VAE for 100 epochs on CIFAR-10 "
        "(60,000 training / 10,000 test images, 32×32 RGB) using Apple M5 MPS. "
        "All Bayesian quantities — prior, likelihood, and approximate posterior — "
        "are directly measured and tracked.",
        ST["body"]))

    story.append(Paragraph("Quantitative Metrics", ST["h2"]))
    results = [
        ["Metric", "Value", "Bayesian Interpretation"],
        ["Final ELBO (test)", "−1820.67", "Evidence lower bound on log p(x)"],
        ["Recon loss BCE (test)", "1777.65", "−E_q[log p(x|z)]: reconstruction quality"],
        ["KL divergence (test)", "43.01", "KL(q_φ(z|x) || p(z)): posterior-prior gap"],
        ["PSNR (test)", "19.44 dB", "Signal-to-noise of decoded images"],
        ["Train Δloss (ep 1→100)", "−20.7", "ELBO improvement over training"],
    ]
    rt = Table(results, colWidths=[(W-2*MARGIN)*f for f in [0.30, 0.20, 0.50]])
    rt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), TEAL),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGRAY]),
        ("ALIGN",         (0,0),(0,-1), "LEFT"),
        ("ALIGN",         (1,0),(1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
    ]))
    story.append(rt)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Training Curve (ELBO)", ST["h2"]))
    story.append(img_fw(f"{ANA_DIR}/loss_curve.png", width=13*cm, height=6.5*cm))
    story.append(Paragraph(
        "Figure 9. Train/val ELBO over 100 epochs. "
        "Rapid initial improvement reflects learning the data distribution; "
        "the flat tail suggests the model has saturated the expressive capacity of the 128-dim latent space.",
        ST["caption"]))

    story.append(Paragraph("Reconstruction Quality and PSNR Distribution", ST["h2"]))
    story.append(img_fw(f"{ANA_DIR}/psnr_histogram.png", width=12*cm, height=6.8*cm))
    story.append(Paragraph(
        "Figure 10. PSNR distribution across the 10,000 test images. "
        "Mean PSNR of 19.44 dB is typical for VAEs on CIFAR-10, "
        "reflecting the inherent blurriness of Gaussian-approximated posteriors.",
        ST["caption"]))

    story.append(Paragraph("Per-Class ELBO (Bayesian Difficulty)", ST["h2"]))
    story.append(Paragraph(
        "The Bayesian lens reveals which classes are hardest to model: "
        "higher ELBO means the model assigns lower probability to test images. "
        "Structured objects (automobile, truck) are easiest; "
        "natural textures (deer, bird) are hardest — consistent with the "
        "prior N(0,I) struggling to represent complex multi-modal posteriors.",
        ST["body"]))

    class_data = [
        ["Class", "Mean ELBO", "Difficulty"],
        ["automobile", "1766", "★ easiest"],
        ["truck",      "1774", "★"],
        ["airplane",   "1773", "★"],
        ["ship",       "1820", ""],
        ["cat",        "1800", ""],
        ["horse",      "1837", "▲"],
        ["dog",        "1826", "▲"],
        ["frog",       "1845", "▲"],
        ["bird",       "1863", "▲▲"],
        ["deer",       "1903", "▲▲ hardest"],
    ]
    ct = Table(class_data, colWidths=[(W-2*MARGIN)*f for f in [0.30, 0.25, 0.45]])
    ct.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGRAY]),
        ("ALIGN",         (1,0),(1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
    ]))
    story.append(ct)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Class-wise Reconstruction Grid", ST["h2"]))
    # 1445x3330 → ratio 2.30; use 7cm wide → 16.1cm tall
    story.append(img_fw(f"{ANA_DIR}/class_recon_grid_annotated.png", width=7*cm, height=16*cm))
    story.append(Paragraph(
        "Figure 11. Original (odd rows) vs. reconstructed (even rows) for all 10 classes. "
        "Smooth outputs reflect the Gaussian posterior's averaging effect — "
        "a direct consequence of maximising E_q[log p(x|z)] under a unimodal q_φ.",
        ST["caption"]))

    story.append(PageBreak())

# ── Section 9: Conclusion ─────────────────────────────────────────────────
def section_conclusion(story):
    story += section_header("9. Conclusion", "h1")

    story.append(Paragraph(
        "This project demonstrates that the Variational Autoencoder is not merely "
        "a neural network architecture — it is a complete Bayesian probabilistic model "
        "where every design decision has a rigorous theoretical foundation "
        "in the probability and statistics framework covered in this course.",
        ST["body"]))

    summary = [
        ["Course Concept", "VAE Component", "Code Location"],
        ["Bayes' Theorem", "Core identity: p(z|x) ∝ p(x|z)p(z)", "vae_loss()"],
        ["Prior P(H)", "p(z) = N(0, I) — latent prior", "reparameterize()"],
        ["Likelihood P(E|H)", "p(x|z) — Bernoulli pixel model", "Decoder + BCE loss"],
        ["Posterior P(H|E)", "q_φ(z|x) = N(μ_φ, σ²_φ) — encoder", "Encoder.fc_mu / fc_var"],
        ["KL Divergence", "-KL(q||p) — regularisation term", "vae_loss() KL term"],
        ["MLE", "Reconstruction loss maximisation", "binary_cross_entropy"],
        ["MAP / Prior regularisation", "KL prevents posterior collapse", "vae_loss() total"],
        ["Gaussian MLE = MSE", "BCE preferred for [0,1] pixels", "Decoder Sigmoid"],
    ]
    st = Table(summary, colWidths=[(W-2*MARGIN)*f for f in [0.28, 0.38, 0.34]])
    st.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGRAY]),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.5*cm))

    story.append(callout(
        "The ELBO objective — E_q[log p(x|z)] - KL(q_φ||p) — unifies all Bayesian concepts: "
        "it simultaneously maximises likelihood (reconstruction), applies a prior (KL), "
        "and performs approximate posterior inference (encoder). "
        "This is Bayesian deep learning in its most elegant form."))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Repository", ST["h2"]))
    story.append(Paragraph(
        "Full source code, trained checkpoints, and analysis notebooks are available at:<br/>"
        "<b>github.com/MarinKIm/vae-cifar10</b>",
        ST["body"]))

# ── Build PDF ─────────────────────────────────────────────────────────────
print("\nBuilding PDF …")

doc = SimpleDocTemplate(
    OUT_PDF,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="Bayesian Probability Theory and Variational Autoencoders",
    author="MarinKIm",
    subject="VAE CIFAR-10 — Bayesian foundations",
)

story = []
cover_page(story)
toc_page(story)
section_bayesian_framework(story)
section_bayes_theorem(story)
section_prior(story)
section_likelihood(story)
section_posterior(story)
section_kl(story)
section_mle_map(story)
section_results(story)
section_conclusion(story)

doc.build(story)
print(f"\nDone → {OUT_PDF}")

import os
size_mb = os.path.getsize(OUT_PDF) / 1e6
print(f"File size: {size_mb:.1f} MB")
