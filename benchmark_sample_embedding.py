# ============================================================
# Atlas Tissue VAE — Sample Embedding Benchmark
#
# Native representation:
# bulk RNA sample -> encoder -> latent mean mu
#
# Expected output:
# 100 samples x 121 dimensions
# ============================================================

from pathlib import Path
import time
import os

import joblib
import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(r"E:\atlas_tissue_representation_fork")

MODEL_PATH = (
    ROOT
    / "model"
    / "vae_tissue.final_model.pth"
)

ARTIFACT_PATH = (
    ROOT
    / "model"
    / "vae_tissue.artifacts.joblib"
)

INPUT_PATH = (
    ROOT
    / "gex.lung_100.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "atlas_tissue_vae"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Load model
#
# IMPORTANT:
# Model-loading time is kept separate from embedding inference.
# ------------------------------------------------------------

print("Loading full pretrained VAE...")

load_start = time.perf_counter()

model = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

model.eval()

model_load_time = (
    time.perf_counter()
    - load_start
)

print(
    "Model load seconds:",
    round(model_load_time, 2),
)


# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------

artifacts = joblib.load(
    ARTIFACT_PATH
)

gene_list = list(
    artifacts[
        "feature_lists"
    ]["gex"]
)

scaler = artifacts[
    "transforms"
]["gex"]

print(
    "Model genes:",
    len(gene_list),
)


# ------------------------------------------------------------
# Load expression data
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_PATH,
    index_col=0,
)


# Detect orientation
genes_in_columns = len(
    set(df.columns)
    & set(gene_list)
)

genes_in_rows = len(
    set(df.index)
    & set(gene_list)
)

if genes_in_rows > genes_in_columns:
    df = df.T


print(
    "Input samples:",
    df.shape[0],
)

print(
    "Input genes:",
    df.shape[1],
)


# ------------------------------------------------------------
# Align input to model's 16,115-gene feature space
# ------------------------------------------------------------

aligned = pd.DataFrame(
    0.0,
    index=df.index,
    columns=gene_list,
)

common_genes = [
    gene
    for gene in gene_list
    if gene in df.columns
]

aligned[common_genes] = (
    df[common_genes].values
)

aligned = aligned.fillna(0)


print(
    "Matched genes:",
    len(common_genes),
)

print(
    "Gene coverage:",
    round(
        100
        * len(common_genes)
        / len(gene_list),
        2,
    ),
    "%",
)


# ------------------------------------------------------------
# Apply stored training-data scaler
# ------------------------------------------------------------

scaled = scaler.transform(
    aligned
)

X = torch.tensor(
    scaled,
    dtype=torch.float32,
)


print(
    "Model input shape:",
    tuple(X.shape),
)


# ------------------------------------------------------------
# Benchmark native sample embedding
#
# Repository implementation:
#
# h  = model.encoders[0](X)
# mu = model.FC_mean(h[0])
# ------------------------------------------------------------

start = time.perf_counter()

with torch.no_grad():

    h = model.encoders[0](X)

    mu = model.FC_mean(
        h[0]
    )


elapsed = (
    time.perf_counter()
    - start
)


# ------------------------------------------------------------
# Convert
# ------------------------------------------------------------

embeddings = (
    mu
    .detach()
    .cpu()
    .numpy()
    .astype(np.float32)
)


# ------------------------------------------------------------
# Validate
# ------------------------------------------------------------

print()
print(
    "ATLAS TISSUE VAE SAMPLE EMBEDDING"
)
print("=" * 60)

print(
    "Samples:",
    embeddings.shape[0],
)

print(
    "Output shape:",
    embeddings.shape,
)

print(
    "Dimension:",
    embeddings.shape[-1],
)

print(
    "dtype:",
    embeddings.dtype,
)

print(
    "min:",
    embeddings.min(),
)

print(
    "max:",
    embeddings.max(),
)

print(
    "mean:",
    embeddings.mean(),
)

print(
    "std:",
    embeddings.std(),
)

print(
    "NaN:",
    np.isnan(
        embeddings
    ).sum(),
)

print(
    "Inf:",
    np.isinf(
        embeddings
    ).sum(),
)

print(
    "Inference time seconds:",
    round(
        elapsed,
        6,
    ),
)

print(
    "Seconds per sample:",
    round(
        elapsed
        / embeddings.shape[0],
        6,
    ),
)

print(
    "Model load seconds:",
    round(
        model_load_time,
        2,
    ),
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_path = (
    OUTPUT_DIR
    / "sample_embeddings.npy"
)

np.save(
    output_path,
    embeddings,
)

np.save(
    OUTPUT_DIR
    / "sample_names.npy",
    np.asarray(
        df.index,
        dtype=object,
    ),
)


print(
    "Saved:",
    output_path,
)