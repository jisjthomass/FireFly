"""
fetch_connectome.py — Sparse Biological Connectome Extraction
=============================================================
Connects to the Princeton FlyWire public database and extracts
real Drosophila melanogaster synaptic connections as a PyTorch
sparse COO tensor. Output is ~2MB instead of ~5GB dense.
"""

import os
import torch
import pandas as pd
import numpy as np
from caveclient import CAVEclient


def fetch_real_flywire_brain(token: str, limit: int = 100_000) -> None:
    """
    Authenticate with FlyWire, download synapses, and save a sparse
    adjacency tensor plus metadata to disk.

    Args:
        token: FlyWire API token.
        limit: Maximum number of synapses to fetch.
    """
    print("═══════════════════════════════════════════════")
    print("  FlyWire Connectome Extraction Pipeline")
    print("═══════════════════════════════════════════════")

    # ── Authenticate ──
    print("\n[1/5] Authenticating with FlyWire CAVEclient...")
    client = CAVEclient()
    client.auth.save_token(token=token, overwrite=True)
    client = CAVEclient("flywire_fafb_public")
    print("       ✓ Authenticated successfully.")

    # ── Fetch synapses ──
    print(f"\n[2/5] Fetching {limit:,} biological synapses...")
    synapses = client.materialize.query_table("synapses_nt_v1", limit=limit)
    print(f"       ✓ Retrieved {len(synapses):,} synapses.")

    # ── Map neuron IDs ──
    print("\n[3/5] Mapping unique neuron identifiers...")
    pre_ids = synapses["pre_pt_root_id"].values
    post_ids = synapses["post_pt_root_id"].values
    all_ids = np.unique(np.concatenate((pre_ids, post_ids)))
    num_neurons = len(all_ids)
    neuron_to_idx = {root_id: idx for idx, root_id in enumerate(all_ids)}
    print(f"       ✓ Found {num_neurons:,} unique neurons.")

    # ── Build sparse COO tensor ──
    print("\n[4/5] Building sparse adjacency tensor...")
    row_indices = np.array([neuron_to_idx[rid] for rid in pre_ids], dtype=np.int64)
    col_indices = np.array([neuron_to_idx[rid] for rid in post_ids], dtype=np.int64)

    # Deduplicate edges (multiple synapses between the same pair → single edge)
    edge_set = set(zip(row_indices, col_indices))
    rows = np.array([e[0] for e in edge_set], dtype=np.int64)
    cols = np.array([e[1] for e in edge_set], dtype=np.int64)

    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    sparse_connectome = torch.sparse_coo_tensor(
        indices, values, size=(num_neurons, num_neurons)
    ).coalesce()

    nnz = sparse_connectome._nnz()
    density = nnz / (num_neurons * num_neurons)
    print(f"       ✓ Sparse tensor: {num_neurons:,} × {num_neurons:,}")
    print(f"         Edges (nnz):   {nnz:,}")
    print(f"         Density:       {density:.6f}")

    # ── Save to disk ──
    print("\n[5/5] Saving to disk...")

    # Save the sparse connectome
    torch.save(sparse_connectome, "sparse_connectome.pt")
    sparse_size = os.path.getsize("sparse_connectome.pt") / (1024 * 1024)
    print(f"       ✓ sparse_connectome.pt ({sparse_size:.1f} MB)")

    # Save metadata for reproducibility and the UI
    metadata = {
        "num_neurons": num_neurons,
        "num_synapses": nnz,
        "density": density,
        "neuron_to_idx": neuron_to_idx,
        "idx_to_neuron": {v: k for k, v in neuron_to_idx.items()},
        "datastack": "flywire_fafb_public",
        "synapse_table": "synapses_nt_v1",
        "limit": limit,
    }
    torch.save(metadata, "connectome_metadata.pt")
    print("       ✓ connectome_metadata.pt")

    print("\n═══════════════════════════════════════════════")
    print("  Extraction complete.")
    print(f"  {num_neurons:,} neurons · {nnz:,} synapses · {sparse_size:.1f} MB")
    print("═══════════════════════════════════════════════")


if __name__ == "__main__":
    TOKEN = os.environ.get("FLYWIRE_TOKEN", "")
    if not TOKEN:
        print("ERROR: Set FLYWIRE_TOKEN environment variable.")
        print("  export FLYWIRE_TOKEN='your_token_here'")
        exit(1)
    fetch_real_flywire_brain(TOKEN)
