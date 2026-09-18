"""
bio_analysis.py — Biological Network Analysis
================================================
Computes graph-theoretic properties of the Drosophila connectome
and compares against random network baselines (Erdős–Rényi, Watts–Strogatz).
"""

import torch
import networkx as nx
import numpy as np
import json
import os
import random


def load_connectome_as_graph(connectome_path: str = "sparse_connectome.pt") -> nx.DiGraph:
    """Load the sparse connectome tensor and convert to a NetworkX directed graph."""
    sparse_adj = torch.load(connectome_path, weights_only=False)
    adj = sparse_adj.coalesce()
    indices = adj.indices().numpy()
    G = nx.DiGraph()
    G.add_nodes_from(range(sparse_adj.shape[0]))
    for i in range(indices.shape[1]):
        G.add_edge(int(indices[0, i]), int(indices[1, i]))
    return G


def analyze_connectome(G: nx.DiGraph) -> dict:
    """
    Compute comprehensive graph metrics for the biological connectome.
    
    Returns a dict with:
    - num_nodes, num_edges
    - density
    - avg_in_degree, avg_out_degree
    - degree_distribution (histogram as list)
    - clustering_coefficient (on undirected version)
    - num_strongly_connected_components
    - largest_scc_size
    - num_weakly_connected_components  
    - largest_wcc_size
    - avg_shortest_path_length (estimated on largest WCC subsample if too large)
    - is_small_world (bool) - compare clustering and path length to random
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes == 0:
        return {
            "num_nodes": 0, "num_edges": 0, "density": 0.0,
            "avg_in_degree": 0.0, "avg_out_degree": 0.0,
            "degree_distribution": [], "clustering_coefficient": 0.0,
            "num_strongly_connected_components": 0, "largest_scc_size": 0,
            "num_weakly_connected_components": 0, "largest_wcc_size": 0,
            "avg_shortest_path_length": 0.0, "is_small_world": False
        }
        
    density = nx.density(G)
    
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    avg_in_degree = sum(in_degrees) / num_nodes if num_nodes > 0 else 0
    avg_out_degree = sum(out_degrees) / num_nodes if num_nodes > 0 else 0
    
    # Degree distribution histogram
    degrees = [d for n, d in G.degree()]
    hist, bins = np.histogram(degrees, bins=20)
    degree_distribution = hist.tolist()
    
    # Clustering coefficient requires an undirected graph
    G_undirected = G.to_undirected()
    
    # Approximation if the graph is large
    if num_nodes > 5000:
        sampled_nodes = random.sample(list(G_undirected.nodes()), 2000)
        clustering_coefficient = nx.average_clustering(G_undirected, nodes=sampled_nodes)
    else:
        clustering_coefficient = nx.average_clustering(G_undirected)
        
    # Strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    num_scc = len(sccs)
    largest_scc_size = len(max(sccs, key=len)) if num_scc > 0 else 0
    
    # Weakly connected components
    wccs = list(nx.weakly_connected_components(G))
    num_wcc = len(wccs)
    largest_wcc = max(wccs, key=len) if num_wcc > 0 else set()
    largest_wcc_size = len(largest_wcc)
    
    # Average shortest path length
    # Sample nodes from the largest WCC to estimate avg shortest path
    avg_shortest_path_length = 0.0
    if largest_wcc_size > 0:
        subgraph = G.subgraph(largest_wcc)
        if largest_wcc_size > 1000:
            sample_size = 100
            sampled_nodes = random.sample(list(largest_wcc), sample_size)
            path_lengths = []
            for n in sampled_nodes:
                lengths = nx.single_source_shortest_path_length(subgraph, n)
                path_lengths.extend(lengths.values())
            avg_shortest_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0
        else:
            avg_shortest_path_length = nx.average_shortest_path_length(subgraph)
    
    metrics = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": float(density),
        "avg_in_degree": float(avg_in_degree),
        "avg_out_degree": float(avg_out_degree),
        "degree_distribution": degree_distribution,
        "clustering_coefficient": float(clustering_coefficient),
        "num_strongly_connected_components": num_scc,
        "largest_scc_size": largest_scc_size,
        "num_weakly_connected_components": num_wcc,
        "largest_wcc_size": largest_wcc_size,
        "avg_shortest_path_length": float(avg_shortest_path_length),
        "is_small_world": False  # Will be calculated in the full analysis
    }
    
    return metrics


def generate_random_baselines(num_nodes: int, num_edges: int) -> dict:
    """
    Generate Erdős–Rényi and Watts–Strogatz random graphs with matching
    size and density, then compute the same metrics.
    
    For Watts-Strogatz: use k = avg_degree (rounded), p = 0.1
    For Erdős–Rényi: use the same density as the real connectome
    
    Returns dict with 'erdos_renyi' and 'watts_strogatz' keys,
    each containing the same metric structure.
    """
    if num_nodes == 0:
        return {"erdos_renyi": {}, "watts_strogatz": {}}
        
    avg_degree = round((num_edges / num_nodes) * 2) if num_nodes > 0 else 0
    avg_degree = max(2, avg_degree) # WS requires k >= 2
    
    p_er = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    
    print("      - Generating Erdős-Rényi...")
    G_er = nx.erdos_renyi_graph(num_nodes, p_er, directed=True)
    metrics_er = analyze_connectome(G_er)
    
    print("      - Generating Watts-Strogatz...")
    try:
        G_ws = nx.watts_strogatz_graph(num_nodes, avg_degree, 0.1)
        G_ws_dir = G_ws.to_directed()
    except Exception as e:
        print(f"      - Watts-Strogatz error: {e}")
        G_ws_dir = nx.DiGraph()
    metrics_ws = analyze_connectome(G_ws_dir)
    
    return {
        "erdos_renyi": metrics_er,
        "watts_strogatz": metrics_ws
    }


def compute_degree_centrality_stats(G: nx.DiGraph) -> dict:
    """
    Compute degree centrality statistics:
    - Top 10 hub neurons (highest out-degree)
    - Top 10 authority neurons (highest in-degree)
    - Degree centrality distribution stats (mean, std, max, min)
    """
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return {}
        
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    top_authorities = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    top_hubs = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    
    in_vals = list(in_degrees.values())
    out_vals = list(out_degrees.values())
    
    stats = {
        "top_10_hubs": [{"node": int(n), "out_degree": int(d)} for n, d in top_hubs],
        "top_10_authorities": [{"node": int(n), "in_degree": int(d)} for n, d in top_authorities],
        "out_degree_stats": {
            "mean": float(np.mean(out_vals)),
            "std": float(np.std(out_vals)),
            "max": int(np.max(out_vals)),
            "min": int(np.min(out_vals))
        },
        "in_degree_stats": {
            "mean": float(np.mean(in_vals)),
            "std": float(np.std(in_vals)),
            "max": int(np.max(in_vals)),
            "min": int(np.min(in_vals))
        }
    }
    
    return stats


def run_full_analysis(connectome_path: str = "sparse_connectome.pt") -> dict:
    """
    Run the complete analysis pipeline and save results to bio_analysis_results.json.
    
    Returns the full results dict.
    """
    print("═══════════════════════════════════════════════")
    print("  Biological Network Analysis")
    print("═══════════════════════════════════════════════\n")
    
    if not os.path.exists(connectome_path):
        print(f"File {connectome_path} not found. Using a generated graph instead.")
        G = nx.erdos_renyi_graph(1000, 0.05, directed=True)
    else:
        G = load_connectome_as_graph(connectome_path)
    
    print("[1/3] Analyzing biological connectome...")
    bio_metrics = analyze_connectome(G)
    
    print("[2/3] Generating random baselines...")
    baselines = generate_random_baselines(bio_metrics["num_nodes"], bio_metrics["num_edges"])
    
    # Small world detection: C_bio >> C_rand and L_bio ~ L_rand
    c_bio = bio_metrics.get("clustering_coefficient", 0)
    l_bio = bio_metrics.get("avg_shortest_path_length", 0)
    
    er_metrics = baselines.get("erdos_renyi", {})
    c_rand = er_metrics.get("clustering_coefficient", 0)
    l_rand = er_metrics.get("avg_shortest_path_length", 0)
    
    is_small_world = False
    if c_rand > 0 and l_rand > 0:
        c_ratio = c_bio / c_rand
        l_ratio = l_bio / l_rand
        # Standard threshold for small-worldness: S = (C/C_rand) / (L/L_rand) > 1 
        # Usually C/C_rand >> 1 and L/L_rand ~ 1
        if c_ratio > 1.5 and l_ratio < 1.5:
            is_small_world = True
    bio_metrics["is_small_world"] = is_small_world
    
    print("[3/3] Computing centrality statistics...")
    centrality = compute_degree_centrality_stats(G)
    
    results = {
        "biological": bio_metrics,
        "baselines": baselines,
        "centrality": centrality,
    }
    
    with open("bio_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Saved bio_analysis_results.json")
    
    # Print summary
    print(f"\n  Biological:  {bio_metrics['num_nodes']:,} neurons, {bio_metrics['num_edges']:,} synapses")
    print(f"  Clustering:  {bio_metrics['clustering_coefficient']:.4f}")
    if 'clustering_coefficient' in er_metrics:
        print(f"  ER Random:   {er_metrics['clustering_coefficient']:.4f}")
    if bio_metrics.get('is_small_world'):
        print("  ✓ Small-world topology detected!")
    
    return results


if __name__ == "__main__":
    run_full_analysis()
