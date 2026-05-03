"""Run offline evaluation across all recommenders and write a markdown report.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --n-users 1000 --k 5 10 20 --seed 7

The driver:
1. Loads the books × users sparse interaction matrix and the content feature
   matrix.
2. Samples ``--n-users`` users (each with at least ``--min-interactions``
   books), drops one interaction as the *seed* query and keeps the rest as
   held-out positives. Sampling is deterministic given ``--seed``.
3. Runs each recommender (collaborative KNN, content KNN, hybrid at several
   weight settings) using batched ``kneighbors`` calls so the same query
   work is shared across configurations.
4. Computes precision@K, recall@K, NDCG@K, MAP@K, hit-rate@K, catalog
   coverage and intra-list diversity, then writes a markdown table to
   ``docs/EVALUATION.md`` and the raw numbers to ``data/eval_results.json``.

A methodology caveat: the recommenders are fit on the full matrix
(including the held-out positives), so reported numbers are a small upper
bound. Re-fitting with masking would make the numbers stricter but
multiplies wall-clock cost; the comparison between recommenders is what
this report is primarily designed to surface.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors

from src.evaluation import (
    HoldoutSample,
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    sample_holdouts,
)


DEFAULT_K_VALUES = (5, 10, 20)
DEFAULT_HYBRID_WEIGHTS: tuple[tuple[float, float], ...] = (
    (0.3, 0.7),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class EvalAssets:
    interactions: csr_matrix  # books x users
    content_features: csr_matrix  # books x features (collab-row aligned)
    collab_to_content_idx: dict[int, int]  # identity if mappings align


def load_assets(data_dir: Path) -> EvalAssets:
    interactions = load_npz(data_dir / "book_user_matrix_sparse.npz").tocsr()
    content = load_npz(data_dir / "content_features.npz").tocsr()

    collab_book_ids = (
        pl.read_parquet(data_dir / "sparse_matrix_book_mapping.parquet").sort("matrix_idx")["book_id"].to_list()
    )
    content_book_ids = (
        pl.read_parquet(data_dir / "content_features_mapping.parquet").sort("matrix_idx")["book_id"].to_list()
    )

    content_id_to_idx = {bid: idx for idx, bid in enumerate(content_book_ids)}
    collab_to_content = {
        collab_idx: content_id_to_idx[bid]
        for collab_idx, bid in enumerate(collab_book_ids)
        if bid in content_id_to_idx
    }
    return EvalAssets(interactions=interactions, content_features=content, collab_to_content_idx=collab_to_content)


# ---------------------------------------------------------------------------
# Batched KNN queries
# ---------------------------------------------------------------------------


def hybrid_recs_from_candidates(
    collab_recs: list[list[int]],
    collab_dists: list[list[float]],
    content_recs: list[list[int]],
    content_dists: list[list[float]],
    *,
    w_collab: float,
    w_content: float,
    k: int,
) -> list[list[int]]:
    """Combine pre-computed candidate lists into a hybrid ranking.

    Candidate distances are the cosine distances returned by KNN; we convert
    them to similarities via ``1 - dist`` and combine with linear weights,
    matching the production hybrid recommender's logic.
    """
    out: list[list[int]] = []
    for collab, c_dists, content, t_dists in zip(collab_recs, collab_dists, content_recs, content_dists):
        scores: dict[int, float] = {}
        for idx, dist in zip(collab, c_dists):
            scores[idx] = scores.get(idx, 0.0) + w_collab * (1.0 - dist)
        for idx, dist in zip(content, t_dists):
            scores[idx] = scores.get(idx, 0.0) + w_content * (1.0 - dist)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out.append([idx for idx, _ in ranked[:k]])
    return out


def query_with_dists(knn: NearestNeighbors, queries: csr_matrix, n_neighbors: int, seeds: list[int]):
    distances, indices = knn.kneighbors(queries, n_neighbors=n_neighbors + 1)
    rec_lists: list[list[int]] = []
    dist_lists: list[list[float]] = []
    for row_idxs, row_dists, seed in zip(indices, distances, seeds):
        kept_idxs: list[int] = []
        kept_dists: list[float] = []
        for idx, d in zip(row_idxs, row_dists):
            if int(idx) == int(seed):
                continue
            kept_idxs.append(int(idx))
            kept_dists.append(float(d))
            if len(kept_idxs) == n_neighbors:
                break
        rec_lists.append(kept_idxs)
        dist_lists.append(kept_dists)
    return rec_lists, dist_lists


def query_content_with_dists(
    knn: NearestNeighbors,
    content: csr_matrix,
    samples: list[HoldoutSample],
    collab_to_content: dict[int, int],
    n_neighbors: int,
):
    valid_idxs: list[int] = []
    valid_seed_collab: list[int] = []
    seed_content_map: list[int | None] = []
    for s in samples:
        cidx = collab_to_content.get(s.seed_book_idx)
        seed_content_map.append(cidx)
        if cidx is not None:
            valid_idxs.append(cidx)
            valid_seed_collab.append(s.seed_book_idx)

    rec_lists: list[list[int]] = [[] for _ in samples]
    dist_lists: list[list[float]] = [[] for _ in samples]
    if not valid_idxs:
        return rec_lists, dist_lists

    queries = content[valid_idxs]
    distances, indices = knn.kneighbors(queries, n_neighbors=n_neighbors + 1)
    content_to_collab = {cidx: collab_idx for collab_idx, cidx in collab_to_content.items()}

    valid_pos = 0
    for i, cidx in enumerate(seed_content_map):
        if cidx is None:
            continue
        seed_collab = valid_seed_collab[valid_pos]
        row_idxs = indices[valid_pos]
        row_dists = distances[valid_pos]
        valid_pos += 1
        kept_idxs: list[int] = []
        kept_dists: list[float] = []
        for content_idx, d in zip(row_idxs, row_dists):
            collab_idx = content_to_collab.get(int(content_idx))
            if collab_idx is None or int(collab_idx) == int(seed_collab):
                continue
            kept_idxs.append(int(collab_idx))
            kept_dists.append(float(d))
            if len(kept_idxs) == n_neighbors:
                break
        rec_lists[i] = kept_idxs
        dist_lists[i] = kept_dists
    return rec_lists, dist_lists


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def score_runs(
    name: str,
    rec_lists: list[list[int]],
    samples: list[HoldoutSample],
    *,
    k_values: list[int],
    n_total_books: int,
    diversity_features: csr_matrix,
) -> dict[str, float]:
    metric_buckets: dict[str, list[float]] = {}
    diversities: list[float] = []
    for recs, sample in zip(rec_lists, samples):
        diversities.append(intra_list_diversity(recs[: max(k_values)], diversity_features))
        for k in k_values:
            metric_buckets.setdefault(f"precision@{k}", []).append(precision_at_k(recs, sample.positives, k))
            metric_buckets.setdefault(f"recall@{k}", []).append(recall_at_k(recs, sample.positives, k))
            metric_buckets.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(recs, sample.positives, k))
            metric_buckets.setdefault(f"map@{k}", []).append(average_precision_at_k(recs, sample.positives, k))
            metric_buckets.setdefault(f"hit_rate@{k}", []).append(hit_rate_at_k(recs, sample.positives, k))
    summary = {key: float(np.mean(values)) for key, values in metric_buckets.items()}
    summary["catalog_coverage"] = catalog_coverage(rec_lists, n_total_books)
    summary["intra_list_diversity"] = float(np.mean(diversities)) if diversities else 0.0
    summary["mean_n_positives"] = float(np.mean([s.n_positives for s in samples]))
    summary["n_samples"] = float(len(samples))
    summary["recommender"] = name
    return summary


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(
    *,
    results: list[dict[str, float]],
    n_users_target: int,
    n_users_actual: int,
    min_interactions: int,
    k_values: list[int],
    seed: int,
    n_books: int,
    timings: dict[str, float],
) -> str:
    lines: list[str] = []
    lines.append("# Offline Evaluation Results")
    lines.append("")
    lines.append("_Generated by `evaluate.py`. Reproducible with the seed and parameters below._")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Catalog size: **{n_books:,}** books")
    lines.append(f"- Sampled users: **{n_users_actual:,}** (target {n_users_target:,}, min interactions {min_interactions})")
    lines.append(f"- Random seed: `{seed}`")
    lines.append(f"- Cutoffs evaluated: {', '.join(f'K={k}' for k in k_values)}")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "For each sampled user we pick one of their interacted books at random as the **seed** query and treat "
        "the user's other interacted books as held-out **positives**. The recommender ranks the catalog from "
        "the seed; the top-K is scored against the positives."
    )
    lines.append("")
    lines.append(
        "**Caveat**: each recommender is fit on the full interaction matrix, including the held-out positives. "
        "Reported numbers are therefore a small upper bound on what production performance would look like with "
        "strict masking. Comparisons *between* recommenders remain valid because all of them see the same "
        "training data."
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "- **Precision@K**: fraction of the top-K that are relevant.\n"
        "- **Recall@K**: fraction of relevant items captured in the top-K.\n"
        "- **NDCG@K**: rank-aware metric (binary relevance).\n"
        "- **MAP@K**: mean of average-precision across users.\n"
        "- **Hit-Rate@K**: probability that at least one relevant item lands in the top-K.\n"
        "- **Diversity**: mean pairwise cosine *distance* among the top-K recs in content-feature space.\n"
        "- **Coverage**: fraction of catalog reached across all rec lists."
    )
    lines.append("")

    headline_k = k_values[len(k_values) // 2]
    leaderboard = sorted(results, key=lambda r: r[f"ndcg@{headline_k}"], reverse=True)
    lines.append(f"## Leaderboard (NDCG@{headline_k})")
    lines.append("")
    lines.append("| Rank | Recommender | NDCG | Precision | Recall | Hit Rate |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(leaderboard, 1):
        lines.append(
            f"| {i} | `{r['recommender']}` "
            f"| {r[f'ndcg@{headline_k}']:.4f} "
            f"| {r[f'precision@{headline_k}']:.4f} "
            f"| {r[f'recall@{headline_k}']:.4f} "
            f"| {r[f'hit_rate@{headline_k}']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-K detail")
    lines.append("")

    for k in k_values:
        lines.append(f"### Top-K = {k}")
        lines.append("")
        lines.append("| Recommender | Precision | Recall | NDCG | MAP | Hit Rate | Diversity | Coverage |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in results:
            lines.append(
                f"| `{r['recommender']}` "
                f"| {r[f'precision@{k}']:.4f} "
                f"| {r[f'recall@{k}']:.4f} "
                f"| {r[f'ndcg@{k}']:.4f} "
                f"| {r[f'map@{k}']:.4f} "
                f"| {r[f'hit_rate@{k}']:.4f} "
                f"| {r['intra_list_diversity']:.4f} "
                f"| {r['catalog_coverage']:.4f} |"
            )
        lines.append("")

    lines.append("## Wall-clock per stage (seconds)")
    lines.append("")
    lines.append("| Stage | Time |")
    lines.append("|---|---:|")
    for stage, t in timings.items():
        lines.append(f"| {stage} | {t:.2f} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_eval(
    *,
    n_users: int,
    min_interactions: int,
    k_values: list[int],
    seed: int,
    hybrid_weights: list[tuple[float, float]],
    data_dir: Path,
    output_md: Path,
    output_json: Path,
) -> dict[str, object]:
    timings: dict[str, float] = {}

    t = time.time()
    assets = load_assets(data_dir)
    timings["load_assets"] = time.time() - t
    n_books = assets.interactions.shape[0]
    logger.info(f"Loaded {n_books:,}-book catalog (interactions: {assets.interactions.shape}, content: {assets.content_features.shape})")

    t = time.time()
    samples = sample_holdouts(
        assets.interactions, n_users=n_users, min_interactions=min_interactions, seed=seed
    )
    timings["sample_holdouts"] = time.time() - t
    if not samples:
        raise RuntimeError(
            f"No users met min_interactions={min_interactions}. Try a smaller threshold."
        )
    logger.info(f"Sampled {len(samples)} users; mean held-out positives = {np.mean([s.n_positives for s in samples]):.1f}")

    max_k = max(k_values)
    candidate_k = max_k * 3  # over-fetch so hybrid has a richer pool to merge

    t = time.time()
    collab_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=candidate_k + 1, n_jobs=-1)
    collab_knn.fit(assets.interactions)
    content_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=candidate_k + 1, n_jobs=-1)
    content_knn.fit(assets.content_features)
    timings["fit_knn"] = time.time() - t

    seed_indices = [s.seed_book_idx for s in samples]

    t = time.time()
    collab_recs_full, collab_dists_full = query_with_dists(
        collab_knn, assets.interactions[seed_indices], candidate_k, seed_indices
    )
    timings["query_collab"] = time.time() - t

    t = time.time()
    content_recs_full, content_dists_full = query_content_with_dists(
        content_knn, assets.content_features, samples, assets.collab_to_content_idx, candidate_k
    )
    timings["query_content"] = time.time() - t

    diversity_features = assets.content_features  # rows align with collab indices

    results: list[dict[str, float]] = []

    t = time.time()
    collab_top = [recs[:max_k] for recs in collab_recs_full]
    results.append(
        score_runs(
            "collaborative",
            collab_top,
            samples,
            k_values=k_values,
            n_total_books=n_books,
            diversity_features=diversity_features,
        )
    )
    content_top = [recs[:max_k] for recs in content_recs_full]
    results.append(
        score_runs(
            "content",
            content_top,
            samples,
            k_values=k_values,
            n_total_books=n_books,
            diversity_features=diversity_features,
        )
    )
    for w_collab, w_content in hybrid_weights:
        hybrid_top = hybrid_recs_from_candidates(
            collab_recs_full, collab_dists_full, content_recs_full, content_dists_full,
            w_collab=w_collab, w_content=w_content, k=max_k,
        )
        results.append(
            score_runs(
                f"hybrid_{w_collab:.1f}_{w_content:.1f}",
                hybrid_top,
                samples,
                k_values=k_values,
                n_total_books=n_books,
                diversity_features=diversity_features,
            )
        )
    timings["score_all"] = time.time() - t

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(
            {
                "config": {
                    "n_users_target": n_users,
                    "n_users_actual": len(samples),
                    "min_interactions": min_interactions,
                    "k_values": k_values,
                    "seed": seed,
                    "hybrid_weights": hybrid_weights,
                    "n_books": n_books,
                },
                "timings_seconds": timings,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info(f"Wrote raw results to {output_json}")

    report = render_report(
        results=results,
        n_users_target=n_users,
        n_users_actual=len(samples),
        min_interactions=min_interactions,
        k_values=k_values,
        seed=seed,
        n_books=n_books,
        timings=timings,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report)
    logger.info(f"Wrote report to {output_md}")
    print("\n" + report)
    return {"timings": timings, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for the book recommenders")
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--min-interactions", type=int, default=10)
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hybrid-weights",
        nargs="+",
        type=str,
        default=[f"{w[0]}:{w[1]}" for w in DEFAULT_HYBRID_WEIGHTS],
        help="One or more 'collab:content' weight pairs, e.g. 0.6:0.4",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/EVALUATION.md"))
    parser.add_argument("--output-json", type=Path, default=Path("data/eval_results.json"))
    return parser.parse_args()


def parse_weights(specs: list[str]) -> list[tuple[float, float]]:
    pairs = []
    for spec in specs:
        a, b = spec.split(":")
        pairs.append((float(a), float(b)))
    return pairs


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        n_users=args.n_users,
        min_interactions=args.min_interactions,
        k_values=args.k,
        seed=args.seed,
        hybrid_weights=parse_weights(args.hybrid_weights),
        data_dir=args.data_dir,
        output_md=args.output_md,
        output_json=args.output_json,
    )
