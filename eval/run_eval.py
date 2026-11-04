"""
Evaluation script for GrantScout.
Runs extraction on labeled grants and reports per-field accuracy.
Also computes matching precision@3 against sample startup profiles.

Usage:
    python eval/run_eval.py --ground-truth eval/ground_truth.csv --profiles eval/sample_profiles.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grantscout.eval")


# ─── Field matching helpers ───────────────────────────────────────────────────

def _normalize(val: Any) -> str:
    if val is None:
        return ""
    return str(val).lower().strip()


def _amount_match(pred: Any, gt: Any, tolerance: float = 0.2) -> bool:
    """Match numeric INR amounts within a percentage tolerance."""
    try:
        p = float(str(pred).replace(",", ""))
        g = float(str(gt).replace(",", ""))
        if g == 0:
            return p == 0
        return abs(p - g) / g <= tolerance
    except (ValueError, TypeError):
        return _normalize(pred) == _normalize(gt)


def _deadline_match(pred: Any, gt: Any) -> bool:
    """Loose deadline match: YYYY-MM-DD or 'rolling' / 'N/A'."""
    p = _normalize(pred)
    g = _normalize(gt)
    if not g or g in ("null", "none", ""):
        return True  # no ground truth, skip
    return p == g or p[:7] == g[:7]  # year-month match


def _list_overlap(pred: list, gt: list, threshold: float = 0.5) -> bool:
    """True if overlap ratio >= threshold (case-insensitive substring)."""
    if not gt:
        return True
    pred_str = " ".join(_normalize(x) for x in pred)
    matches = sum(1 for g in gt if _normalize(g) in pred_str or pred_str in _normalize(g))
    return matches / len(gt) >= threshold


# ─── Extract from all synthetic files ────────────────────────────────────────

def _extract_all_synthetic() -> list[dict]:
    """Run extraction pipeline on all synthetic .txt files."""
    from core.ingestion import ingest_document
    from core.extraction import extract_from_chunks

    synthetic_dir = Path(__file__).parent.parent / "data" / "synthetic"
    results = []
    for txt_file in sorted(synthetic_dir.glob("*.txt")):
        logger.info("Extracting: %s", txt_file.name)
        try:
            chunks = ingest_document(txt_file, store_in_cos=False)
            grants = extract_from_chunks(chunks)
            for g in grants:
                results.append({"file": txt_file.name, "grant": g})
        except Exception as exc:
            logger.error("Failed on %s: %s", txt_file.name, exc)
    return results


# ─── Extraction accuracy ──────────────────────────────────────────────────────

FIELDS = ["name", "issuing_body", "funding_amount_inr", "deadline_date", "eligible_stages", "sectors", "location_restrictions"]


def compute_extraction_accuracy(ground_truth_csv: Path, extracted: list[dict]) -> pd.DataFrame:
    """
    Compare extracted fields against ground truth CSV.
    Returns a DataFrame with per-field accuracy.
    """
    with open(ground_truth_csv, newline="", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    # Build map: grant name -> gt row
    gt_map = {_normalize(row["name"]): row for row in gt_rows}

    field_results: dict[str, list[bool]] = {f: [] for f in FIELDS}

    for item in extracted:
        grant = item["grant"]
        key = _normalize(grant.name)
        gt = gt_map.get(key)
        if not gt:
            logger.warning("No ground truth for: %s", grant.name)
            continue

        field_results["name"].append(True)  # if we got here, name matched
        field_results["issuing_body"].append(
            _normalize(gt.get("issuing_body")) in _normalize(grant.issuing_body or "")
            or _normalize(grant.issuing_body or "") in _normalize(gt.get("issuing_body", ""))
        )
        field_results["funding_amount_inr"].append(
            _amount_match(grant.funding_amount_inr, gt.get("funding_amount_inr"))
        )
        field_results["deadline_date"].append(
            _deadline_match(grant.deadline_date, gt.get("deadline_date"))
        )
        gt_stages = [s.strip() for s in gt.get("eligible_stages", "").split(",") if s.strip()]
        field_results["eligible_stages"].append(
            _list_overlap([str(s) for s in grant.eligible_stages], gt_stages)
        )
        gt_sectors = [s.strip() for s in gt.get("sectors", "").split(",") if s.strip()]
        field_results["sectors"].append(_list_overlap(grant.sectors, gt_sectors))
        field_results["location_restrictions"].append(
            _normalize(gt.get("location_restrictions", "")) in _normalize(grant.location_restrictions or "")
        )

    rows = []
    for field, results in field_results.items():
        if results:
            acc = sum(results) / len(results)
            rows.append({"field": field, "correct": sum(results), "total": len(results), "accuracy": round(acc, 3)})
    return pd.DataFrame(rows)


# ─── Matching precision@3 ─────────────────────────────────────────────────────

def compute_matching_precision(profiles_json: Path, grants: list) -> pd.DataFrame:
    """
    For each sample startup profile, run matching and check if expected grants appear in top-3.
    profiles_json: list of {profile: {...}, expected_grants: ["name1", "name2"]}
    """
    from core.schemas import StartupProfile
    from core.matching import match_grants

    with open(profiles_json, encoding="utf-8") as f:
        test_cases = json.load(f)

    rows = []
    for tc in test_cases:
        profile_data = tc["profile"]
        expected = [_normalize(n) for n in tc.get("expected_grants", [])]
        try:
            profile = StartupProfile(**profile_data)
            matches = match_grants(profile, grants, top_k=3)
            top3_names = [_normalize(m.grant.name) for m in matches]
            hits = sum(1 for e in expected if any(e in t or t in e for t in top3_names))
            p3 = hits / max(len(expected), 1)
            rows.append({
                "profile": profile.name,
                "expected_grants": ", ".join(tc.get("expected_grants", [])),
                "top3_retrieved": " | ".join(m.grant.name[:30] for m in matches),
                "precision@3": round(p3, 3),
            })
        except Exception as exc:
            rows.append({"profile": str(tc), "expected_grants": "", "top3_retrieved": f"ERROR: {exc}", "precision@3": 0.0})

    return pd.DataFrame(rows)


# ─── Output helpers ───────────────────────────────────────────────────────────

def _print_markdown_table(df: pd.DataFrame, title: str):
    print(f"\n## {title}\n")
    print(df.to_markdown(index=False))


def _save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved: %s", path)


def _save_chart(df: pd.DataFrame, title: str, path: Path):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df["field"] if "field" in df.columns else df["profile"], df.iloc[:, -1])
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path))
        logger.info("Chart saved: %s", path)
        plt.close()
    except ImportError:
        logger.warning("matplotlib not installed – skipping chart generation.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GrantScout Evaluation")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).parent / "ground_truth.csv",
        help="Path to ground truth CSV",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).parent / "sample_profiles.json",
        help="Path to sample profiles JSON for matching eval",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory for output CSVs and charts",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GrantScout Evaluation")
    print("=" * 60)

    # Step 1: Extract
    print("\n[1/3] Running extraction on synthetic documents...")
    extracted = _extract_all_synthetic()
    grants = [item["grant"] for item in extracted]
    print(f"  Extracted {len(grants)} grants.")

    # Step 2: Extraction accuracy
    if args.ground_truth.exists():
        print("\n[2/3] Computing extraction accuracy...")
        acc_df = compute_extraction_accuracy(args.ground_truth, extracted)
        _print_markdown_table(acc_df, "Extraction Accuracy by Field")
        _save_csv(acc_df, args.output_dir / "extraction_accuracy.csv")
        _save_chart(acc_df, "Extraction Field Accuracy", args.output_dir / "extraction_accuracy.png")
    else:
        print(f"\n[2/3] Ground truth CSV not found at {args.ground_truth} – skipping extraction eval.")
        print("       Run with a ground_truth.csv (see template: eval/ground_truth_template.csv)")
        acc_df = pd.DataFrame()

    # Step 3: Matching precision
    if args.profiles.exists() and grants:
        print("\n[3/3] Computing matching precision@3...")
        prec_df = compute_matching_precision(args.profiles, grants)
        _print_markdown_table(prec_df, "Matching Precision@3")
        _save_csv(prec_df, args.output_dir / "matching_precision.csv")
        avg = prec_df["precision@3"].mean() if len(prec_df) else 0
        print(f"\n  Average Precision@3: {avg:.3f}")
    else:
        print("\n[3/3] Sample profiles JSON not found or no grants extracted – skipping matching eval.")

    print("\n✅ Evaluation complete. Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
