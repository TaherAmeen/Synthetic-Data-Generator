"""
Utility Module for Synthetic Data Generator

This module provides utility functions for:
1. Loading and processing data files
2. Generating reports from existing synthetic data
3. Comparing multiple model outputs
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


def load_json_file(filepath: str) -> Any:
    """Load and parse a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def get_output_files(output_dir: str = "data/output") -> List[str]:
    """Get list of all output review files."""
    if not os.path.exists(output_dir):
        return []
    
    files = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            files.append(os.path.join(output_dir, filename))
    return sorted(files)


def extract_model_name_from_filename(filename: str) -> str:
    """Extract model name from output filename."""
    # Expected format: reviews_{model_name}_{timestamp}.json
    basename = os.path.basename(filename)
    parts = basename.replace(".json", "").split("_")
    if len(parts) >= 3:
        # Remove 'reviews' prefix and timestamp suffix
        return "_".join(parts[1:-2])  # Get model name parts
    return "unknown"


def generate_report_from_file(
    synthetic_file: str,
    real_reviews_path: str = "data/real_reviews.json",
    output_dir: str = "reports"
) -> Dict[str, str]:
    """
    Generate quality reports from an existing synthetic reviews file.
    
    Args:
        synthetic_file: Path to synthetic reviews JSON file
        real_reviews_path: Path to real reviews for comparison
        output_dir: Directory to save reports
    
    Returns:
        Dictionary mapping report type to file path
    """
    from modules.embedder import LocalEmbedder
    from modules.reporting import (
        generate_quality_report,
        generate_comparison_report,
        save_report,
        load_real_reviews
    )
    from modules.metrics import ModelPerformanceTracker
    
    # Load data
    synthetic_reviews = load_json_file(synthetic_file)
    real_reviews = load_real_reviews(real_reviews_path)
    
    # Initialize embedder
    embedder = LocalEmbedder()
    
    # Extract model name from filename
    model_name = extract_model_name_from_filename(synthetic_file)
    
    saved_reports = {}
    
    # Generate quality report
    quality_report = generate_quality_report(synthetic_reviews, embedder, model_name)
    saved_reports["quality"] = save_report(quality_report)
    
    # Generate comparison report
    if real_reviews:
        comparison_report = generate_comparison_report(
            synthetic_reviews, real_reviews, embedder, model_name
        )
        saved_reports["comparison"] = save_report(comparison_report)
    
    return saved_reports


def compare_output_files(
    file_paths: List[str],
    real_reviews_path: str = "data/real_reviews.json"
) -> Dict[str, Any]:
    """
    Compare multiple synthetic output files.
    
    Args:
        file_paths: List of paths to synthetic review files
        real_reviews_path: Path to real reviews for comparison
    
    Returns:
        Comparison report dictionary
    """
    from modules.embedder import LocalEmbedder
    from modules.metrics import (
        calculate_text_length_stats,
        calculate_vocabulary_diversity,
        calculate_diversity_metrics,
        calculate_rating_distribution
    )
    from modules.reporting import extract_review_texts, load_real_reviews
    
    embedder = LocalEmbedder()
    real_reviews = load_real_reviews(real_reviews_path)
    
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "files_compared": len(file_paths),
        "models": {}
    }
    
    for filepath in file_paths:
        model_name = extract_model_name_from_filename(filepath)
        reviews = load_json_file(filepath)
        texts = extract_review_texts(reviews)
        comments = [r.get("comment", "") for r in reviews if r.get("comment")]
        
        model_metrics = {
            "file": filepath,
            "review_count": len(reviews),
            "text_length": calculate_text_length_stats(texts),
            "vocabulary": calculate_vocabulary_diversity(texts),
            "rating_distribution": calculate_rating_distribution(reviews)
        }
        
        # Add diversity metrics
        if comments:
            embeddings = embedder.get_embeddings(comments)
            model_metrics["diversity"] = calculate_diversity_metrics(embeddings)
        
        comparison["models"][model_name] = model_metrics
    
    # Add summary comparison
    if len(comparison["models"]) > 1:
        comparison["summary"] = generate_comparison_summary(comparison["models"])
    
    return comparison


def generate_comparison_summary(models_data: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate a summary comparing multiple models."""
    summary = {
        "best_diversity": {"model": None, "score": 0},
        "highest_avg_length": {"model": None, "length": 0},
        "most_vocabulary": {"model": None, "unique_words": 0}
    }
    
    for model_name, data in models_data.items():
        # Check diversity
        diversity = data.get("diversity", {}).get("diversity_score", 0)
        if diversity > summary["best_diversity"]["score"]:
            summary["best_diversity"] = {"model": model_name, "score": diversity}
        
        # Check text length
        avg_length = data.get("text_length", {}).get("mean", 0)
        if avg_length > summary["highest_avg_length"]["length"]:
            summary["highest_avg_length"] = {"model": model_name, "length": avg_length}
        
        # Check vocabulary
        unique = data.get("vocabulary", {}).get("unique_words", 0)
        if unique > summary["most_vocabulary"]["unique_words"]:
            summary["most_vocabulary"] = {"model": model_name, "unique_words": unique}
    
    return summary


# Command-line utility functions

def cli_generate_reports():
    """Command-line interface for generating reports from existing files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate quality reports for synthetic reviews")
    parser.add_argument("input_file", help="Path to synthetic reviews JSON file")
    parser.add_argument("--real", default="data/real_reviews.json", help="Path to real reviews file")
    parser.add_argument("--output-dir", default="reports", help="Directory for output reports")
    
    args = parser.parse_args()
    
    print(f"Generating reports for: {args.input_file}")
    reports = generate_report_from_file(args.input_file, args.real, args.output_dir)
    
    print("Reports generated:")
    for report_type, path in reports.items():
        print(f"  - {report_type}: {path}")


if __name__ == "__main__":
    cli_generate_reports()
