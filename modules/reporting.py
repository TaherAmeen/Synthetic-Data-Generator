"""
Reporting Module for Synthetic Review Quality Assessment

This module generates comprehensive quality reports:
1. Quality Report: Text quality metrics, diversity scores, readability analysis
2. Synthetic vs Real Comparison: How synthetic reviews compare to real ones
3. Model Performance Report: Generation time, success rate, quality per model
4. Combined Summary Reports: Overall assessment and recommendations
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from modules.metrics import (
    calculate_text_length_stats,
    calculate_vocabulary_diversity,
    calculate_readability_stats,
    calculate_diversity_metrics,
    calculate_synthetic_vs_real_similarity,
    calculate_rating_distribution,
    calculate_persona_distribution,
    compare_distributions,
    calculate_overall_quality_score,
    ModelPerformanceTracker
)


REPORTS_DIR = "reports"


def ensure_reports_dir(output_dir: str = None) -> str:
    """Ensure reports directory exists and return the path."""
    target_dir = output_dir if output_dir else REPORTS_DIR
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir


def load_real_reviews(path: str = "data/real_reviews.json") -> List[Dict]:
    """Load real reviews for comparison."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load real reviews from {path}: {e}")
        return []


def extract_review_texts(reviews: List[Dict]) -> List[str]:
    """Extract all text content from reviews for analysis."""
    texts = []
    for r in reviews:
        parts = []
        if r.get("title"):
            parts.append(r["title"])
        if r.get("comment"):
            parts.append(r["comment"])
        if r.get("pros"):
            parts.append(r["pros"])
        if r.get("cons"):
            parts.append(r["cons"])
        if parts:
            texts.append(" ".join(parts))
    return texts


# ==============================================================================
# QUALITY REPORT
# ==============================================================================

def generate_quality_report(
    synthetic_reviews: List[Dict],
    embedder=None,
    model_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Generate a comprehensive quality report for synthetic reviews.
    
    Args:
        synthetic_reviews: List of generated review dictionaries
        embedder: LocalEmbedder instance for similarity calculations
        model_name: Name of the model that generated these reviews
    
    Returns:
        Quality report dictionary
    """
    report = {
        "report_type": "quality_report",
        "generated_at": datetime.now().isoformat(),
        "model": model_name,
        "total_reviews": len(synthetic_reviews),
        "metrics": {}
    }
    
    if not synthetic_reviews:
        report["metrics"]["error"] = "No reviews to analyze"
        return report
    
    # Extract texts for analysis
    texts = extract_review_texts(synthetic_reviews)
    comments = [r.get("comment", "") for r in synthetic_reviews if r.get("comment")]
    
    # Text Quality Metrics
    report["metrics"]["text_quality"] = {
        "length_stats": calculate_text_length_stats(texts),
        "comment_length_stats": calculate_text_length_stats(comments),
        "vocabulary_diversity": calculate_vocabulary_diversity(texts),
        "readability": calculate_readability_stats(texts)
    }
    
    # Distribution Metrics
    report["metrics"]["distributions"] = {
        "rating": calculate_rating_distribution(synthetic_reviews),
        "persona": calculate_persona_distribution(synthetic_reviews)
    }
    
    # Diversity Metrics (if embedder available)
    if embedder and comments:
        try:
            embeddings = embedder.get_embeddings(comments)
            diversity = calculate_diversity_metrics(embeddings)
            report["metrics"]["diversity"] = diversity
        except Exception as e:
            report["metrics"]["diversity"] = {"error": str(e)}
    
    # Calculate summary score
    try:
        diversity_score = report["metrics"].get("diversity", {}).get("diversity_score", 0.5)
        readability_mean = report["metrics"]["text_quality"]["readability"].get("flesch_mean", 60)
        
        report["metrics"]["summary"] = {
            "diversity_score": diversity_score,
            "avg_readability": readability_mean,
            "avg_review_length": report["metrics"]["text_quality"]["length_stats"]["mean"],
            "rating_variance": report["metrics"]["distributions"]["rating"].get("std", 0)
        }
    except Exception as e:
        report["metrics"]["summary"] = {"error": str(e)}
    
    return report


# ==============================================================================
# SYNTHETIC VS REAL COMPARISON REPORT
# ==============================================================================

def generate_comparison_report(
    synthetic_reviews: List[Dict],
    real_reviews: List[Dict],
    embedder=None,
    model_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Generate a comparison report between synthetic and real reviews.
    
    Args:
        synthetic_reviews: List of generated review dictionaries
        real_reviews: List of real review dictionaries
        embedder: LocalEmbedder instance for similarity calculations
        model_name: Name of the model that generated synthetic reviews
    
    Returns:
        Comparison report dictionary
    """
    report = {
        "report_type": "comparison_report",
        "generated_at": datetime.now().isoformat(),
        "model": model_name,
        "synthetic_count": len(synthetic_reviews),
        "real_count": len(real_reviews),
        "comparison": {}
    }
    
    if not synthetic_reviews or not real_reviews:
        report["comparison"]["error"] = "Insufficient data for comparison"
        return report
    
    # Extract texts
    synthetic_texts = extract_review_texts(synthetic_reviews)
    real_texts = extract_review_texts(real_reviews)
    
    synthetic_comments = [r.get("comment", "") for r in synthetic_reviews if r.get("comment")]
    real_comments = [r.get("comment", "") for r in real_reviews if r.get("comment")]
    
    # Text Length Comparison
    report["comparison"]["text_length"] = {
        "synthetic": calculate_text_length_stats(synthetic_texts),
        "real": calculate_text_length_stats(real_texts),
        "difference": {}
    }
    
    syn_mean = report["comparison"]["text_length"]["synthetic"]["mean"]
    real_mean = report["comparison"]["text_length"]["real"]["mean"]
    if real_mean > 0:
        report["comparison"]["text_length"]["difference"] = {
            "absolute": round(syn_mean - real_mean, 2),
            "percentage": round((syn_mean - real_mean) / real_mean * 100, 2)
        }
    
    # Vocabulary Comparison
    report["comparison"]["vocabulary"] = {
        "synthetic": calculate_vocabulary_diversity(synthetic_texts),
        "real": calculate_vocabulary_diversity(real_texts)
    }
    
    # Readability Comparison
    report["comparison"]["readability"] = {
        "synthetic": calculate_readability_stats(synthetic_texts),
        "real": calculate_readability_stats(real_texts)
    }
    
    # Rating Distribution Comparison
    syn_rating_dist = calculate_rating_distribution(synthetic_reviews)
    real_rating_dist = calculate_rating_distribution(real_reviews)
    
    report["comparison"]["rating_distribution"] = {
        "synthetic": syn_rating_dist,
        "real": real_rating_dist
    }
    
    # Calculate distribution similarity
    try:
        syn_dist_values = {k: v["percentage"] for k, v in syn_rating_dist["distribution"].items()}
        real_dist_values = {k: v["percentage"] for k, v in real_rating_dist["distribution"].items()}
        if syn_dist_values and real_dist_values:
            dist_comparison = compare_distributions(syn_dist_values, real_dist_values)
            report["comparison"]["rating_distribution"]["similarity"] = dist_comparison
    except Exception as e:
        report["comparison"]["rating_distribution"]["similarity"] = {"error": str(e)}
    
    # Semantic Similarity (if embedder available)
    if embedder and synthetic_comments and real_comments:
        try:
            syn_embeddings = embedder.get_embeddings(synthetic_comments)
            real_embeddings = embedder.get_embeddings(real_comments)
            
            semantic_sim = calculate_synthetic_vs_real_similarity(syn_embeddings, real_embeddings)
            report["comparison"]["semantic_similarity"] = semantic_sim
            
            # Diversity comparison
            syn_diversity = calculate_diversity_metrics(syn_embeddings)
            real_diversity = calculate_diversity_metrics(real_embeddings)
            
            report["comparison"]["diversity"] = {
                "synthetic": syn_diversity,
                "real": real_diversity
            }
        except Exception as e:
            report["comparison"]["semantic_similarity"] = {"error": str(e)}
    
    # Calculate overall comparison score
    try:
        semantic_sim = report["comparison"].get("semantic_similarity", {}).get("mean_similarity", 0.5)
        dist_sim = report["comparison"].get("rating_distribution", {}).get("similarity", {}).get("js_similarity", 0.5)
        diversity = report["comparison"].get("diversity", {}).get("synthetic", {}).get("diversity_score", 0.5)
        readability = report["comparison"]["readability"]["synthetic"].get("flesch_mean", 60)
        
        overall_score = calculate_overall_quality_score(
            diversity_score=diversity,
            readability_score=readability,
            synthetic_real_similarity=semantic_sim,
            distribution_similarity=dist_sim
        )
        
        report["comparison"]["overall_quality_score"] = overall_score
        report["comparison"]["quality_assessment"] = get_quality_assessment(overall_score)
    except Exception as e:
        report["comparison"]["overall_quality_score"] = {"error": str(e)}
    
    return report


def get_quality_assessment(score: float) -> str:
    """Get a textual assessment based on quality score."""
    if score >= 0.8:
        return "Excellent - Synthetic reviews are high quality and closely match real review characteristics"
    elif score >= 0.6:
        return "Good - Synthetic reviews are of acceptable quality with minor differences from real reviews"
    elif score >= 0.4:
        return "Fair - Synthetic reviews show noticeable differences from real reviews, consider adjusting generation parameters"
    elif score >= 0.2:
        return "Poor - Synthetic reviews significantly differ from real reviews, recommend reviewing prompts and model"
    else:
        return "Very Poor - Synthetic reviews do not resemble real reviews, major changes needed"


# ==============================================================================
# MODEL PERFORMANCE REPORT
# ==============================================================================

def generate_model_performance_report(
    performance_tracker: ModelPerformanceTracker,
    include_comparison: bool = True
) -> Dict[str, Any]:
    """
    Generate a model performance report from tracked data.
    
    Args:
        performance_tracker: ModelPerformanceTracker instance with recorded runs
        include_comparison: Whether to include model comparison summary
    
    Returns:
        Performance report dictionary
    """
    report = {
        "report_type": "model_performance_report",
        "generated_at": datetime.now().isoformat(),
        "runs": performance_tracker.get_all_runs(),
        "total_runs": len(performance_tracker.get_all_runs())
    }
    
    if include_comparison and report["total_runs"] > 1:
        report["comparison"] = performance_tracker.get_comparative_summary()
    
    # Add per-run quality assessment
    for run in report["runs"]:
        run["assessment"] = assess_model_run(run)
    
    return report


def assess_model_run(run: Dict) -> Dict[str, Any]:
    """Generate assessment for a single model run."""
    assessment = {
        "success_rate_grade": "A" if run["success_rate"] >= 95 else 
                             "B" if run["success_rate"] >= 85 else 
                             "C" if run["success_rate"] >= 70 else 
                             "D" if run["success_rate"] >= 50 else "F",
        "speed_grade": "A" if run["avg_generation_time"] <= 2 else 
                      "B" if run["avg_generation_time"] <= 5 else 
                      "C" if run["avg_generation_time"] <= 10 else 
                      "D" if run["avg_generation_time"] <= 20 else "F",
        "efficiency_grade": "A" if run["reviews_per_minute"] >= 10 else 
                           "B" if run["reviews_per_minute"] >= 5 else 
                           "C" if run["reviews_per_minute"] >= 2 else 
                           "D" if run["reviews_per_minute"] >= 1 else "F",
        "retry_analysis": {
            "similarity_retry_rate": round(run["similarity_retries"] / max(run["total_attempts"], 1), 2),
            "reality_check_retry_rate": round(run["reality_check_retries"] / max(run["total_attempts"], 1), 2),
            "sentiment_retry_rate": round(run["sentiment_alignment_retries"] / max(run["total_attempts"], 1), 2)
        }
    }
    
    # Overall grade
    grades = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
    avg_grade = (grades[assessment["success_rate_grade"]] + 
                 grades[assessment["speed_grade"]] + 
                 grades[assessment["efficiency_grade"]]) / 3
    
    assessment["overall_grade"] = "A" if avg_grade >= 3.5 else \
                                  "B" if avg_grade >= 2.5 else \
                                  "C" if avg_grade >= 1.5 else \
                                  "D" if avg_grade >= 0.5 else "F"
    
    return assessment


# ==============================================================================
# COMBINED SUMMARY REPORT
# ==============================================================================

def generate_summary_report(
    synthetic_reviews: List[Dict],
    real_reviews: List[Dict],
    performance_tracker: ModelPerformanceTracker,
    embedder=None,
    model_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Generate a combined summary report with all metrics.
    
    Args:
        synthetic_reviews: Generated reviews
        real_reviews: Real reviews for comparison
        performance_tracker: Performance tracking data
        embedder: Embedder for similarity calculations
        model_name: Model name
    
    Returns:
        Combined summary report
    """
    report = {
        "report_type": "summary_report",
        "generated_at": datetime.now().isoformat(),
        "model": model_name,
        "overview": {
            "synthetic_reviews_generated": len(synthetic_reviews),
            "real_reviews_analyzed": len(real_reviews)
        }
    }
    
    # Include quality report
    quality = generate_quality_report(synthetic_reviews, embedder, model_name)
    report["quality_metrics"] = quality.get("metrics", {})
    
    # Include comparison if real reviews available
    if real_reviews:
        comparison = generate_comparison_report(synthetic_reviews, real_reviews, embedder, model_name)
        report["comparison_metrics"] = comparison.get("comparison", {})
    
    # Include performance metrics
    runs = performance_tracker.get_all_runs()
    if runs:
        # Find the run for this model
        model_runs = [r for r in runs if model_name in f"{r['model_provider']}/{r['model_name']}"]
        if model_runs:
            report["performance_metrics"] = model_runs[-1]  # Latest run
    
    # Generate recommendations
    report["recommendations"] = generate_recommendations(report)
    
    return report


def generate_recommendations(report: Dict) -> List[str]:
    """Generate actionable recommendations based on report metrics."""
    recommendations = []
    
    # Check diversity
    diversity_score = report.get("quality_metrics", {}).get("diversity", {}).get("diversity_score", 1)
    if diversity_score < 0.5:
        recommendations.append(
            "Low diversity detected - Consider increasing temperature or adding more variation to prompts"
        )
    
    # Check readability
    flesch_mean = report.get("quality_metrics", {}).get("text_quality", {}).get("readability", {}).get("flesch_mean", 60)
    if flesch_mean < 30:
        recommendations.append(
            "Reviews may be too complex - Consider simplifying language in prompts"
        )
    elif flesch_mean > 80:
        recommendations.append(
            "Reviews may be too simple - Consider encouraging more detailed responses"
        )
    
    # Check review length
    mean_length = report.get("quality_metrics", {}).get("text_quality", {}).get("length_stats", {}).get("mean", 0)
    if mean_length < 50:
        recommendations.append(
            "Reviews are relatively short - Consider adjusting prompts to encourage longer, more detailed reviews"
        )
    
    # Check comparison metrics
    overall_score = report.get("comparison_metrics", {}).get("overall_quality_score", 0.5)
    if isinstance(overall_score, (int, float)) and overall_score < 0.5:
        recommendations.append(
            "Quality score is below average - Review the prompts and consider using more context from real reviews"
        )
    
    # Check performance
    perf = report.get("performance_metrics", {})
    if perf.get("success_rate", 100) < 80:
        recommendations.append(
            "Generation success rate is low - Check for API errors or adjust similarity threshold"
        )
    
    if perf.get("avg_generation_time", 0) > 10:
        recommendations.append(
            "Generation is slow - Consider using a faster model or reducing prompt complexity"
        )
    
    if not recommendations:
        recommendations.append("All metrics look good! No immediate improvements needed.")
    
    return recommendations


# ==============================================================================
# REPORT SAVING FUNCTIONS
# ==============================================================================

def save_report(report: Dict, filename: Optional[str] = None, output_dir: str = None) -> str:
    """
    Save a report to the reports directory.
    
    Args:
        report: Report dictionary to save
        filename: Optional custom filename (without extension)
        output_dir: Optional custom output directory
    
    Returns:
        Path to saved report
    """
    target_dir = ensure_reports_dir(output_dir)
    
    if filename is None:
        report_type = report.get("report_type", "report")
        model = report.get("model", "unknown").replace("/", "-").replace(":", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{model}_{timestamp}"
    
    # Save as JSON
    json_path = os.path.join(target_dir, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Also save as markdown for readability
    md_path = os.path.join(target_dir, f"{filename}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(format_report_as_markdown(report))
    
    print(f"Report saved to: {json_path}")
    print(f"Markdown version: {md_path}")
    
    return json_path


def format_report_as_markdown(report: Dict) -> str:
    """Format a report dictionary as readable markdown."""
    lines = []
    
    report_type = report.get("report_type", "Report").replace("_", " ").title()
    lines.append(f"# {report_type}")
    lines.append("")
    lines.append(f"**Generated:** {report.get('generated_at', 'N/A')}")
    lines.append(f"**Model:** {report.get('model', 'N/A')}")
    lines.append("")
    
    # Overview
    if "overview" in report:
        lines.append("## Overview")
        for key, value in report["overview"].items():
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        lines.append("")
    
    # Quality Metrics
    if "quality_metrics" in report or "metrics" in report:
        metrics = report.get("quality_metrics", report.get("metrics", {}))
        lines.append("## Quality Metrics")
        lines.append("")
        
        if "text_quality" in metrics:
            lines.append("### Text Quality")
            tq = metrics["text_quality"]
            if "length_stats" in tq:
                ls = tq["length_stats"]
                lines.append(f"- **Average Length:** {ls.get('mean', 0):.1f} words (std: {ls.get('std', 0):.1f})")
                lines.append(f"- **Length Range:** {ls.get('min', 0)} - {ls.get('max', 0)} words")
            if "vocabulary_diversity" in tq:
                vd = tq["vocabulary_diversity"]
                lines.append(f"- **Type-Token Ratio:** {vd.get('type_token_ratio', 0):.4f}")
                lines.append(f"- **Unique Words:** {vd.get('unique_words', 0)}")
            if "readability" in tq:
                rd = tq["readability"]
                lines.append(f"- **Flesch Reading Ease:** {rd.get('flesch_mean', 0):.1f} (std: {rd.get('flesch_std', 0):.1f})")
            lines.append("")
        
        if "diversity" in metrics:
            lines.append("### Diversity")
            div = metrics["diversity"]
            lines.append(f"- **Diversity Score:** {div.get('diversity_score', 0):.4f}")
            lines.append(f"- **Mean Pairwise Similarity:** {div.get('mean_pairwise_similarity', 0):.4f}")
            lines.append("")
        
        if "distributions" in metrics:
            lines.append("### Distributions")
            dist = metrics["distributions"]
            if "rating" in dist:
                lines.append("#### Rating Distribution")
                lines.append(f"- **Mean Rating:** {dist['rating'].get('mean', 0):.2f}")
                lines.append(f"- **Rating Std Dev:** {dist['rating'].get('std', 0):.2f}")
                if "distribution" in dist["rating"]:
                    lines.append("| Rating | Count | Percentage |")
                    lines.append("|--------|-------|------------|")
                    for rating, data in sorted(dist["rating"]["distribution"].items()):
                        lines.append(f"| {rating} | {data['count']} | {data['percentage']:.1f}% |")
                lines.append("")
    
    # Comparison Metrics
    if "comparison_metrics" in report or "comparison" in report:
        comparison = report.get("comparison_metrics", report.get("comparison", {}))
        lines.append("## Comparison with Real Reviews")
        lines.append("")
        
        if "semantic_similarity" in comparison:
            ss = comparison["semantic_similarity"]
            lines.append("### Semantic Similarity")
            lines.append(f"- **Mean Similarity:** {ss.get('mean_similarity', 0):.4f}")
            lines.append(f"- **Max Similarity:** {ss.get('max_similarity', 0):.4f}")
            lines.append(f"- **Min Similarity:** {ss.get('min_similarity', 0):.4f}")
            lines.append("")
        
        if "overall_quality_score" in comparison:
            score = comparison["overall_quality_score"]
            if isinstance(score, (int, float)):
                lines.append(f"### Overall Quality Score: **{score:.4f}**")
                lines.append(f"*{comparison.get('quality_assessment', '')}*")
                lines.append("")
    
    # Performance Metrics
    if "performance_metrics" in report:
        perf = report["performance_metrics"]
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append(f"- **Total Time:** {perf.get('total_time_seconds', 0):.1f} seconds")
        lines.append(f"- **Success Rate:** {perf.get('success_rate', 0):.1f}%")
        lines.append(f"- **Avg Generation Time:** {perf.get('avg_generation_time', 0):.2f} seconds")
        lines.append(f"- **Reviews per Minute:** {perf.get('reviews_per_minute', 0):.1f}")
        lines.append(f"- **Successful Generations:** {perf.get('successful_generations', 0)}")
        lines.append(f"- **Failed Generations:** {perf.get('failed_generations', 0)}")
        lines.append(f"- **Similarity Retries:** {perf.get('similarity_retries', 0)}")
        lines.append(f"- **Reality Check Retries:** {perf.get('reality_check_retries', 0)}")
        lines.append("")
        
        if "assessment" in perf:
            lines.append("### Assessment")
            assessment = perf["assessment"]
            lines.append(f"- **Overall Grade:** {assessment.get('overall_grade', 'N/A')}")
            lines.append(f"- **Success Rate Grade:** {assessment.get('success_rate_grade', 'N/A')}")
            lines.append(f"- **Speed Grade:** {assessment.get('speed_grade', 'N/A')}")
            lines.append(f"- **Efficiency Grade:** {assessment.get('efficiency_grade', 'N/A')}")
            lines.append("")
    
    # Recommendations
    if "recommendations" in report:
        lines.append("## Recommendations")
        lines.append("")
        for rec in report["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")
    
    # Model Performance Runs (for performance reports)
    if "runs" in report:
        lines.append("## Model Runs")
        lines.append("")
        for i, run in enumerate(report["runs"], 1):
            model_id = f"{run.get('model_provider', 'unknown')}/{run.get('model_name', 'unknown')}"
            lines.append(f"### Run {i}: {model_id}")
            lines.append(f"- **Duration:** {run.get('total_time_seconds', 0):.1f}s")
            lines.append(f"- **Success Rate:** {run.get('success_rate', 0):.1f}%")
            lines.append(f"- **Avg Time/Review:** {run.get('avg_generation_time', 0):.2f}s")
            lines.append(f"- **Reviews/Minute:** {run.get('reviews_per_minute', 0):.1f}")
            if "assessment" in run:
                lines.append(f"- **Grade:** {run['assessment'].get('overall_grade', 'N/A')}")
            lines.append("")
    
    # Comparison summary (for multi-model reports)
    if "comparison" in report and "models_compared" in report.get("comparison", {}):
        comp = report["comparison"]
        lines.append("## Model Comparison Summary")
        lines.append("")
        lines.append(f"**Models Compared:** {', '.join(comp.get('models_compared', []))}")
        lines.append("")
        
        if comp.get("best_success_rate", {}).get("model"):
            lines.append(f"- **Best Success Rate:** {comp['best_success_rate']['model']} ({comp['best_success_rate']['rate']:.1f}%)")
        if comp.get("fastest_model", {}).get("model"):
            lines.append(f"- **Fastest Model:** {comp['fastest_model']['model']} ({comp['fastest_model']['avg_time']:.2f}s avg)")
        if comp.get("most_efficient", {}).get("model"):
            lines.append(f"- **Most Efficient:** {comp['most_efficient']['model']} ({comp['most_efficient']['reviews_per_minute']:.1f} reviews/min)")
        lines.append("")
    
    lines.append("---")
    lines.append("*Report generated by Synthetic Data Generator*")
    
    return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def generate_all_reports(
    synthetic_reviews: List[Dict],
    real_reviews_path: str = "data/real_reviews.json",
    performance_tracker: Optional[ModelPerformanceTracker] = None,
    embedder=None,
    model_name: str = "unknown",
    output_dir: str = None
) -> Dict[str, str]:
    """
    Generate all report types and save them.
    
    Args:
        synthetic_reviews: List of generated reviews
        real_reviews_path: Path to real reviews for comparison
        performance_tracker: Performance metrics tracker
        embedder: Embedder for similarity calculations
        model_name: Name of the model
        output_dir: Custom output directory for reports
    
    Returns:
        Dictionary mapping report type to file path
    """
    real_reviews = load_real_reviews(real_reviews_path)
    
    saved_reports = {}
    
    # Quality Report
    quality_report = generate_quality_report(synthetic_reviews, embedder, model_name)
    saved_reports["quality"] = save_report(quality_report, output_dir=output_dir)
    
    # Comparison Report (if real reviews available)
    if real_reviews:
        comparison_report = generate_comparison_report(synthetic_reviews, real_reviews, embedder, model_name)
        saved_reports["comparison"] = save_report(comparison_report, output_dir=output_dir)
    
    # Performance Report (if tracker available)
    if performance_tracker and performance_tracker.get_all_runs():
        perf_report = generate_model_performance_report(performance_tracker)
        saved_reports["performance"] = save_report(perf_report, output_dir=output_dir)
    
    # Summary Report
    if performance_tracker is None:
        performance_tracker = ModelPerformanceTracker()
    
    summary_report = generate_summary_report(
        synthetic_reviews, real_reviews, performance_tracker, embedder, model_name
    )
    saved_reports["summary"] = save_report(summary_report, output_dir=output_dir)
    
    return saved_reports
