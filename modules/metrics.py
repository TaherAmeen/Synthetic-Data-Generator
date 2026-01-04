"""
Metrics Module for Synthetic Review Quality Assessment

This module provides comprehensive metrics for evaluating synthetic review quality:
1. Text quality metrics (length, vocabulary diversity, readability)
2. Similarity metrics (synthetic vs. real, inter-review diversity)
3. Distribution metrics (rating, sentiment, persona coverage)
4. Model performance metrics (generation time, success rate)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re
import math
from datetime import datetime


# ==============================================================================
# TEXT QUALITY METRICS
# ==============================================================================

def calculate_text_length_stats(texts: List[str]) -> Dict[str, float]:
    """Calculate word count statistics for a list of texts."""
    if not texts:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    
    word_counts = [len(text.split()) for text in texts if text]
    if not word_counts:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    
    return {
        "mean": float(np.mean(word_counts)),
        "std": float(np.std(word_counts)),
        "min": int(min(word_counts)),
        "max": int(max(word_counts)),
        "median": float(np.median(word_counts))
    }


def calculate_vocabulary_diversity(texts: List[str]) -> Dict[str, float]:
    """
    Calculate vocabulary diversity metrics.
    
    Returns:
        - type_token_ratio: Unique words / Total words (higher = more diverse)
        - unique_words: Count of unique words
        - total_words: Total word count
    """
    if not texts:
        return {"type_token_ratio": 0, "unique_words": 0, "total_words": 0}
    
    all_words = []
    for text in texts:
        if text:
            # Simple tokenization: lowercase, split, remove punctuation
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            all_words.extend(words)
    
    if not all_words:
        return {"type_token_ratio": 0, "unique_words": 0, "total_words": 0}
    
    unique_words = set(all_words)
    ttr = len(unique_words) / len(all_words)
    
    return {
        "type_token_ratio": round(ttr, 4),
        "unique_words": len(unique_words),
        "total_words": len(all_words)
    }


def calculate_flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.
    Higher scores indicate easier reading.
    
    Score interpretation:
    - 90-100: Very easy (5th grade)
    - 80-89: Easy (6th grade)
    - 70-79: Fairly easy (7th grade)
    - 60-69: Standard (8th-9th grade)
    - 50-59: Fairly difficult (10th-12th grade)
    - 30-49: Difficult (College)
    - 0-29: Very difficult (Professional)
    """
    if not text:
        return 0.0
    
    # Count sentences (simple approximation)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    if sentence_count == 0:
        sentence_count = 1
    
    # Count words
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    word_count = len(words)
    if word_count == 0:
        return 0.0
    
    # Count syllables (approximation)
    def count_syllables(word):
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        # Handle silent 'e'
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
    
    syllable_count = sum(count_syllables(word) for word in words)
    
    # Flesch formula
    score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
    return round(max(0, min(100, score)), 2)


def calculate_readability_stats(texts: List[str]) -> Dict[str, float]:
    """Calculate readability statistics for a list of texts."""
    if not texts:
        return {"flesch_mean": 0, "flesch_std": 0, "flesch_min": 0, "flesch_max": 0}
    
    scores = [calculate_flesch_reading_ease(text) for text in texts if text]
    if not scores:
        return {"flesch_mean": 0, "flesch_std": 0, "flesch_min": 0, "flesch_max": 0}
    
    return {
        "flesch_mean": round(float(np.mean(scores)), 2),
        "flesch_std": round(float(np.std(scores)), 2),
        "flesch_min": round(min(scores), 2),
        "flesch_max": round(max(scores), 2)
    }


# ==============================================================================
# SIMILARITY & DIVERSITY METRICS
# ==============================================================================

def calculate_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Calculate pairwise cosine similarity matrix."""
    if embeddings is None or len(embeddings) == 0:
        return np.array([])
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms
    
    # Cosine similarity matrix
    return np.dot(normalized, normalized.T)


def calculate_diversity_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Calculate diversity metrics from embeddings.
    
    Returns:
        - mean_pairwise_similarity: Average similarity between all pairs (lower = more diverse)
        - min_pairwise_similarity: Minimum similarity (lower = more diverse extremes)
        - max_pairwise_similarity: Maximum similarity (should be < 1 for good diversity)
        - diversity_score: 1 - mean_similarity (higher = more diverse)
    """
    if embeddings is None or len(embeddings) < 2:
        return {
            "mean_pairwise_similarity": 0,
            "min_pairwise_similarity": 0,
            "max_pairwise_similarity": 0,
            "diversity_score": 1.0
        }
    
    sim_matrix = calculate_cosine_similarity_matrix(embeddings)
    
    # Get upper triangle (excluding diagonal)
    upper_tri_indices = np.triu_indices(len(sim_matrix), k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]
    
    if len(pairwise_sims) == 0:
        return {
            "mean_pairwise_similarity": 0,
            "min_pairwise_similarity": 0,
            "max_pairwise_similarity": 0,
            "diversity_score": 1.0
        }
    
    mean_sim = float(np.mean(pairwise_sims))
    
    return {
        "mean_pairwise_similarity": round(mean_sim, 4),
        "min_pairwise_similarity": round(float(np.min(pairwise_sims)), 4),
        "max_pairwise_similarity": round(float(np.max(pairwise_sims)), 4),
        "diversity_score": round(1 - mean_sim, 4)
    }


def calculate_synthetic_vs_real_similarity(
    synthetic_embeddings: np.ndarray,
    real_embeddings: np.ndarray
) -> Dict[str, float]:
    """
    Calculate similarity metrics between synthetic and real reviews.
    
    Returns:
        - mean_similarity: Average similarity to nearest real review
        - max_similarity: Maximum similarity to any real review
        - min_similarity: Minimum similarity to any real review
        - closest_match_distribution: Distribution of closest matches
    """
    if synthetic_embeddings is None or real_embeddings is None:
        return {"mean_similarity": 0, "max_similarity": 0, "min_similarity": 0}
    
    if len(synthetic_embeddings) == 0 or len(real_embeddings) == 0:
        return {"mean_similarity": 0, "max_similarity": 0, "min_similarity": 0}
    
    # Normalize embeddings
    syn_norms = np.linalg.norm(synthetic_embeddings, axis=1, keepdims=True)
    syn_norms[syn_norms == 0] = 1
    syn_normalized = synthetic_embeddings / syn_norms
    
    real_norms = np.linalg.norm(real_embeddings, axis=1, keepdims=True)
    real_norms[real_norms == 0] = 1
    real_normalized = real_embeddings / real_norms
    
    # Cross-similarity matrix (synthetic x real)
    cross_sim = np.dot(syn_normalized, real_normalized.T)
    
    # For each synthetic, find max similarity to any real
    max_sims_per_synthetic = np.max(cross_sim, axis=1)
    
    return {
        "mean_similarity": round(float(np.mean(max_sims_per_synthetic)), 4),
        "max_similarity": round(float(np.max(max_sims_per_synthetic)), 4),
        "min_similarity": round(float(np.min(max_sims_per_synthetic)), 4),
        "std_similarity": round(float(np.std(max_sims_per_synthetic)), 4)
    }


# ==============================================================================
# DISTRIBUTION METRICS
# ==============================================================================

def calculate_rating_distribution(reviews: List[Dict]) -> Dict[str, Any]:
    """Calculate rating distribution metrics."""
    if not reviews:
        return {"distribution": {}, "mean": 0, "std": 0}
    
    ratings = [r.get("rating", 0) for r in reviews if r.get("rating") is not None]
    if not ratings:
        return {"distribution": {}, "mean": 0, "std": 0}
    
    counter = Counter(ratings)
    total = len(ratings)
    distribution = {str(k): {"count": v, "percentage": round(v/total*100, 2)} 
                   for k, v in sorted(counter.items())}
    
    return {
        "distribution": distribution,
        "mean": round(float(np.mean(ratings)), 2),
        "std": round(float(np.std(ratings)), 2),
        "median": float(np.median(ratings))
    }


def calculate_persona_distribution(reviews: List[Dict]) -> Dict[str, Any]:
    """Calculate persona/role distribution."""
    if not reviews:
        return {"distribution": {}}
    
    roles = [r.get("reviewer_role", "Unknown") for r in reviews]
    counter = Counter(roles)
    total = len(roles)
    
    distribution = {k: {"count": v, "percentage": round(v/total*100, 2)} 
                   for k, v in sorted(counter.items())}
    
    return {"distribution": distribution}


def compare_distributions(
    synthetic_dist: Dict[str, float],
    real_dist: Dict[str, float]
) -> Dict[str, float]:
    """
    Compare two distributions using statistical measures.
    
    Returns:
        - kl_divergence: KL divergence (lower = more similar)
        - js_divergence: Jensen-Shannon divergence (lower = more similar, bounded [0,1])
        - chi_square: Chi-square statistic
    """
    # Get all keys
    all_keys = set(synthetic_dist.keys()) | set(real_dist.keys())
    
    # Convert to probability arrays (with smoothing to avoid zeros)
    epsilon = 1e-10
    p = np.array([synthetic_dist.get(k, 0) + epsilon for k in sorted(all_keys)])
    q = np.array([real_dist.get(k, 0) + epsilon for k in sorted(all_keys)])
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # KL Divergence
    kl_div = float(np.sum(p * np.log(p / q)))
    
    # Jensen-Shannon Divergence
    m = (p + q) / 2
    js_div = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    
    return {
        "kl_divergence": round(kl_div, 4),
        "js_divergence": round(js_div, 4),
        "js_similarity": round(1 - js_div, 4)  # Higher = more similar
    }


# ==============================================================================
# MODEL PERFORMANCE METRICS
# ==============================================================================

class ModelPerformanceTracker:
    """Track performance metrics for model generation runs."""
    
    def __init__(self):
        self.runs = []
        self.current_run = None
    
    def start_run(self, model_config: Dict):
        """Start tracking a new model run."""
        self.current_run = {
            "model_provider": model_config.get("provider", "unknown"),
            "model_name": model_config.get("name", "unknown"),
            "start_time": datetime.now(),
            "end_time": None,
            "total_attempts": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "similarity_retries": 0,
            "reality_check_retries": 0,
            "sentiment_alignment_retries": 0,
            "skipped_reviews": 0,
            "generation_times": [],  # Per-review generation times
            "quality_scores": []  # Per-review quality scores
        }
    
    def record_generation(
        self,
        success: bool,
        generation_time: float,
        similarity_retries: int = 0,
        reality_check_retries: int = 0,
        sentiment_retries: int = 0,
        skipped: bool = False,
        quality_score: Optional[float] = None
    ):
        """Record metrics for a single generation attempt."""
        if self.current_run is None:
            return
        
        self.current_run["total_attempts"] += 1
        
        if success:
            self.current_run["successful_generations"] += 1
            self.current_run["generation_times"].append(generation_time)
            if quality_score is not None:
                self.current_run["quality_scores"].append(quality_score)
        else:
            self.current_run["failed_generations"] += 1
        
        if skipped:
            self.current_run["skipped_reviews"] += 1
        
        self.current_run["similarity_retries"] += similarity_retries
        self.current_run["reality_check_retries"] += reality_check_retries
        self.current_run["sentiment_alignment_retries"] += sentiment_retries
    
    def end_run(self) -> Dict[str, Any]:
        """End current run and calculate summary metrics."""
        if self.current_run is None:
            return {}
        
        self.current_run["end_time"] = datetime.now()
        
        # Calculate derived metrics
        run = self.current_run
        total_time = (run["end_time"] - run["start_time"]).total_seconds()
        
        gen_times = run["generation_times"]
        quality_scores = run["quality_scores"]
        
        summary = {
            **run,
            "total_time_seconds": round(total_time, 2),
            "success_rate": round(run["successful_generations"] / max(run["total_attempts"], 1) * 100, 2),
            "avg_generation_time": round(np.mean(gen_times), 2) if gen_times else 0,
            "std_generation_time": round(np.std(gen_times), 2) if gen_times else 0,
            "min_generation_time": round(min(gen_times), 2) if gen_times else 0,
            "max_generation_time": round(max(gen_times), 2) if gen_times else 0,
            "avg_quality_score": round(np.mean(quality_scores), 4) if quality_scores else None,
            "reviews_per_minute": round(run["successful_generations"] / (total_time / 60), 2) if total_time > 0 else 0
        }
        
        # Convert datetime to string for JSON serialization
        summary["start_time"] = run["start_time"].isoformat()
        summary["end_time"] = run["end_time"].isoformat()
        
        self.runs.append(summary)
        self.current_run = None
        
        return summary
    
    def get_all_runs(self) -> List[Dict]:
        """Get all recorded runs."""
        return self.runs
    
    def get_comparative_summary(self) -> Dict[str, Any]:
        """Get a comparative summary of all model runs."""
        if not self.runs:
            return {}
        
        summary = {
            "total_runs": len(self.runs),
            "models_compared": [],
            "best_success_rate": {"model": None, "rate": 0},
            "fastest_model": {"model": None, "avg_time": float('inf')},
            "best_quality": {"model": None, "score": 0},
            "most_efficient": {"model": None, "reviews_per_minute": 0}
        }
        
        for run in self.runs:
            model_id = f"{run['model_provider']}/{run['model_name']}"
            summary["models_compared"].append(model_id)
            
            if run["success_rate"] > summary["best_success_rate"]["rate"]:
                summary["best_success_rate"] = {"model": model_id, "rate": run["success_rate"]}
            
            if run["avg_generation_time"] < summary["fastest_model"]["avg_time"] and run["avg_generation_time"] > 0:
                summary["fastest_model"] = {"model": model_id, "avg_time": run["avg_generation_time"]}
            
            if run["avg_quality_score"] and run["avg_quality_score"] > summary["best_quality"]["score"]:
                summary["best_quality"] = {"model": model_id, "score": run["avg_quality_score"]}
            
            if run["reviews_per_minute"] > summary["most_efficient"]["reviews_per_minute"]:
                summary["most_efficient"] = {"model": model_id, "reviews_per_minute": run["reviews_per_minute"]}
        
        return summary


# ==============================================================================
# COMBINED QUALITY SCORE
# ==============================================================================

def calculate_overall_quality_score(
    diversity_score: float,
    readability_score: float,
    synthetic_real_similarity: float,
    distribution_similarity: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate an overall quality score combining multiple metrics.
    
    Args:
        diversity_score: 0-1, higher is better (from diversity metrics)
        readability_score: 0-100, normalized to 0-1 (target range: 50-70)
        synthetic_real_similarity: 0-1, moderate is good (around 0.5-0.7)
        distribution_similarity: 0-1, higher is better
        weights: Optional custom weights for each component
    
    Returns:
        Overall quality score between 0 and 1
    """
    if weights is None:
        weights = {
            "diversity": 0.25,
            "readability": 0.20,
            "realism": 0.30,
            "distribution": 0.25
        }
    
    # Normalize readability (target: 50-70 is optimal)
    # Score highest at 60, decrease towards 0 or 100
    readability_normalized = 1 - abs(readability_score - 60) / 60
    readability_normalized = max(0, min(1, readability_normalized))
    
    # For realism, moderate similarity is ideal (not too similar = copied, not too different = unrealistic)
    # Optimal range: 0.4-0.7
    if synthetic_real_similarity < 0.4:
        realism_score = synthetic_real_similarity / 0.4
    elif synthetic_real_similarity > 0.7:
        realism_score = 1 - (synthetic_real_similarity - 0.7) / 0.3
    else:
        realism_score = 1.0
    realism_score = max(0, min(1, realism_score))
    
    # Calculate weighted score
    overall = (
        weights["diversity"] * diversity_score +
        weights["readability"] * readability_normalized +
        weights["realism"] * realism_score +
        weights["distribution"] * distribution_similarity
    )
    
    return round(overall, 4)
