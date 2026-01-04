"""
Test script for the reporting module.
Run this to verify reporting works correctly.
"""

import json
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test basic metrics without embedder
def test_basic_metrics():
    """Test basic text metrics without embeddings."""
    from modules.metrics import (
        calculate_text_length_stats,
        calculate_vocabulary_diversity,
        calculate_readability_stats,
        calculate_rating_distribution,
        calculate_persona_distribution,
        calculate_flesch_reading_ease
    )
    
    print("=" * 60)
    print("Testing Basic Metrics")
    print("=" * 60)
    
    # Load synthetic reviews
    synthetic_file = "data/output/reviews_20260103_235727.json"
    with open(synthetic_file, "r") as f:
        synthetic_reviews = json.load(f)
    
    # Load real reviews
    with open("data/real_reviews.json", "r") as f:
        real_reviews = json.load(f)
    
    print(f"\nLoaded {len(synthetic_reviews)} synthetic reviews")
    print(f"Loaded {len(real_reviews)} real reviews")
    
    # Extract texts
    def extract_texts(reviews):
        texts = []
        for r in reviews:
            parts = [r.get("title", ""), r.get("comment", ""), r.get("pros", ""), r.get("cons", "")]
            texts.append(" ".join([p for p in parts if p]))
        return texts
    
    synthetic_texts = extract_texts(synthetic_reviews)
    real_texts = extract_texts(real_reviews)
    
    # Test text length stats
    print("\n--- Text Length Stats ---")
    syn_length = calculate_text_length_stats(synthetic_texts)
    real_length = calculate_text_length_stats(real_texts)
    print(f"Synthetic - Mean: {syn_length['mean']:.1f}, Std: {syn_length['std']:.1f}")
    print(f"Real      - Mean: {real_length['mean']:.1f}, Std: {real_length['std']:.1f}")
    
    # Test vocabulary diversity
    print("\n--- Vocabulary Diversity ---")
    syn_vocab = calculate_vocabulary_diversity(synthetic_texts)
    real_vocab = calculate_vocabulary_diversity(real_texts)
    print(f"Synthetic - TTR: {syn_vocab['type_token_ratio']:.4f}, Unique: {syn_vocab['unique_words']}")
    print(f"Real      - TTR: {real_vocab['type_token_ratio']:.4f}, Unique: {real_vocab['unique_words']}")
    
    # Test readability
    print("\n--- Readability (Flesch Reading Ease) ---")
    syn_read = calculate_readability_stats(synthetic_texts)
    real_read = calculate_readability_stats(real_texts)
    print(f"Synthetic - Mean: {syn_read['flesch_mean']:.1f}, Std: {syn_read['flesch_std']:.1f}")
    print(f"Real      - Mean: {real_read['flesch_mean']:.1f}, Std: {real_read['flesch_std']:.1f}")
    
    # Test rating distribution
    print("\n--- Rating Distribution ---")
    syn_ratings = calculate_rating_distribution(synthetic_reviews)
    real_ratings = calculate_rating_distribution(real_reviews)
    print(f"Synthetic - Mean: {syn_ratings['mean']:.2f}, Std: {syn_ratings['std']:.2f}")
    print(f"Real      - Mean: {real_ratings['mean']:.2f}, Std: {real_ratings['std']:.2f}")
    print(f"\nSynthetic distribution: {syn_ratings['distribution']}")
    
    # Test persona distribution
    print("\n--- Persona Distribution ---")
    syn_personas = calculate_persona_distribution(synthetic_reviews)
    print(f"Synthetic personas: {syn_personas['distribution']}")
    
    print("\n✅ Basic metrics test passed!")
    return True


def test_report_generation_no_embedder():
    """Test report generation without embeddings."""
    from modules.reporting import (
        generate_quality_report,
        generate_comparison_report,
        save_report,
        format_report_as_markdown
    )
    
    print("\n" + "=" * 60)
    print("Testing Report Generation (without embeddings)")
    print("=" * 60)
    
    # Load reviews
    with open("data/output/reviews_20260103_235727.json", "r") as f:
        synthetic_reviews = json.load(f)
    
    with open("data/real_reviews.json", "r") as f:
        real_reviews = json.load(f)
    
    # Generate quality report (without embedder)
    print("\nGenerating quality report...")
    quality_report = generate_quality_report(
        synthetic_reviews=synthetic_reviews,
        embedder=None,  # No embedder for basic test
        model_name="test-model"
    )
    
    print(f"Quality report generated with {len(quality_report['metrics'])} metric categories")
    
    # Generate comparison report (without embedder)
    print("\nGenerating comparison report...")
    comparison_report = generate_comparison_report(
        synthetic_reviews=synthetic_reviews,
        real_reviews=real_reviews,
        embedder=None,
        model_name="test-model"
    )
    
    print(f"Comparison report generated with {len(comparison_report['comparison'])} comparisons")
    
    # Save reports
    print("\nSaving reports...")
    quality_path = save_report(quality_report, "test_quality_report")
    comparison_path = save_report(comparison_report, "test_comparison_report")
    
    print(f"Quality report saved to: {quality_path}")
    print(f"Comparison report saved to: {comparison_path}")
    
    # Print markdown preview
    print("\n--- Markdown Preview (Quality Report) ---")
    md_content = format_report_as_markdown(quality_report)
    print(md_content[:1500] + "..." if len(md_content) > 1500 else md_content)
    
    print("\n✅ Report generation test passed!")
    return True


def test_performance_tracker():
    """Test the performance tracker."""
    from modules.metrics import ModelPerformanceTracker
    import time
    
    print("\n" + "=" * 60)
    print("Testing Performance Tracker")
    print("=" * 60)
    
    tracker = ModelPerformanceTracker()
    
    # Simulate a model run
    tracker.start_run({"provider": "openai", "name": "gpt-4o-mini"})
    
    # Simulate some generations
    for i in range(5):
        time.sleep(0.1)  # Simulate generation time
        tracker.record_generation(
            success=True if i < 4 else False,  # 4 success, 1 fail
            generation_time=0.5 + (i * 0.1),
            similarity_retries=i % 2,
            reality_check_retries=0,
            sentiment_retries=0
        )
    
    # End run
    summary = tracker.end_run()
    
    print(f"\nRun Summary:")
    print(f"  Model: {summary['model_provider']}/{summary['model_name']}")
    print(f"  Total attempts: {summary['total_attempts']}")
    print(f"  Success rate: {summary['success_rate']}%")
    print(f"  Avg generation time: {summary['avg_generation_time']:.2f}s")
    print(f"  Reviews/minute: {summary['reviews_per_minute']:.1f}")
    
    # Simulate another model
    tracker.start_run({"provider": "ollama", "name": "llama3"})
    for i in range(3):
        tracker.record_generation(
            success=True,
            generation_time=1.0 + (i * 0.2),
            similarity_retries=0,
            reality_check_retries=0,
            sentiment_retries=0
        )
    summary2 = tracker.end_run()
    
    # Get comparison
    comparison = tracker.get_comparative_summary()
    print(f"\nComparison Summary:")
    print(f"  Total runs: {comparison['total_runs']}")
    print(f"  Best success rate: {comparison['best_success_rate']}")
    print(f"  Fastest model: {comparison['fastest_model']}")
    
    print("\n✅ Performance tracker test passed!")
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATOR - REPORTING MODULE TEST")
    print("=" * 60 + "\n")
    
    try:
        test_basic_metrics()
        test_report_generation_no_embedder()
        test_performance_tracker()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60 + "\n")
        
        print("Reports have been saved to the reports/ directory.")
        print("You can find both JSON and Markdown versions of each report.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
