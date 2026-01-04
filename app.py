import json
import random
import os
import sys
import time
import argparse
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
try:
    from modules.graph import build_graph
    from modules.embedder import LocalEmbedder
    from modules.metrics import ModelPerformanceTracker
    from modules.reporting import (
        generate_all_reports,
        generate_summary_report,
        save_report,
        load_real_reviews
    )
except ImportError as e:
    print(f"DEBUG: Import Error Details: {e}")
    import traceback
    traceback.print_exc()
    # Fallback to allow app to start even if dependencies aren't fully installed yet (during setup)
    build_graph = None
    LocalEmbedder = None
    ModelPerformanceTracker = None
    generate_all_reports = None

DEFAULT_CONFIG_PATH = "config.json"
OUTPUT_DIR = "data/output"
REPORTS_DIR = "reports"
RESEARCH_DIR = "data/research"

def create_run_folder() -> str:
    """Create a timestamped folder for this run's reports."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(REPORTS_DIR, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

class Logger:
    """Logs to both console and file, overwriting the log file each run."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def load_config(config_input: str = None) -> dict:
    """
    Load configuration from a file path or JSON string.
    
    Args:
        config_input: Either a path to a JSON file or a JSON string.
                     If None, uses the default config path.
    
    Returns:
        Parsed configuration dictionary.
    """
    if config_input is None:
        config_input = DEFAULT_CONFIG_PATH
    
    # Try to parse as JSON string first
    try:
        return json.loads(config_input)
    except json.JSONDecodeError:
        pass
    
    # Try to load as file path
    if os.path.isfile(config_input):
        with open(config_input, "r") as f:
            return json.load(f)
    
    raise ValueError(f"Config input is neither valid JSON nor a valid file path: {config_input}")

def select_rating(distribution: dict):
    ratings = list(distribution.keys())
    # JSON keys are strings, convert to float for weights
    weights = [distribution[r] for r in ratings]
    return int(random.choices(ratings, weights=weights, k=1)[0])

def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def run_product_research(config: dict):
    """
    Run research for all products using the first available model.
    Saves reports to data/research.
    """
    if not config.get("options", {}).get("use_research", False):
        print("Skipping research phase (use_research=false)")
        return

    print("\n" + "="*60)
    print("STARTING PRODUCT RESEARCH PHASE")
    print("="*60 + "\n")

    os.makedirs(RESEARCH_DIR, exist_ok=True)
    
    # Use the first model for research
    if not config["models"]:
        print("No models configured, cannot perform research.")
        return
        
    research_model = config["models"][0]
    print(f"Using model for research: {research_model['provider']}/{research_model['name']}")

    try:
        from modules.research import build_research_graph
        research_graph = build_research_graph()
    except ImportError:
        print("Could not import research module. Skipping.")
        return

    products = config["products"]
    # Deduplicate products by name
    unique_products = {p["name"]: p for p in products}.values()

    for product in unique_products:
        product_name = product["name"]
        safe_name = sanitize_filename(product_name)
        file_path = os.path.join(RESEARCH_DIR, f"{safe_name}.md")
        
        print(f"Researching {product_name}...")
        try:
            research_res = research_graph.invoke({
                "product_name": product_name,
                "model_config": research_model,
                "messages": []
            })
            report = research_res.get("final_report", "No report generated.")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
                
            print(f"Saved research to {file_path}")
            
        except Exception as e:
            print(f"Research failed for {product_name}: {e}")

    print("\nResearch phase complete.\n")

def run_generation(config: dict, reports_dir: str = None) -> dict:
    """
    Run the synthetic review generation process.
    
    Args:
        config: Configuration dictionary.
        reports_dir: Directory to save reports (defaults to timestamped folder).
    
    Returns:
        Dictionary with model results and output files.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Create run-specific reports folder if not provided
    if reports_dir is None:
        reports_dir = create_run_folder()
    print(f"Reports will be saved to: {reports_dir}")

    products = config["products"]
    personas = config["personas"]
    rating_dist = config["rating_distribution"]
    models = config["models"]
    options = config["options"]
    samples = config["samples_number"]
    similarity_threshold = config.get("similarity_threshold", 0.8)
    
    if LocalEmbedder is None:
        raise RuntimeError("sentence-transformers not installed. Please run pip install -r requirements.txt")

    print("Initializing Embedder...")
    embedder = LocalEmbedder()
    graph = build_graph()
    
    # Run research phase first
    run_product_research(config)
    
    # Load research data
    product_research_map = {}
    if options.get("use_research", False):
        for product in products:
            safe_name = sanitize_filename(product["name"])
            file_path = os.path.join(RESEARCH_DIR, f"{safe_name}.md")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    product_research_map[product["name"]] = f.read()

    # Initialize performance tracker for all models
    performance_tracker = ModelPerformanceTracker() if ModelPerformanceTracker else None
    all_model_results = {}  # Store results per model for reporting

    # Loop over each model configuration
    for model_idx, model_config in enumerate(models):
        print(f"\n{'='*60}")
        print(f"MODEL {model_idx + 1}/{len(models)}: {model_config['provider']}/{model_config['name']}")
        print(f"{'='*60}\n")
        
        # ----------------------
        # Review Generation Phase
        # ----------------------
        
        # Store all embeddings for similarity check (reset per model)
        all_embeddings = None
        
        results = []
        
        # Start performance tracking for this model
        model_id = f"{model_config['provider']}/{model_config['name']}"
        if performance_tracker:
            performance_tracker.start_run(model_config)
        
        print(f"Generating {samples} reviews using {model_config['provider']}/{model_config['name']}...")
        
        # Track shuffled features per product for progressive shuffling across reviews
        product_shuffled_features = {}  # product_name -> shuffled_features list
        
        for i in range(samples):
            product = random.choice(products)
            persona = random.choice(personas)
            rating = select_rating(rating_dist)
            
            print(f"[{i+1}/{samples}] Generating review for {product['name']} by {persona['role']} (Rating: {rating})")
            
            # Get previously shuffled features for this product (if any)
            current_shuffled_features = product_shuffled_features.get(product["name"])
            
            initial_state = {
                "product": product,
                "persona": persona,
                "rating": rating,
                "options": options,
                "model_config": model_config,
                "generated_review": None,
                "previous_embeddings": all_embeddings,
                "embedder": embedder,
                "similarity_threshold": similarity_threshold,
                "feedback": None,
                "research_context": product_research_map.get(product["name"], ""),
                "shuffled_features": current_shuffled_features
            }
            
            # Track generation time
            gen_start_time = time.time()
            success = False
            skipped = False
            similarity_retries = 0
            reality_retries = 0
            sentiment_retries = 0
            
            try:
                # Recursion limit: scenario(1) + up to 5 attempts * 2 nodes each = 11 minimum
                # Set higher to be safe
                output_state = graph.invoke(initial_state, {"recursion_limit": 25})
                review_data = output_state["generated_review"]
                
                # Update shuffled features for this product (progressive shuffling)
                if output_state.get("shuffled_features"):
                    product_shuffled_features[product["name"]] = output_state["shuffled_features"]
                
                # Track retry counts from state
                similarity_retries = output_state.get("similarity_attempts", 0)
                reality_retries = output_state.get("reality_check_attempts", 0)
                sentiment_retries = output_state.get("sentiment_alignment_attempts", 0)
                skipped = output_state.get("review_skipped", False)
                
                # Update embeddings list
                comment = review_data.get("comment", "") if review_data else ""
                if comment:
                    new_embedding = embedder.get_embeddings([comment])
                    import torch
                    if all_embeddings is None:
                        all_embeddings = new_embedding
                    else:
                        import numpy as np
                        all_embeddings = np.concatenate((all_embeddings, new_embedding), axis=0)

                # Enrich/Combine data for final output
                if review_data and not skipped:
                    final_record = {
                        "title": review_data.get("title"),
                        "reviewer_role": persona["role"],
                        "comment": review_data.get("comment"),
                        "pros": review_data.get("pros"),
                        "cons": review_data.get("cons"),
                        "rating": review_data.get("rating", rating),
                        "product_name": product["name"]
                    }
                    results.append(final_record)
                    success = True
            except Exception as e:
                print(f"Error generating review {i+1}: {e}")
            
            # Record performance metrics
            gen_time = time.time() - gen_start_time
            if performance_tracker:
                performance_tracker.record_generation(
                    success=success,
                    generation_time=gen_time,
                    similarity_retries=similarity_retries,
                    reality_check_retries=reality_retries,
                    sentiment_retries=sentiment_retries,
                    skipped=skipped
                )

        # Create a safe model name for the filename
        safe_model_name = model_config['name'].replace(':', '-').replace('/', '-')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"reviews_{safe_model_name}_{timestamp}.json")
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"Done! Saved {len(results)} reviews to {output_file}")
        
        # End performance tracking and store results for this model
        if performance_tracker:
            run_summary = performance_tracker.end_run()
            print(f"\n--- Performance Summary for {model_id} ---")
            print(f"Success Rate: {run_summary.get('success_rate', 0):.1f}%")
            print(f"Avg Generation Time: {run_summary.get('avg_generation_time', 0):.2f}s")
            print(f"Reviews/Minute: {run_summary.get('reviews_per_minute', 0):.1f}")
            print(f"Total Time: {run_summary.get('total_time_seconds', 0):.1f}s")
        
        # Store results for reporting
        all_model_results[model_id] = {
            "reviews": results,
            "output_file": output_file
        }
        
    # Generate comprehensive reports after all models have run
    print(f"\n{'='*60}")
    print("GENERATING QUALITY REPORTS")
    print(f"{'='*60}\n")
    
    if generate_all_reports and performance_tracker:
        for model_id, model_data in all_model_results.items():
            print(f"Generating reports for {model_id}...")
            try:
                saved_reports = generate_all_reports(
                    synthetic_reviews=model_data["reviews"],
                    real_reviews_path="data/real_reviews.json",
                    performance_tracker=performance_tracker,
                    embedder=embedder,
                    model_name=model_id.replace("/", "-"),
                    output_dir=reports_dir
                )
                print(f"Reports saved for {model_id}:")
                for report_type, path in saved_reports.items():
                    print(f"  - {report_type}: {path}")
            except Exception as e:
                print(f"Error generating reports for {model_id}: {e}")
    else:
        print("Reporting module not available - skipping report generation")
    
    return all_model_results

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Review Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                           # Use default config.json
  python app.py --config myconfig.json    # Use custom config file
  python app.py --config '{"products": [...], "personas": [...], ...}'  # Use JSON string
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config JSON file or a JSON string with configuration"
    )
    args = parser.parse_args()
    
    # Create run-specific folder for reports and logs
    run_folder = create_run_folder()
    log_file = os.path.join(run_folder, "run.log")
    
    # Set up logging to file
    logger = Logger(log_file)
    sys.stdout = logger
    
    try:
        config = load_config(args.config)
        all_model_results = run_generation(config, reports_dir=run_folder)
        return all_model_results
    
    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()
        print(f"Run complete! All outputs saved to: {run_folder}")

if __name__ == "__main__":
    main()
