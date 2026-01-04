"""
Test script for the research module.
Runs the research graph for a product and prints the final report.
"""
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from modules.research import build_research_graph

OUTPUT_DIR = "data/research"

def save_report_as_markdown(product_name: str, result: dict):
    """Save the research report as a markdown file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    safe_name = product_name.replace(" ", "_").lower()
    filename = f"{OUTPUT_DIR}/research_{safe_name}.md"
    
    official_urls = result.get("official_urls", [])
    official_features = result.get("official_features", "None")
    review_urls = result.get("review_urls", [])
    reviews_summary = result.get("reviews_summary", "None")
    final_report = result.get("final_report", "No report generated")
    
    md_content = f"""# Research Report: {product_name}

*Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

---

## Official URLs Found

{chr(10).join(f"- {url}" for url in official_urls) if official_urls else "None found"}

---

## Official Features

{official_features}

---

## Review URLs Found

{chr(10).join(f"- {url}" for url in review_urls) if review_urls else "None found"}

---

## Reviews Summary

{reviews_summary}

---

## Final Report

{final_report}
"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"\n📄 Report saved to: {filename}")
    return filename

def test_research(product_name: str = "Easygenerator"):
    print(f"=" * 60)
    print(f"Research Report for: {product_name}")
    print(f"=" * 60)
    
    # Build and run the research graph
    graph = build_research_graph()
    
    initial_state = {
        "messages": [],
        "product_name": product_name,
        "official_urls": [],
        "official_features": "",
        "review_urls": [],
        "reviews_summary": "",
        "final_report": ""
    }
    
    print("\nRunning research pipeline...")
    result = graph.invoke(initial_state)
    
    # Print results
    print("\n" + "-" * 60)
    print("OFFICIAL URLs FOUND:")
    print("-" * 60)
    for url in result.get("official_urls", []):
        print(f"  • {url}")
    
    print("\n" + "-" * 60)
    print("OFFICIAL FEATURES:")
    print("-" * 60)
    print(result.get("official_features", "None"))
    
    print("\n" + "-" * 60)
    print("REVIEW URLs FOUND:")
    print("-" * 60)
    for url in result.get("review_urls", []):
        print(f"  • {url}")
    
    print("\n" + "-" * 60)
    print("REVIEWS SUMMARY:")
    print("-" * 60)
    print(result.get("reviews_summary", "None"))
    
    print("\n" + "=" * 60)
    print("FINAL RESEARCH REPORT:")
    print("=" * 60)
    print(result.get("final_report", "No report generated"))
    
    # Save as markdown
    save_report_as_markdown(product_name, result)
    
    return result

if __name__ == "__main__":
    # You can change the product name here or pass it as command line argument
    product = sys.argv[1] if len(sys.argv) > 1 else "Easygenerator"
    test_research(product)
