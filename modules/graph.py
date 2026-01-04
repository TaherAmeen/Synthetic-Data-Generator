from typing import TypedDict, Optional, List, Any
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import random
from modules.prompts import get_review_prompt, get_scenario_prompt
from modules.embedder import LocalEmbedder
from modules.QA import (
    reality_check, 
    sentiment_alignment_check,
    check_semantic_similarity, 
    MAX_SIMILARITY_ATTEMPTS, 
    MAX_REALITY_CHECK_ATTEMPTS,
    MAX_SENTIMENT_ALIGNMENT_ATTEMPTS
)

load_dotenv()

class ReviewState(TypedDict):
    product: dict
    persona: dict
    rating: int
    options: dict
    model_config: dict
    generated_review: Optional[dict]
    previous_embeddings: Optional[Any] # Tensor or list of embeddings
    embedder: Optional[LocalEmbedder]
    similarity_threshold: float
    feedback: Optional[str]
    research_context: Optional[str]
    scenario: Optional[str]  # The usage scenario generated for the persona
    # Retry tracking for similarity check
    similarity_attempts: int
    best_review: Optional[dict]  # Best (least similar) review so far
    best_similarity: float  # Lowest similarity score so far
    # Retry tracking for reality check
    reality_check_attempts: int
    reality_check_passed: bool
    review_skipped: bool  # True if review was skipped due to failed QA checks
    # Retry tracking for sentiment alignment check
    sentiment_alignment_attempts: int
    sentiment_alignment_passed: bool
    predicted_ratings_history: List[int]  # Track all predicted ratings for mode calculation

def get_llm(model_config: dict, temperature: float = 0.7, top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, max_tokens: int = None):
    provider = model_config.get("provider", "openai")
    model_name = model_config.get("name", "gpt-4o-mini")
    
    if provider == "openai":
        return ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens
        )
    elif provider == "ollama":
        return ChatOllama(
            model=model_name, 
            temperature=temperature,
            top_p=top_p,
            num_predict=max_tokens
        )
    elif provider == "mistral":
        return ChatMistralAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def extract_features_from_research(research_context: str, model_config: dict) -> List[str]:
    """Extract a list of features from the research context using LLM."""
    from modules.prompts import FEATURE_EXTRACTION_PROMPT
    import json
    
    llm = get_llm(model_config, temperature=0.3)  # Low temp for consistent extraction
    prompt = FEATURE_EXTRACTION_PROMPT.format(research_context=research_context)
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Handle markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                if "[" in part and "]" in part:
                    content = part.replace("json", "").strip()
                    break
        
        features = json.loads(content)
        if isinstance(features, list):
            return features
    except Exception as e:
        print(f"DEBUG: Feature extraction failed: {e}")
    
    return []

def generate_scenario_node(state: ReviewState):
    """Generate a usage scenario for the persona based on product features."""
    model_config = state["model_config"]
    research_context = state.get("research_context", "")
    
    if not research_context or research_context == "No research available.":
        # No research available, skip scenario generation
        return {"scenario": "General product usage."}
    
    # Extract features from research
    features = extract_features_from_research(research_context, model_config)
    
    # Randomly select 1 to all features (weighted towards fewer features)
    if features:
        # Use weighted random to prefer fewer features but allow up to all
        max_features = len(features)
        # Weights decrease as number increases (1 feature most likely, all features least likely)
        weights = [1.0 / (i + 1) for i in range(max_features)]
        num_features = random.choices(range(1, max_features + 1), weights=weights)[0]
        selected_features = random.sample(features, num_features)
        assigned_features = "\n".join(f"- {f}" for f in selected_features)
        all_features = "\n".join(f"- {f}" for f in features)
        print(f"DEBUG: Assigned {num_features}/{max_features} features: {selected_features}")
    else:
        assigned_features = "- General product features"
        all_features = "- General product features"
    
    # Randomize parameters for variety
    temperature = random.uniform(0.7, 1.0)
    top_p = random.uniform(0.85, 1.0)
    
    llm = get_llm(model_config, temperature=temperature, top_p=top_p)
    
    scenario_prompt = get_scenario_prompt().format(
        role=state["persona"]["role"],
        description=state["persona"]["description"],
        product_name=state["product"]["name"],
        product_type=state["product"]["type"],
        research_context=research_context,
        assigned_features=assigned_features,
        all_features=all_features
    )
    
    response = llm.invoke(scenario_prompt)
    scenario = response.content.strip()
    
    return {"scenario": scenario}

def generate_review_node(state: ReviewState):
    model_config = state["model_config"]
    
    # Randomize parameters
    temperature = random.uniform(0.5, 0.9)
    top_p = random.uniform(0.8, 1.0)
    frequency_penalty = random.uniform(0.0, 0.5)
    presence_penalty = random.uniform(0.0, 0.5)
    # Generous max_tokens as safety net (prompt controls actual length)
    max_tokens = 600
    
    llm = get_llm(model_config, temperature, top_p, frequency_penalty, presence_penalty, max_tokens)
    
    prompt = get_review_prompt(state["options"])
    
    # Randomize length instruction for varied review lengths
    length_options = [
        "Keep your review brief and to the point (2-3 sentences).",
        "Write a short but helpful review (3-4 sentences).",
        "Write a moderately detailed review covering your main impressions.",
        "Write a detailed and thorough review of your experience.",
        "Write a comprehensive review with specific examples from your usage."
    ]
    length_instruction = random.choice(length_options)
    
    # Fill in the prompt details
    chain = prompt | llm | JsonOutputParser()
    
    inputs = {
        "role": state["persona"]["role"],
        "description": state["persona"]["description"],
        "product_name": state["product"]["name"],
        "product_type": state["product"]["type"],
        "product_description": state["product"]["description"],
        "rating": state["rating"],
        "research_context": state.get("research_context", "No research available."),
        "scenario": state.get("scenario", "General product usage."),
        "length_instruction": length_instruction
    }
    
    if state.get("feedback"):
        # We can append feedback to the prompt, or just rely on re-generation with temperature
        # For better results, we should ideally inject the feedback into messages. 
        # But `get_review_prompt` builds a template. 
        # A simple hack: Append feedback to description or instruction temporarily, 
        # or just rely on randomness for now as requested "regenerate".
        pass

    result = chain.invoke(inputs)
    
    return {"generated_review": result, "feedback": None}


def reality_check_node(state: ReviewState):
    """Check if the review's premises align with actual product research."""
    generated_review = state["generated_review"]
    research_context = state.get("research_context", "")
    model_config = state["model_config"]
    attempts = state.get("reality_check_attempts", 0) + 1
    
    passed = reality_check(generated_review, research_context, model_config)
    
    print(f"DEBUG: Reality check attempt {attempts}/{MAX_REALITY_CHECK_ATTEMPTS} - Passed: {passed}")
    
    if passed:
        return {
            "reality_check_passed": True,
            "reality_check_attempts": attempts,
            "feedback": None,
            "review_skipped": False
        }
    
    # Failed reality check
    if attempts >= MAX_REALITY_CHECK_ATTEMPTS:
        print(f"DEBUG: Max reality check attempts reached. Skipping this review.")
        return {
            "reality_check_passed": False,
            "reality_check_attempts": attempts,
            "feedback": None,
            "review_skipped": True,  # Skip this review entirely
            "generated_review": None  # Clear the failed review
        }
    
    # Need to regenerate
    return {
        "reality_check_passed": False,
        "reality_check_attempts": attempts,
        "feedback": f"Review failed reality check. Regenerating ({attempts}/{MAX_REALITY_CHECK_ATTEMPTS})...",
        "review_skipped": False
    }


def route_after_reality_check(state: ReviewState):
    """Route after reality check - skip, regenerate, or proceed to parallel QA checks."""
    if state.get("review_skipped", False):
        return END  # Skip to end, review will be None
    if not state.get("reality_check_passed", False):
        return "generate_review"
    return "parallel_qa_checks"


def parallel_qa_checks_node(state: ReviewState):
    """
    Run sentiment alignment and similarity checks in parallel.
    Both checks run together; if either fails beyond max attempts, the review is handled accordingly.
    """
    generated_review = state["generated_review"]
    actual_rating = state["rating"]
    model_config = state["model_config"]
    
    # Get current attempt counts
    sentiment_attempts = state.get("sentiment_alignment_attempts", 0) + 1
    similarity_attempts = state.get("similarity_attempts", 0) + 1
    best_review = state.get("best_review")
    best_similarity = state.get("best_similarity", float('inf'))
    threshold = state.get("similarity_threshold", 0.8)
    previous_embeddings = state.get("previous_embeddings")
    embedder = state.get("embedder")
    
    # Track predicted ratings history
    predicted_ratings_history = state.get("predicted_ratings_history", []).copy()
    
    # Run both checks (parallel execution within the node)
    sentiment_passed, predicted_rating = sentiment_alignment_check(generated_review, actual_rating, model_config)
    similarity_passed, max_similarity = check_semantic_similarity(
        generated_review, previous_embeddings, embedder, threshold
    )
    
    # Add current prediction to history
    predicted_ratings_history.append(predicted_rating)
    
    print(f"DEBUG: Parallel QA checks - Sentiment: attempt {sentiment_attempts}/{MAX_SENTIMENT_ALIGNMENT_ATTEMPTS}, predicted={predicted_rating}, actual={actual_rating}, passed={sentiment_passed}")
    print(f"DEBUG: Parallel QA checks - Similarity: attempt {similarity_attempts}/{MAX_SIMILARITY_ATTEMPTS}, max={max_similarity:.4f}, threshold={threshold}, passed={similarity_passed}")
    
    # Track best review for similarity
    if max_similarity < best_similarity:
        best_similarity = max_similarity
        best_review = generated_review
    
    # Check sentiment alignment first - if it fails and max attempts reached, try to use mode rating
    if not sentiment_passed:
        if sentiment_attempts >= MAX_SENTIMENT_ALIGNMENT_ATTEMPTS:
            # Find the most common predicted rating (mode)
            from collections import Counter
            rating_counts = Counter(predicted_ratings_history)
            most_common = rating_counts.most_common()
            
            # Check if there's a repeated rating (count > 1)
            if most_common and most_common[0][1] > 1:
                new_rating = most_common[0][0]
                print(f"DEBUG: Max sentiment alignment attempts reached. Reassigning rating from {actual_rating} to mode {new_rating} (predictions: {predicted_ratings_history})")
                # Update the review with the new rating
                updated_review = generated_review.copy()
                updated_review["rating"] = new_rating
                return {
                    "sentiment_alignment_passed": True,  # Accept with new rating
                    "sentiment_alignment_attempts": sentiment_attempts,
                    "similarity_attempts": similarity_attempts,
                    "best_review": best_review,
                    "best_similarity": best_similarity,
                    "predicted_ratings_history": predicted_ratings_history,
                    "feedback": None,
                    "review_skipped": False,
                    "generated_review": updated_review,
                    "rating": new_rating  # Update the state rating
                }
            else:
                # No repeated rating, skip the review
                print(f"DEBUG: Max sentiment alignment attempts reached. No repeated rating in predictions {predicted_ratings_history}. Skipping review.")
                return {
                    "sentiment_alignment_passed": False,
                    "sentiment_alignment_attempts": sentiment_attempts,
                    "similarity_attempts": similarity_attempts,
                    "best_review": best_review,
                    "best_similarity": best_similarity,
                    "predicted_ratings_history": predicted_ratings_history,
                    "feedback": None,
                    "review_skipped": True,
                    "generated_review": None
                }
        # Need to regenerate for sentiment
        return {
            "sentiment_alignment_passed": False,
            "sentiment_alignment_attempts": sentiment_attempts,
            "similarity_attempts": 0,  # Reset similarity for new review
            "best_review": best_review,
            "best_similarity": best_similarity,
            "predicted_ratings_history": predicted_ratings_history,
            "feedback": f"Review sentiment mismatch (predicted {predicted_rating}, actual {actual_rating}). Regenerating ({sentiment_attempts}/{MAX_SENTIMENT_ALIGNMENT_ATTEMPTS})...",
            "review_skipped": False,
            "reality_check_attempts": 0
        }
    
    # Sentiment passed, now check similarity
    if not similarity_passed:
        if similarity_attempts >= MAX_SIMILARITY_ATTEMPTS:
            print(f"DEBUG: Max similarity attempts reached. Using best review with similarity: {best_similarity:.4f}")
            return {
                "sentiment_alignment_passed": True,
                "sentiment_alignment_attempts": sentiment_attempts,
                "similarity_attempts": similarity_attempts,
                "generated_review": best_review,
                "best_review": best_review,
                "best_similarity": best_similarity,
                "predicted_ratings_history": [],  # Reset for next review
                "feedback": None,
                "review_skipped": False
            }
        # Need to regenerate for similarity
        return {
            "sentiment_alignment_passed": True,
            "sentiment_alignment_attempts": 0,  # Reset sentiment for new review
            "similarity_attempts": similarity_attempts,
            "best_review": best_review,
            "best_similarity": best_similarity,
            "predicted_ratings_history": [],  # Reset for new review
            "feedback": f"Review is too similar (Max: {max_similarity:.4f}). Regenerating ({similarity_attempts}/{MAX_SIMILARITY_ATTEMPTS})...",
            "review_skipped": False,
            "reality_check_attempts": 0
        }
    
    # Both checks passed!
    return {
        "sentiment_alignment_passed": True,
        "sentiment_alignment_attempts": sentiment_attempts,
        "similarity_attempts": similarity_attempts,
        "best_review": best_review,
        "best_similarity": best_similarity,
        "predicted_ratings_history": [],  # Reset for next review
        "feedback": None,
        "review_skipped": False
    }


def route_after_parallel_qa_checks(state: ReviewState):
    """Route after parallel QA checks - skip, regenerate, or end."""
    if state.get("review_skipped", False):
        return END
    if state.get("feedback"):
        return "generate_review"
    return END


def build_graph():
    builder = StateGraph(ReviewState)
    
    builder.add_node("generate_scenario", generate_scenario_node)
    builder.add_node("generate_review", generate_review_node)
    builder.add_node("reality_check", reality_check_node)
    builder.add_node("parallel_qa_checks", parallel_qa_checks_node)
    
    # Flow: START -> scenario -> review -> reality_check -> parallel_qa_checks -> END
    builder.add_edge(START, "generate_scenario")
    builder.add_edge("generate_scenario", "generate_review")
    builder.add_edge("generate_review", "reality_check")
    
    # After reality check: skip (END), regenerate, or proceed to parallel QA checks
    builder.add_conditional_edges(
        "reality_check",
        route_after_reality_check,
        {
            "generate_review": "generate_review",
            "parallel_qa_checks": "parallel_qa_checks",
            END: END
        }
    )
    
    # After parallel QA checks: skip (END), regenerate, or end successfully
    builder.add_conditional_edges(
        "parallel_qa_checks",
        route_after_parallel_qa_checks,
        {
            "generate_review": "generate_review",
            END: END
        }
    )
    
    return builder.compile()
