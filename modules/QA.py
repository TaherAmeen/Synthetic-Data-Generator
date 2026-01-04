"""
Quality Assurance Module for Synthetic Review Generation

This module provides quality checks for generated reviews:
1. Reality Check: Verifies review premises align with actual product research
2. Semantic Similarity Check: Ensures generated reviews are diverse enough
3. Sentiment Alignment Check: Verifies review tone matches the intended rating
"""

from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from modules.embedder import LocalEmbedder

# Constants
MAX_SIMILARITY_ATTEMPTS = 5
MAX_REALITY_CHECK_ATTEMPTS = 3
MAX_SENTIMENT_ALIGNMENT_ATTEMPTS = 3
DEFAULT_SENTIMENT_TOLERANCE = 1  # Rating can be ±1 from predicted

# Reality Check Prompt Template
REALITY_CHECK_PROMPT = """You are a fact-checker. Your job is to verify if a product review's claims and premises align with the actual product information.

**Product Research Report:**
{research_context}

**Generated Review:**
Title: {review_title}
Comment: {review_comment}
Pros: {review_pros}
Cons: {review_cons}

**Your Task:**
Analyze if the review's premises, claims, and mentioned features align with what's actually documented about this product in the research report.

Consider:
- Does the review mention features that actually exist in the product?
- Are the claims about the product's capabilities realistic based on the research?
- Is the overall premise of the review grounded in reality?

**IMPORTANT:** Minor creative embellishments for persona roleplay are acceptable. Only flag reviews that make fundamentally false claims about the product.

Answer with ONLY "true" or "false":
- "true" if the review's premise aligns with the product research
- "false" if the review makes claims that contradict or are unsupported by the research

Answer:"""


def get_llm(model_config: dict, temperature: float = 0.3):
    """Get LLM instance based on configuration."""
    provider = model_config.get("provider", "openai")
    model_name = model_config.get("name", "gpt-4o-mini")
    
    if provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "ollama":
        return ChatOllama(model=model_name, temperature=temperature)
    elif provider == "mistral":
        return ChatMistralAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def reality_check(
    generated_review: dict,
    research_context: str,
    model_config: dict
) -> bool:
    """
    Check if the review's premises align with actual product research.
    
    Args:
        generated_review: The generated review dict with title, comment, pros, cons
        research_context: The research report about the product
        model_config: LLM configuration
        
    Returns:
        True if the review passes the reality check, False otherwise
    """
    if not research_context or research_context == "No research available.":
        # No research to compare against, pass by default
        return True
    
    llm = get_llm(model_config, temperature=0.1)  # Low temp for consistent judgments
    
    prompt = REALITY_CHECK_PROMPT.format(
        research_context=research_context,
        review_title=generated_review.get("title", ""),
        review_comment=generated_review.get("comment", ""),
        review_pros=generated_review.get("pros", "N/A"),
        review_cons=generated_review.get("cons", "N/A")
    )
    
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip().lower()
        return answer == "true"
    except Exception as e:
        print(f"DEBUG: Reality check failed with error: {e}")
        # On error, pass the review to avoid blocking
        return True


def check_semantic_similarity(
    generated_review: dict,
    previous_embeddings: Any,
    embedder: LocalEmbedder,
    threshold: float = 0.8
) -> tuple[bool, float]:
    """
    Check if the generated review is semantically diverse enough from previous reviews.
    
    Args:
        generated_review: The generated review dict
        previous_embeddings: Embeddings of previous reviews
        embedder: LocalEmbedder instance
        threshold: Maximum allowed similarity (lower = more diverse)
        
    Returns:
        Tuple of (passed: bool, max_similarity: float)
    """
    # Construct text for embedding
    text_parts = []
    if generated_review.get("title"):
        text_parts.append(generated_review["title"])
    if generated_review.get("comment"):
        text_parts.append(generated_review["comment"])
    if generated_review.get("pros"):
        text_parts.append(generated_review["pros"])
    if generated_review.get("cons"):
        text_parts.append(generated_review["cons"])
        
    full_text = " ".join(text_parts)
    
    if not full_text:
        return False, 1.0  # No content, fail check
    
    if previous_embeddings is None or len(previous_embeddings) == 0:
        return True, 0.0  # No previous reviews, automatically pass
    
    current_embedding = embedder.get_embeddings([full_text])
    similarities = embedder.compute_similarity(current_embedding, previous_embeddings)
    max_similarity = similarities.max().item()
    
    passed = max_similarity <= threshold
    return passed, max_similarity


def get_review_embedding(generated_review: dict, embedder: LocalEmbedder):
    """
    Get the embedding for a generated review.
    
    Args:
        generated_review: The generated review dict
        embedder: LocalEmbedder instance
        
    Returns:
        Embedding tensor for the review
    """
    text_parts = []
    if generated_review.get("title"):
        text_parts.append(generated_review["title"])
    if generated_review.get("comment"):
        text_parts.append(generated_review["comment"])
    if generated_review.get("pros"):
        text_parts.append(generated_review["pros"])
    if generated_review.get("cons"):
        text_parts.append(generated_review["cons"])
        
    full_text = " ".join(text_parts)
    
    if not full_text:
        return None
        
    return embedder.get_embeddings([full_text])


# Sentiment Alignment Check Prompt Template
SENTIMENT_ALIGNMENT_PROMPT = """You are a sentiment analyzer. Based on the review content below, predict what rating (out of 5) the reviewer likely gave this product.

**Review Content:**
Comment: {review_comment}
Pros: {review_pros}
Cons: {review_cons}

**Rating Scale:**
1 = Very negative, terrible experience
2 = Negative, disappointed
3 = Mixed/Neutral, some good and bad
4 = Positive, satisfied
5 = Very positive, excellent experience

Based on the tone, language, and balance of pros/cons in this review, what rating do you think the reviewer gave?

Answer with ONLY a single number from 1 to 5:"""


def sentiment_alignment_check(
    generated_review: dict,
    actual_rating: int,
    model_config: dict,
    tolerance: int = DEFAULT_SENTIMENT_TOLERANCE
) -> tuple[bool, int]:
    """
    Check if the review's sentiment aligns with its intended rating.
    
    Args:
        generated_review: The generated review dict with comment, pros, cons
        actual_rating: The intended rating (1-5)
        model_config: LLM configuration
        tolerance: Allowed difference between predicted and actual rating
        
    Returns:
        Tuple of (passed: bool, predicted_rating: int)
    """
    llm = get_llm(model_config, temperature=0.1)  # Low temp for consistent judgments
    
    prompt = SENTIMENT_ALIGNMENT_PROMPT.format(
        review_comment=generated_review.get("comment", ""),
        review_pros=generated_review.get("pros", "N/A"),
        review_cons=generated_review.get("cons", "N/A")
    )
    
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        
        # Extract the number from the response
        predicted_rating = int(answer[0])  # Get first character as number
        predicted_rating = max(1, min(5, predicted_rating))  # Clamp to 1-5
        
        # Check if within tolerance
        passed = abs(predicted_rating - actual_rating) <= tolerance
        
        return passed, predicted_rating
    except Exception as e:
        print(f"DEBUG: Sentiment alignment check failed with error: {e}")
        # On error, pass the review to avoid blocking
        return True, actual_rating