from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

PROMPT_TEMPLATES = {
    "system_base": """You are excellent at roleplaying. You are a {role}.
Description of your persona: {description}

IMPORTANT: Write authentically as this persona would. Vary your vocabulary, sentence structure, and expression naturally. Avoid generic phrases like "I highly recommend" or "game-changer" unless they genuinely fit your persona's voice.
""",
    "scenario_generation": """You are roleplaying as a {role}.
Persona description: {description}

You are considering the product "{product_name}" ({product_type}).

Here is what we know about the product's features and capabilities:
{research_context}

**Assigned features to focus on:**
{assigned_features}

**All available features (for reference):**
{all_features}

**Target Rating:** {rating}/5

**IMPORTANT INSTRUCTIONS:**
- Focus on the assigned features above for your scenario.
- If any assigned feature is NOT relevant or suitable for your persona/role, simply IGNORE that feature.
- If NONE of the assigned features are suitable for your role, you may select and use features from the "All available features" list that ARE relevant to your persona.
- Only mention features that make sense for someone in your role to actually use.
- **Crucial:** The scenario must set the stage for a review with a rating of {rating}/5.
  - If the rating is low (1-2), describe a scenario where you encountered significant issues, frustration, or where the features failed to meet your needs.
  - If the rating is medium (3), describe a mixed experience with some utility but also some drawbacks or limitations.
  - If the rating is high (4-5), describe a successful scenario where the features worked well and solved your problems.

Based on your role, the suitable features, and the target rating, create a brief, realistic usage scenario.
Describe how YOU would use these specific features in your work/life and what happened.
Be specific and personal.

Scenario:""",
    "task_instruction": """
You have bought the product: "{product_name}".
Product Type: {product_type}
Product Description: {product_description}

Research Context (Real-world data about this product):
{research_context}

Your Usage Scenario:
{scenario}

Your task is to write an authentic review for this product based on your experience in the scenario above.
You must give it a rating of {rating}/5.

**Writing Guidelines:**
{length_instruction}

**Authenticity Guidelines:**
- Write as your unique persona would - use vocabulary and tone appropriate for your role
- Be specific about YOUR experience, not generic praise/criticism
- Mention concrete features or aspects from your scenario
- Vary your sentence structure naturally
- Do NOT use cliché review phrases unless they genuinely fit
- Do NOT mention your role or the product's name explicitly (unless it suits the narrative)
- Make it sound like a real person wrote this, not a template

**Avoid these overused phrases (unless genuinely appropriate):**
- "game-changer", "highly recommend", "couldn't be happier"
- "exceeded my expectations", "worth every penny"
- "intuitive interface", "user-friendly", "seamless experience"
- Starting with "I" as the first word
"""
}

def get_scenario_prompt() -> str:
    """
    Returns the scenario generation prompt template.
    """
    return PROMPT_TEMPLATES["scenario_generation"]

FEATURE_EXTRACTION_PROMPT = """Extract a list of distinct product features from the following research report.
Output ONLY a JSON array of short feature descriptions (1-5 words each).
Example: ["AI-powered course creation", "SCORM export", "Built-in analytics"]

Research Report:
{research_context}

Features (JSON array):"""

def get_review_prompt(options: dict) -> ChatPromptTemplate:
    """
    Constructs the prompt based on options.
    """
    system_template = PROMPT_TEMPLATES["system_base"]
    
    human_parts = [PROMPT_TEMPLATES["task_instruction"]]
    
    keys = ["- title: A short title for your review", "- comment: The main body of the review (general comments)"]
    if options.get("pros_and_cons", False):
        keys.append("- pros: A string describing the pros")
        keys.append("- cons: A string describing the cons")
    keys.append("- rating: The numerical rating ({rating})")
    
    output_format_str = "Return your response in standard JSON format with the following keys:\n" + "\n".join(keys)
    
    human_parts.append(output_format_str)
    
    human_template = "\n".join(human_parts)
    
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
