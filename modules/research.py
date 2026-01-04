import json
import os
import requests
import urllib.parse
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from modules.graph import get_llm

load_dotenv()

# --- Bright Data Client ---

class BrightDataClient:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.unlocker_zone = "mcp_unlocker"
        self._ensure_zone()

    def _ensure_zone(self):
        """Ensure the required Bright Data zone exists."""
        try:
            resp = requests.get(
                "https://api.brightdata.com/zone/get_active_zones",
                headers=self.headers
            )
            if resp.status_code == 200:
                zones = resp.json()
                if any(z.get('name') == self.unlocker_zone for z in zones):
                    return
            
            print(f"Creating Bright Data zone: {self.unlocker_zone}")
            resp = requests.post(
                "https://api.brightdata.com/zone",
                headers={**self.headers, "Content-Type": "application/json"},
                json={
                    "zone": {"name": self.unlocker_zone, "type": "unblocker"},
                    "plan": {"type": "unblocker"}
                }
            )
            if resp.status_code not in [200, 201]:
                print(f"Warning: Failed to create zone. Status: {resp.status_code}")
        except Exception as e:
            print(f"Warning: Could not ensure Bright Data zone: {e}")

    def search(self, query: str) -> dict:
        """Perform a web search and return parsed results."""
        q = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={q}"
        
        try:
            resp = requests.post(
                "https://api.brightdata.com/request",
                headers=self.headers,
                json={
                    "url": url,
                    "zone": self.unlocker_zone,
                    "format": "raw",
                    "data_format": "parsed"
                }
            )
            resp.raise_for_status()
            return json.loads(resp.text)
        except Exception as e:
            return {"error": str(e), "organic": []}

    def scrape(self, url: str) -> str:
        """Scrape a webpage as markdown."""
        try:
            resp = requests.post(
                "https://api.brightdata.com/request",
                headers=self.headers,
                json={
                    "url": url,
                    "zone": self.unlocker_zone,
                    "format": "raw",
                    "data_format": "markdown"
                }
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            return f"Error scraping: {e}"

# Initialize client
bd_client = None
api_key = os.getenv("BRIGHTDATA_API_KEY")
if api_key:
    bd_client = BrightDataClient(api_key)
else:
    print("Warning: BRIGHTDATA_API_KEY not found in environment variables.")

# --- Tool for Agent ---

@tool
def scrape_as_markdown(url: str) -> str:
    """
    Scrape a webpage and return its content as markdown.
    Use this to read the content of URLs from search results.
    """
    if not bd_client:
        return "Error: BRIGHTDATA_API_KEY not configured."
    return bd_client.scrape(url)

# --- State ---

class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    product_name: str
    model_config: dict
    official_urls: List[str]
    official_features: str
    review_urls: List[str]
    reviews_summary: str
    final_report: str

# --- Helper: Extract URLs from search results ---

def extract_urls_from_search(search_result: dict, max_urls: int = 5) -> List[str]:
    """Extract URLs from Bright Data parsed search results."""
    urls = []
    organic = search_result.get("organic", [])
    for item in organic[:max_urls]:
        if isinstance(item, dict) and "link" in item:
            urls.append(item["link"])
    return urls

# --- Nodes ---

def official_search_node(state: ResearchState):
    """
    Search for the official product website using simple query.
    """
    product_name = state["product_name"]
    
    # Simple search: just the product name
    query = product_name
    print(f"  Searching: {query}")
    
    if not bd_client:
        return {"official_urls": [], "messages": [AIMessage(content="No API key configured")]}
    
    search_result = bd_client.search(query)
    urls = extract_urls_from_search(search_result, max_urls=3)
    
    return {
        "official_urls": urls,
        "messages": [AIMessage(content=f"Official search found {len(urls)} URLs")]
    }

def official_page_reader_node(state: ResearchState):
    """
    Use an agent to read official pages and extract features.
    """
    official_urls = state.get("official_urls", [])
    if not official_urls:
        return {"official_features": "No official URLs found."}

    product_name = state["product_name"]
    model_config = state["model_config"]
    urls_list = "\n".join(f"- {url}" for url in official_urls[:2])
    
    # Create agent with scrape tool
    llm = get_llm(model_config)
    agent = create_react_agent(llm, [scrape_as_markdown])
    
    task = f"""You are researching the product "{product_name}".

Here are the official/product URLs to investigate:
{urls_list}

Your task:
1. Scrape each URL to read its content
2. Extract the key product features and specifications
3. Ignore marketing fluff, focus on facts

Output a bullet-point list of features."""

    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    
    # Get the last AI message
    final_response = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_response = msg.content
            break
    
    return {"official_features": final_response, "messages": result["messages"]}

def review_search_node(state: ResearchState):
    """
    Search for product reviews using simple query.
    """
    product_name = state["product_name"]
    
    # Simple search: product name + reviews
    query = f"{product_name} reviews"
    print(f"  Searching: {query}")
    
    if not bd_client:
        return {"review_urls": [], "messages": [AIMessage(content="No API key configured")]}
    
    search_result = bd_client.search(query)
    urls = extract_urls_from_search(search_result, max_urls=5)
    
    return {
        "review_urls": urls,
        "messages": [AIMessage(content=f"Review search found {len(urls)} URLs")]
    }

def review_page_reader_node(state: ResearchState):
    """
    Use an agent to read review pages and synthesize findings.
    """
    review_urls = state.get("review_urls", [])
    if not review_urls:
        return {"reviews_summary": "No review URLs found."}

    product_name = state["product_name"]
    model_config = state["model_config"]
    urls_list = "\n".join(f"- {url}" for url in review_urls[:3])
    
    # Create agent with scrape tool
    llm = get_llm(model_config)
    agent = create_react_agent(llm, [scrape_as_markdown])
    
    task = f"""You are researching reviews for "{product_name}".

Here are review URLs to investigate:
{urls_list}

Your task:
1. Scrape each URL to read the review content
2. Identify common pros and cons mentioned
3. Note any disagreements between reviewers

Output a summary of findings."""

    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    
    # Get the last AI message
    final_response = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_response = msg.content
            break
    
    return {"reviews_summary": final_response, "messages": result["messages"]}

def final_report_node(state: ResearchState):
    """
    Generate the final research report.
    """
    official_features = state.get("official_features", "")
    reviews_summary = state.get("reviews_summary", "")
    product_name = state["product_name"]
    model_config = state["model_config"]
    
    sys_msg = SystemMessage(content=f"""You are a senior product analyst.
Create a concise research report for "{product_name}".

Official Features:
{official_features}

Reviews Summary:
{reviews_summary}

Your report should:
1. List key product features
2. Highlight what reviewers liked/disliked
3. Note any gaps between official claims and real-world feedback
4. Conclude with who this product is best suited for
""")
    
    llm = get_llm(model_config)
    response = llm.invoke([sys_msg])
    
    return {"final_report": response.content, "messages": [response]}

# --- Graph Construction ---

def build_research_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("official_search", official_search_node)
    workflow.add_node("official_page_reader", official_page_reader_node)
    workflow.add_node("review_search", review_search_node)
    workflow.add_node("review_page_reader", review_page_reader_node)
    workflow.add_node("final_report", final_report_node)

    workflow.add_edge(START, "official_search")
    workflow.add_edge("official_search", "official_page_reader")
    workflow.add_edge("official_page_reader", "review_search")
    workflow.add_edge("review_search", "review_page_reader")
    workflow.add_edge("review_page_reader", "final_report")
    workflow.add_edge("final_report", END)

    return workflow.compile()
