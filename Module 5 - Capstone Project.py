import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ==========================================
# 1. Define the Shared State (Orchestrator)
# ==========================================
# This acts as the shared memory across all three agents[cite: 8, 29].
class RMState(TypedDict):
    client_profile: Dict[str, Any]
    current_portfolio: Dict[str, Any]
    leads: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    diversification_report: List[Dict[str, Any]]

# ==========================================
# 2. Agent 1: Lead Sourcer
# ==========================================
# Scrapes financial news and market data to surface actionable leads[cite: 5, 37].
# *Note: For demo purposes, scraping is mocked to ensure reliable execution.*
def lead_sourcer_agent(state: RMState):
    print(">>> [Agent 1] Lead Sourcer: Scraping market data and news...")
    
    # Mocked structured lead signals [cite: 42]
    mock_leads = [
        {
            "ticker": "RELIANCE", 
            "headline": "Reliance announces major breakthrough in green energy sector",
            "trigger": "macro", 
            "sentiment": "Positive", 
            "score": 0.95, 
            "url": "https://news.mock/reliance"
        },
        {
            "ticker": "HDFCBANK", 
            "headline": "HDFC Bank Q3 earnings beat analyst estimates",
            "trigger": "earnings", 
            "sentiment": "Positive", 
            "score": 0.88, 
            "url": "https://news.mock/hdfc"
        }
    ]
    
    print(f"    Found {len(mock_leads)} actionable leads.")
    return {"leads": mock_leads}

# ==========================================
# 3. Agent 2: Portfolio Suggester
# ==========================================
# Analyses market trends and recommends HNI-grade investment opportunities[cite: 6].
def portfolio_suggester_agent(state: RMState):
    print(">>> [Agent 2] Portfolio Suggester: Analyzing leads against client profile...")
    leads = state.get("leads", [])
    client_profile = state.get("client_profile", {})
    
    suggestions = []
    # Simulating LLM reasoning based on HNI risk grade and leads [cite: 55, 56]
    for lead in leads:
        if lead["sentiment"] == "Positive":
            # Apply HNI filters (e.g., risk grade) [cite: 55]
            if client_profile.get("risk_grade") in ["Balanced", "Aggressive"]:
                suggestions.append({
                    "instrument": lead["ticker"],
                    "rationale": f"Strong positive sentiment driven by {lead['trigger']} event. Aligns with {client_profile['risk_grade']} profile.",
                    "horizon": "12-24 Months",
                    "risk": "Moderate-High",
                    "confidence": 0.92
                })
                
    print(f"    Generated {len(suggestions)} investment suggestions.")
    return {"suggestions": suggestions}

# ==========================================
# 4. Agent 3: Portfolio Diversifier
# ==========================================
# Evaluates existing portfolios and proposes rebalancing strategies[cite: 7].
def portfolio_diversifier_agent(state: RMState):
    print(">>> [Agent 3] Portfolio Diversifier: Evaluating portfolio health and concentration...")
    portfolio = state.get("current_portfolio", {})
    suggestions = state.get("suggestions", [])
    
    # Simulating concentration analysis and gap detection [cite: 65, 66]
    report = []
    
    equity_exposure = portfolio.get("asset_class_weights", {}).get("Equity", 0)
    if equity_exposure > 70:
        report.append({
            "action": "Reduce",
            "instrument": "Broad Equity Index",
            "target_weight": "-5%",
            "justification": "Equity exposure exceeds 70% threshold. Rebalancing required to mitigate risk."
        })
        
    # Cross-reference with Agent 2 suggestions [cite: 68]
    for sug in suggestions:
        report.append({
            "action": "Add",
            "instrument": sug["instrument"],
            "target_weight": "+2%",
            "justification": f"Adding {sug['instrument']} based on positive momentum to utilize cash reserves. Expected Sharpe improvement: +0.05"
        })
        
    print(f"    Generated {len(report)} diversification and rebalancing actions.")
    return {"diversification_report": report}

# ==========================================
# 5. Build the LangGraph Orchestrator
# ==========================================
def build_rm_orchestrator():
    # Initialize the graph with the shared state [cite: 31]
    workflow = StateGraph(RMState)
    
    # Add nodes (Agents) [cite: 28]
    workflow.add_node("lead_sourcer", lead_sourcer_agent)
    workflow.add_node("portfolio_suggester", portfolio_suggester_agent)
    workflow.add_node("portfolio_diversifier", portfolio_diversifier_agent)
    
    # Define the event-driven data flow (Edges) [cite: 98]
    workflow.set_entry_point("lead_sourcer")
    workflow.add_edge("lead_sourcer", "portfolio_suggester")
    workflow.add_edge("portfolio_suggester", "portfolio_diversifier")
    workflow.add_edge("portfolio_diversifier", END)
    
    # Compile the graph
    return workflow.compile()

# ==========================================
# 6. Execute the Demonstration
# ==========================================
if __name__ == "__main__":
    print("\n--- Starting Wealth & Asset Management Agentic AI Prototype ---\n")
    
    # Initialize app
    app = build_rm_orchestrator()
    
    # Sample HNI Client Data (Anonymised) [cite: 35]
    initial_state = {
        "client_profile": {
            "client_id": "HNI_8832",
            "risk_grade": "Balanced",
            "liquidity_constraints": "Low",
            "min_ticket_size_in_L": 50
        },
        "current_portfolio": {
            "total_value_in_cr": 15,
            "asset_class_weights": {
                "Equity": 75, # Deliberately high to trigger Agent 3 alert
                "Debt": 20,
                "Cash": 5
            },
            "holdings": ["TCS", "INFY", "GOVT_BONDS"]
        },
        "leads": [],
        "suggestions": [],
        "diversification_report": []
    }
    
    # Run the graph orchestrator
    final_state = app.invoke(initial_state)
    
    print("\n--- Final RM Workbench Output ---")
    print(json.dumps({
        "Generated Leads": final_state["leads"],
        "Investment Suggestions": final_state["suggestions"],
        "Diversification Actions": final_state["diversification_report"]
    }, indent=2))