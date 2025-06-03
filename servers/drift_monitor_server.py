# servers/drift_monitor.py

from mcp.server.fastmcp import FastMCP
import random
import json
from typing import List, Dict
import os

# Attempt to import Modal stub; if missing, fall back to random drift
try:
    from modal_setup import compute_drift_score
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

# Create the FastMCP instance and bind to port 6002
mcp = FastMCP("DriftMonitorServer", port=6002)


@mcp.tool("fetch_recent")
def fetch_recent(agent_name: str, since: str) -> List[Dict]:
    """
    Returns a list of recent logs for the given agent since the timestamp.

    If `servers/sample_logs.json` exists, load from it; otherwise return dummy data.
    """
    log_path = os.path.join(os.path.dirname(__file__), "sample_logs.json")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # Fallback dummy data
        return [
            {
                "timestamp": "2025-05-28T12:00:00Z",
                "input": "How do I pay my bill?",
                "output": "You can pay via our portal at example.com."
            },
            {
                "timestamp": "2025-05-29T15:30:00Z",
                "input": "Billing complaint",
                "output": "We apologize for the inconvenience."
            }
        ]


@mcp.tool("detect_drift")
def detect_drift(agent_name: str, time_window: str, drift_type: str) -> dict:
    """
    Detects semantic drift for the given agent over the specified time window.

    If Modal is available, calls compute_drift_score remotely; otherwise returns a random score.
    """
    # 1) Fetch recent logs
    logs = fetch_recent(agent_name, since="2025-05-25T00:00:00Z")
    recent_texts = [item["input"] + " " + item["output"] for item in logs]

    # 2) Define a simple baseline (in production, load older logs from a DB or file)
    baseline = [
        {"input": "Hello, how can I assist you today?", "output": "I am here to help with billing questions."}
    ]
    baseline_texts = [item["input"] + " " + item["output"] for item in baseline]

    # 3) Compute drift_score
    if MODAL_AVAILABLE:
        try:
            import modal
            # Call Modal's remote function by name
            score = modal.Function.lookup("agent-lifecycle-retrain", "compute_drift_score")\
                           .call(recent_texts, baseline_texts)
        except Exception as e:
            print(f"[Modal Fallback] Error calling compute_drift_score: {e}")
            score = round(random.uniform(0, 1), 2)
    else:
        score = round(random.uniform(0, 1), 2)

    # 4) Build response
    status = "alert" if score > 0.3 else "ok"
    anomalies = []
    if status == "alert":
        anomalies.append({
            "timestamp": "2025-05-29T16:00:00Z",
            "description": "Detected a significant drop in semantic similarity over last week.",
            "severity": "medium"
        })

    return {
        "drift_score": score,
        "status": status,
        "anomalies": anomalies
    }


if __name__ == "__main__":
    # Run the MCP server on port 6002
    mcp.run()
