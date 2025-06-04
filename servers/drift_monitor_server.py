# servers/drift_monitor.py

import threading
import time # Added for heartbeat
import traceback # Added for better error logging
from mcp.server.fastmcp import FastMCP
import random
import json
from typing import List, Dict
import os

# Define HOST, PORT, and LOG_LEVEL constants
HOST = "0.0.0.0"  # Assuming you want it to listen on all interfaces
PORT = 7002       # Specific port for DriftMonitorServer
LOG_LEVEL = "INFO" # Recommended log level

# Attempt to import Modal stub; if missing, fall back to random drift
try:
    from modal_setup import compute_drift_score
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

# Create the FastMCP instance and bind to port 7002
# Pass all configuration (host, port, log_level) to the constructor
mcp = FastMCP("DriftMonitorServer", host=HOST, port=PORT, log_level=LOG_LEVEL)

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


def _run_drift_monitor():
    # Adding a heartbeat for consistency
    def heartbeat():
        while True:
            print(">>> [drift_monitor] still running…")
            time.sleep(5)

    threading.Thread(target=heartbeat, daemon=True).start()

    # Log what's being attempted before mcp.run()
    print(f">>> [drift_monitor] Attempting to bind DriftMonitorServer on port {PORT} (host {HOST}) using Streamable HTTP…")
    try:
        # Call mcp.run() with the streamable-http transport
        # Host, port, and log_level are already configured in the FastMCP instance
        mcp.run(transport="streamable-http")
        print(">>> [drift_monitor] INFO: DriftMonitorServer (Streamable HTTP) has started successfully.")
    except Exception as e:
        print(f"!!! [drift_monitor] ERROR: Failed to start DriftMonitorServer (Streamable HTTP): {e}")
        traceback.print_exc() # Print full traceback for debugging
    finally:
        print(">>> [drift_monitor] DriftMonitorServer thread is terminating.")

# (optional) keep the __main__ for standalone runs:
if __name__ == "__main__":
    _run_drift_monitor()