# servers/retraining.py

import threading
import time  # Added for heartbeat
import traceback  # Added for better error logging
from mcp.server.fastmcp import FastMCP
from modal import Function

# Define HOST, PORT, and LOG_LEVEL constants
HOST = "0.0.0.0"  # Assuming you want it to listen on all interfaces
PORT = 7004       # Specific port for RetrainingServer
LOG_LEVEL = "INFO" # Recommended log level

# Create the FastMCP instance and bind to port 7004
# Pass all configuration (host, port, log_level) to the constructor
mcp = FastMCP("RetrainingServer", host=HOST, port=PORT, log_level=LOG_LEVEL)


@mcp.tool("trigger_retraining")
def trigger_retraining(base_version: str) -> dict:
    """
    Enqueue a Modal job to retrain the agent from the specified base version.

    Args:
        base_version (str): The version to fine-tune from (e.g., "v1.0").

    Returns:
        dict: { "job_id": "<modal-job-id>" }
    """
    # Lookup the deployed Modal function by app name and function name
    modal_func = Function.lookup("agent-lifecycle-retrain", "trigger_retraining_job")
    handle = modal_func.call(base_version)
    return {"job_id": handle.id}


@mcp.tool("check_retraining_status")
def check_retraining_status(job_id: str) -> dict:
    """
    Poll Modal for the status of a previously triggered retraining job.

    Args:
        job_id (str): The Modal job ID obtained from trigger_retraining.

    Returns:
        dict: {
            "status": "<PENDING|RUNNING|SUCCEEDED|FAILED>",
            "new_version": "<new_version_id>"  # only if status == "SUCCEEDED"
        }
    """
    # Lookup the same deployed Modal function to check its job
    modal_func = Function.lookup("agent-lifecycle-retrain", "trigger_retraining_job")
    job = modal_func.get_job(job_id)

    if job.state == "SUCCEEDED":
        new_v = job.return_value.get("new_version")
        return {"status": "done", "new_version": new_v}
    else:
        return {"status": job.state, "new_version": None}


def _run_retraining():
    # Adding a heartbeat for consistency
    def heartbeat():
        while True:
            print(">>> [retraining] still running…")
            time.sleep(5)

    threading.Thread(target=heartbeat, daemon=True).start()

    # Log what's being attempted before mcp.run()
    print(f">>> [retraining] Attempting to bind RetrainingServer on port {PORT} (host {HOST}) using Streamable HTTP…")
    try:
        # Call mcp.run() with the streamable-http transport
        # Host, port, and log_level are already configured in the FastMCP instance
        mcp.run(transport="streamable-http")
        print(">>> [retraining] INFO: RetrainingServer (Streamable HTTP) has started successfully.")
    except Exception as e:
        print(f"!!! [retraining] ERROR: Failed to start RetrainingServer (Streamable HTTP): {e}")
        traceback.print_exc()  # Print full traceback for debugging
    finally:
        print(">>> [retraining] RetrainingServer thread is terminating.")

# (optional) keep the __main__ for standalone runs:
if __name__ == "__main__":
    _run_retraining()