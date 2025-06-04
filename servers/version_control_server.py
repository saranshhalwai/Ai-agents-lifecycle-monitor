# servers/version_control_server.py

import threading # Added for consistency, though not strictly used in this snippet yet
import time      # Added for consistency (if you later add a heartbeat)
import traceback
from mcp.server.fastmcp import FastMCP

# Define HOST, PORT, and LOG_LEVEL constants
HOST = "0.0.0.0"  # Assuming you want it to listen on all interfaces
PORT = 7003       # Specific port for VersionControlServer
LOG_LEVEL = "INFO" # Recommended log level

# Instantiate FastMCP, passing ALL server configuration here
mcp = FastMCP("VersionControlServer", host=HOST, port=PORT, log_level=LOG_LEVEL)

@mcp.resource("resource://list_versions")
def list_versions() -> list[str]:
    return ["v1.0", "v2.0"]

@mcp.tool()
def compare_versions(left: str, right: str) -> dict:
    return {
        "metric_differences": {"accuracy": 0.03, "latency": 0.10},
        "diff_report_url": ""
    }

@mcp.tool()
def store_version(
    new_version: str,
    base_version: str,
    model_uri: str,
    prompt_config: dict,
    notes: str
) -> dict:
    return {"success": True, "new_version": new_version}

def _run_version_control():
    # Adding a heartbeat for consistency in monitoring, if you want it
    def heartbeat():
        while True:
            print(">>> [version_control] still running…")
            time.sleep(5)

    threading.Thread(target=heartbeat, daemon=True).start()

    # Log what's being attempted before mcp.run()
    print(f">>> [version_control] Attempting to bind VersionControlServer on port {PORT} (host {HOST}) using Streamable HTTP…")
    try:
        # Call mcp.run() with the streamable-http transport
        # Host, port, and log_level are already configured in the FastMCP instance
        mcp.run(transport="streamable-http")
        print(">>> [version_control] INFO: VersionControlServer (Streamable HTTP) has started successfully.")
    except Exception as e:
        print(f"!!! [version_control] ERROR: Failed to start VersionControlServer (Streamable HTTP): {e}")
        traceback.print_exc()
    finally:
        print(">>> [version_control] VersionControlServer thread is terminating.")


if __name__ == "__main__":
    _run_version_control()