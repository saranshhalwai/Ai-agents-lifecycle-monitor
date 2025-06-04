'''# servers/test_runner_server.py

import threading
import time
import traceback
from mcp.server.fastmcp import FastMCP

# Instantiate FastMCP, binding to all interfaces on port 7001
mcp = FastMCP("TestRunnerServer", host="0.0.0.0", port=7001)

@mcp.resource("resource://list_tests")
def list_tests() -> list[str]:
    """
    Returns a list of test scenario identifiers.
    Example scenarios: “sanity_check”, “billing_edge_case”.
    """
    return ["sanity_check", "billing_edge_case"]

@mcp.tool()
def run_test(test_name: str, agent_version: str) -> dict:
    """
    Runs a single test scenario against the specified agent version.

    Args:
        test_name (str): Name of the test scenario (e.g., "sanity_check").
        agent_version (str): Identifier of the agent version to test (e.g., "v1.0").

    Returns:
        dict: A structure containing:
            - test_name (str)
            - status (str): "pass" or "fail"
            - details (dict): { passed: int, failed: int, log: str }
    """
    # Stubbed logic for demonstration:
    # In a real system, you would load the agent (e.g., a prompt + LLM),
    # run the scenario, and compute real pass/fail counts.
    if test_name == "sanity_check":
        return {
            "test_name": test_name,
            "status": "pass",
            "details": {
                "passed": 5,
                "failed": 0,
                "log": "All sanity checks passed successfully."
            }
        }
    else:
        # billing_edge_case or any other test_name
        return {
            "test_name": test_name,
            "status": "fail",
            "details": {
                "passed": 3,
                "failed": 2,
                "log": "2 out of 5 billing edge cases failed."
            }
        }

def _run_test_runner():
    """
    Internal helper to start the TestRunnerServer and print heartbeat.
    """
    def heartbeat():
        while True:
            print(">>> [test_runner] still running…")
            time.sleep(5)

    # Start a daemon thread for periodic heartbeat logs
    threading.Thread(target=heartbeat, daemon=True).start()

    print(">>> [test_runner] About to bind on port 7001 …")
    try:
        mcp.run()
        # If mcp.run() ever returns, it means the server has stopped
        print("*** [test_runner] mcp.run() returned (server has exited) ***")
    except Exception:
        print("*** [test_runner] mcp.run() failed with exception:***")
        traceback.print_exc()

if __name__ == "__main__":
    _run_test_runner()'''
# servers/test_runner_server.py

import threading
import time
import traceback
from mcp.server.fastmcp import FastMCP

# Define HOST and PORT
HOST = "0.0.0.0"
PORT = 7001
LOG_LEVEL = "INFO" # Define log level here

# Instantiate FastMCP, passing ALL server configuration here
mcp = FastMCP("TestRunnerServer", host=HOST, port=PORT, log_level=LOG_LEVEL) # <-- Pass host, port, and log_level here

@mcp.resource("resource://list_tests")
def list_tests() -> list[str]:
    return ["sanity_check", "billing_edge_case"]

@mcp.tool()
def run_test(test_name: str, agent_version: str) -> dict:
    if test_name == "sanity_check":
        return {
            "test_name": test_name,
            "status": "pass",
            "details": {
                "passed": 5,
                "failed": 0,
                "log": "All sanity checks passed successfully."
            }
        }
    else:
        return {
            "test_name": test_name,
            "status": "fail",
            "details": {
                "passed": 3,
                "failed": 2,
                "log": "2 out of 5 billing edge cases failed."
            }
        }

def _run_test_runner():
    def heartbeat():
        while True:
            print(">>> [test_runner] still running…")
            time.sleep(5)

    threading.Thread(target=heartbeat, daemon=True).start()

    # Now, when calling mcp.run(), only pass the transport.
    # All other configuration is already part of the 'mcp' instance.
    print(f">>> [test_runner] Attempting to bind TestRunnerServer on port {PORT} (host {HOST}) using Streamable HTTP…")
    try:
        mcp.run(transport="streamable-http") # <-- Only transport argument here
        print(">>> [test_runner] INFO: TestRunnerServer (Streamable HTTP) has started successfully.")
    except Exception as e:
        print(f"!!! [test_runner] ERROR: Failed to start TestRunnerServer (Streamable HTTP): {e}")
        traceback.print_exc()
    finally:
        print(">>> [test_runner] TestRunnerServer thread is terminating.")

if __name__ == "__main__":
    _run_test_runner()

