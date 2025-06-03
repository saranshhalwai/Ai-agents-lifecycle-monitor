# servers/test_runner.py

"""
Test Runner MCP Server

This server exposes:
  - Resource: list_tests() → returns available test scenario names.
  - Tool: run_test(test_name, agent_version) → returns pass/fail results for that test on the specified agent version.

In a full implementation, run_test would load an agent’s prompt/config and execute inference. Here, we stub results for a hackathon MVP.
"""

from mcp.server.fastmcp import FastMCP

# Create a FastMCP instance named "TestRunnerServer" with port 6001
mcp = FastMCP("TestRunnerServer", port=6001)

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

if __name__ == "__main__":
    mcp.run()
