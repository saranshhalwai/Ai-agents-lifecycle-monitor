# servers/retraining.py

"""
Retraining MCP Server

This server exposes:
  - Tool: trigger_retraining(base_version) → enqueues a Modal job for fine-tuning; returns a job_id.
  - Tool: check_retraining_status(job_id) → polls Modal to see if the retraining job is complete; returns status + new_version if done.

Uses Modal’s Function.lookup to call deployed functions, so no direct import of modal_setup.
"""

from mcp.server.fastmcp import FastMCP
from modal import Function

# Create the FastMCP instance and bind to port 6004
mcp = FastMCP("RetrainingServer", port=6004)


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


if __name__ == "__main__":
    # Run the MCP server on port 6004
    mcp.run()
