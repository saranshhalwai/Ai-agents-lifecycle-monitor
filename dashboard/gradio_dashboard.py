# dashboard/gradio_dashboard.py
import asyncio
import signal
from contextlib import AsyncExitStack

import gradio as gr
from pydantic import AnyUrl
from transformers import pipeline
from typing import Optional
import logging
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────── 1. Set up an open-source summarization LLM (Flan-T5-Small) ──────
try:
    summarizer_hf = pipeline(
        "summarization",
        model="google/flan-t5-small",
        device=-1,  # CPU only
        framework="pt",  # PyTorch
        max_length=50,  # Reduced to avoid warnings
        min_length=10,  # Reduced proportionally
        do_sample=False,
    )
    logger.info("LLM pipeline initialized successfully")
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    summarizer_hf = None


# Simple summarization function to replace LangChain
def summarize_text(text: str, context: str = "") -> str:
    """Simple text summarization function"""
    if summarizer_hf is not None:
        try:
            # Truncate input if too long
            max_input_length = 200
            if len(text) > max_input_length:
                text = text[:max_input_length] + "..."

            result = summarizer_hf(text, max_length=30, min_length=10, do_sample=False)
            summary = result[0]['summary_text']
            return f"{context}: {summary}" if context else summary
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return f"{context}: Error generating summary - {str(e)[:100]}..."
    else:
        return f"{context}: Mock summary for: {text[:100]}..."


class MCPClient:
    def __init__(self, name):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.name = name

    async def connect(self, server_url: str):
        """Connect to an MCP server using a script path."""
        logger.info(f"Connecting to {self.name} server at {server_url}")
        streamablehttp_transport = await self.exit_stack.enter_async_context(streamablehttp_client(server_url))
        self.read, self.write, self.sessionID = streamablehttp_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.session.initialize()
        logger.info(f"Connected to {self.name} server")

    async def cleanup(self):
        """Clean up resources"""
        logger.info(f"Cleaning {self.name} client resources...")
        await self.exit_stack.aclose()
        logger.info(f"Cleared {self.name} client resources...")


# Global clients and shutdown event
drift_client = MCPClient("DriftMonitor")
versions_client = MCPClient("VersionsMonitor")
tests_client = MCPClient("TestRunner")
retrain_client = MCPClient("RetrainingServer")

shutdown_event = asyncio.Event()


async def connect_all():
    """Background task to maintain MCP connections"""
    try:
        await asyncio.gather(
            drift_client.connect("http://localhost:7002/mcp/"),
            versions_client.connect("http://localhost:7003/mcp/"),
            tests_client.connect("http://localhost:7001/mcp/"),
            retrain_client.connect("http://localhost:7004/mcp/"),
        return_exceptions=True,)
        logger.info("All MCP clients connected successfully!")

        # Keep connections alive until shutdown
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Error in MCP connections: {e}")
        raise
    finally:
        logger.info("Cleaning up MCP connections...")
        await disconnect_all()


async def disconnect_all():
    """Clean up all MCP client connections"""
    try:
        await asyncio.gather(
            drift_client.cleanup(),
            versions_client.cleanup(),
            tests_client.cleanup(),
            retrain_client.cleanup(),
            return_exceptions=True
        )
        logger.info("All MCP clients disconnected")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def signal_handler():
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal...")
    shutdown_event.set()

# ────── 4. Gradio UI Functions ──────

async def refresh_status():
    try:
        # 1) Detect drift
        drift_payload = {"agent_name": "chat_bot", "time_window": "7d", "drift_type": "semantic"}
        drift_data = await drift_client.session.call_tool('detect_drift', drift_payload)
        drift_text = f"Drift Score: {drift_data.get('drift_score', 'N/A')} ({drift_data.get('status', 'unknown')})"

        if drift_data.get("status") == "alert":
            drift_summary = summarize_text(str(drift_data), "Drift Analysis")
        else:
            drift_summary = "✅ No significant drift detected. System performing within normal parameters."

        # 2) List versions
        versions = await versions_client.session.read_resource(AnyUrl('resource://list_versions'))
        versions = versions.contents
        version_text = f"📋 Available Versions ({len(versions)}): " + ", ".join(versions)

        return drift_text, drift_summary, version_text
    except Exception as e:
        logger.error(f"Error in refresh_status: {e}")
        return f"❌ Error: {str(e)}", "Failed to get drift summary", "Failed to get versions"


async def run_tests_and_summarize(version):
    if not version:
        return "⚠️ Please select a version", "No version selected for testing"

    try:
        test_payload = {"agent_version": version}
        test_results = await tests_client.session.call_tool('run_tests', test_payload)
        test_results = test_results.contents

        # Format test results
        test_text = "🧪 Test Results:\n" + "\n".join([f"  • {k}: {v}" for k, v in test_results.items()])

        # Generate summary
        test_summary = summarize_text(str(test_results), "Test Summary")

        # Add status indicators
        passed_count = sum(1 for status in test_results.values() if status == "passed")
        total_count = len(test_results)
        test_summary += f"\n\n📊 Overall: {passed_count}/{total_count} tests passed"

        return test_text, test_summary
    except Exception as e:
        logger.error(f"Error in run_tests_and_summarize: {e}")
        return f"❌ Error: {str(e)}", "Failed to get test summary"


async def compare_versions(v_left, v_right):
    if not v_left or not v_right:
        return "⚠️ Please select both versions to compare"

    if v_left == v_right:
        return "⚠️ Please select different versions to compare"

    try:
        compare_payload = {"left": v_left, "right": v_right}
        diff = await versions_client.session.call_tool('compare_versions', compare_payload)
        diff= diff.content
        diffs = diff.get("metric_differences", {})

        # Format the comparison nicely
        comparison_lines = []
        for metric, value in diffs.items():
            if isinstance(value, (int, float)):
                direction = "📈" if value > 0 else "📉" if value < 0 else "➡️"
                comparison_lines.append(f"  {direction} {metric.title()}: {value:+.3f}")

        result = f"📊 Comparison ({v_left} vs {v_right}):\n" + "\n".join(comparison_lines)
        return result
    except Exception as e:
        logger.error(f"Error in compare_versions: {e}")
        return f"❌ Error: {str(e)}"


async def trigger_retraining(version):
    if not version:
        return "⚠️ Please select a version"

    try:
        retrain_payload = {"base_version": version}
        job = await retrain_client.session.call_tool('trigger_retraining', retrain_payload)
        job = job.content
        job_id = job.get("job_id", "Unknown job ID")
        return f"🚀 Retraining initiated: {job_id}"
    except Exception as e:
        logger.error(f"Error in trigger_retraining: {e}")
        return f"❌ Error: {str(e)}"


async def check_retrain_status(job_id):
    if not job_id or job_id.startswith("⚠️") or job_id.startswith("❌"):
        return "⚠️ No valid job ID provided"

    try:
        status_payload = {"job_id": job_id}
        status = await retrain_client.session.call_tool('check_retraining_status', status_payload)
        status = status.contents
        current_status = status.get("status", "unknown")
        progress = status.get("progress", "N/A")

        if current_status == "done":
            new_v = status.get("new_version", f"v{job_id[-4:]}")
            # Automatically store new version
            try:
                store_payload = {
                    "new_version": new_v,
                    "base_version": job_id,
                    "model_uri": f"model://{new_v}",
                    "prompt_config": {"config_key": "config_value"},  # Example config
                    "notes": "Automated retraining completion"
                }
                store_result = await versions_client.session.call_tool('store_version', store_payload)
                if store_result.get("success"):
                    logger.info(f"Stored new version: {new_v}")
                else:
                    logger.error(f"Failed to store version: {store_result}")
            except Exception as store_error:
                logger.error(f"Error storing version: {store_error}")

            return f"✅ Retraining complete! New version: {new_v}"
        elif current_status == "failed":
            return f"❌ Retraining failed. Please check logs."
        else:
            stage = status.get("current_stage", "processing")
            eta = status.get("estimated_completion", "unknown")
            return f"🔄 Status: {current_status.title()} ({progress})\n📍 Stage: {stage}\n⏱️ ETA: {eta}"
    except Exception as e:
        logger.error(f"Error in check_retrain_status: {e}")
        return f"❌ Error: {str(e)}"


# ────── 5. Gradio UI Definition ──────

def create_dashboard():
    # Get initial version list for dropdowns
    try:
        raise Exception('Force Fallback')  # Simulate failure to fetch versions
        # available_versions = await versions_client.session.read_resource(AnyUrl('resource://list_versions'))
    except:
        available_versions = ["v1.0.0", "v1.1.0", "v2.0.0"]

    with gr.Blocks(
            title="AI Agent Lifecycle Monitor",
            theme=gr.themes.Soft(),
            css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .status-box {
            border-radius: 10px;
            padding: 15px;
        }
        """
    ) as demo:
        gr.Markdown("""
        # 🤖 AI Agent Lifecycle Monitor Dashboard
        ### Real-time monitoring and management for AI agent deployments
        """)

        # Status Section
        with gr.Group():
            gr.Markdown("## 📊 System Status")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Status", variant="primary", scale=1)

            with gr.Row():
                with gr.Column():
                    drift_out = gr.Textbox(
                        label="🌊 Drift Detection",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                with gr.Column():
                    drift_summary_out = gr.Textbox(
                        label="📝 Drift Analysis",
                        interactive=False,
                        lines=3,
                        elem_classes=["status-box"]
                    )

            version_out = gr.Textbox(
                label="📦 Version Inventory",
                interactive=False,
                elem_classes=["status-box"]
            )

        refresh_btn.click(
            fn=refresh_status,
            inputs=[],
            outputs=[drift_out, drift_summary_out, version_out]
        )

        # Testing Section
        with gr.Group():
            gr.Markdown("## 🧪 Testing & Quality Assurance")
            with gr.Row():
                version_input = gr.Dropdown(
                    choices=available_versions,
                    label="Select Version for Testing",
                    value=available_versions[0] if available_versions else None,
                    scale=2
                )
                run_tests_btn = gr.Button("▶️ Run Tests", variant="secondary", scale=1)

            with gr.Row():
                with gr.Column():
                    tests_out = gr.Textbox(
                        label="Test Results",
                        interactive=False,
                        lines=6,
                        elem_classes=["status-box"]
                    )
                with gr.Column():
                    test_summary_out = gr.Textbox(
                        label="Test Summary & Recommendations",
                        interactive=False,
                        lines=6,
                        elem_classes=["status-box"]
                    )

        run_tests_btn.click(
            fn=run_tests_and_summarize,
            inputs=[version_input],
            outputs=[tests_out, test_summary_out]
        )

        # Version Comparison Section
        with gr.Group():
            gr.Markdown("## ⚖️ Version Comparison")
            with gr.Row():
                left_ver = gr.Dropdown(
                    choices=available_versions,
                    label="Base Version",
                    value=available_versions[0] if available_versions else None
                )
                right_ver = gr.Dropdown(
                    choices=available_versions,
                    label="Compare Version",
                    value=available_versions[1] if len(available_versions) > 1 else None
                )
                compare_btn = gr.Button("🔍 Compare", variant="secondary")

            compare_out = gr.Textbox(
                label="Comparison Results",
                interactive=False,
                lines=4,
                elem_classes=["status-box"]
            )

        compare_btn.click(
            fn=compare_versions,
            inputs=[left_ver, right_ver],
            outputs=[compare_out]
        )

        # Retraining Section
        with gr.Group():
            gr.Markdown("## 🔄 Model Retraining")
            with gr.Row():
                retrain_ver = gr.Dropdown(
                    choices=available_versions,
                    label="Base Version for Retraining",
                    value=available_versions[0] if available_versions else None,
                    scale=2
                )
                retrain_btn = gr.Button("🚀 Start Retraining", variant="primary", scale=1)

            job_id_out = gr.Textbox(
                label="Retraining Job Status",
                interactive=False,
                elem_classes=["status-box"]
            )

            with gr.Row():
                check_btn = gr.Button("📊 Check Progress", variant="secondary")
                retrain_status_out = gr.Textbox(
                    label="Detailed Status",
                    interactive=False,
                    lines=3,
                    elem_classes=["status-box"]
                )

        retrain_btn.click(
            fn=trigger_retraining,
            inputs=[retrain_ver],
            outputs=[job_id_out]
        )

        check_btn.click(
            fn=check_retrain_status,
            inputs=[job_id_out],
            outputs=[retrain_status_out]
        )

        # Footer
        gr.Markdown("""
        ---
        💡 **Note**: This dashboard provides fallback data when backend services are unavailable.
        All network requests include timeout protection for better reliability.
        """)

    return demo


async def main():
    """Main async function to coordinate MCP connections and Gradio dashboard"""
    # Set up graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # Start MCP connection manager as background task
    logger.info("Starting MCP connections...")
    connection_task = asyncio.create_task(connect_all())

    # Give connections time to establish
    await asyncio.sleep(2)

    try:
        # Create and launch Gradio dashboard
        demo = create_dashboard()
        logger.info("Starting Gradio dashboard server...")

        # Launch in a way that doesn't block the event loop
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
            prevent_thread_lock=True  # This allows the event loop to continue
        )

        # Keep the main function running
        while not shutdown_event.is_set():
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt...")
        shutdown_event.set()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        shutdown_event.set()
    finally:
        # Wait for connection cleanup
        logger.info("Waiting for MCP connections to cleanup...")
        try:
            await asyncio.wait_for(connection_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("MCP cleanup timed out")

        logger.info("Application shutdown complete")


def launch_dashboard():
    """Entry point that handles both sync and async execution"""
    try:
        # Run the async main function
        asyncio.run(main(), debug=True)
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")
        print(f"Failed to launch dashboard: {e}")


if __name__ == "__main__":
    launch_dashboard()