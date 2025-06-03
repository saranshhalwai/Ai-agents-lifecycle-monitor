'''import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()'''
# File: app.py

import threading
from servers.test_runner_server import mcp as test_runner_mcp
from servers.drift_monitor_server import mcp as drift_monitor_mcp
from servers.version_control_server import mcp as version_control_mcp
from servers.retraining import mcp as retraining_mcp
from dashboard.gradio_dashboard import launch_dashboard


def start_test_runner():
    # FastMCP was instantiated with its port inside test_runner.py
    test_runner_mcp.run()

def start_drift_monitor():
    drift_monitor_mcp.run()

def start_version_control():
    version_control_mcp.run()

def start_retraining():
    retraining_mcp.run()

if __name__ == "__main__":
    # Launch each micro-server in its own daemon thread
    for fn in (start_test_runner, start_drift_monitor, start_version_control, start_retraining):
        t = threading.Thread(target=fn, daemon=True)
        t.start()

    # Launch the Gradio dashboard (blocks on port 7860)
    launch_dashboard()

