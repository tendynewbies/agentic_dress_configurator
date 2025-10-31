import gradio as gr
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8001/agent/run"


def check_server_status() -> tuple[bool, str]:
    """Check if the FastAPI server is running and return status with message."""
    try:
        # Check the agent endpoint
        response = requests.post(
            "http://127.0.0.1:8001/agent/run",
            json={"prompt": "ping"},
            timeout=5
        )
        if response.status_code == 200:
            return True, "Server is running and responding."
        else:
            return False, f"Agent endpoint returned status code: {response.status_code}. Response: {response.text}"

    except requests.exceptions.ConnectionError:
        return False, "❌ Error: Could not connect to the server. Make sure the FastAPI server is running on port 8001."
    except requests.exceptions.Timeout:
        return False, "❌ Error: Connection to server timed out. The server might be overloaded or not responding."
    except requests.exceptions.RequestException as e:
        return False, f"❌ Error: {str(e)}"
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"


def chat(message: str, history: list) -> str:
    """Handle chat messages with error handling."""
    if not message or not message.strip():
        return "Please enter a message to start the chat."

    try:
        logger.info(f"Sending request to {API_URL} with message: {message}")
        response = requests.post(
            API_URL,
            json={"prompt": message},
            timeout=60  # Increased timeout for potentially long-running agent tasks
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        data = response.json()
        logger.info(f"Received response: {data}")

        # Extract the output from the response
        if isinstance(data, dict) and 'output' in data:
            return data['output']
        elif isinstance(data, str):
            return data
        else:
            return str(data)  # Fallback to string representation

    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error from server: {str(e)}"
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_msg += f" - {e.response.text}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

    except requests.exceptions.RequestException as e:
        error_msg = f"Error communicating with the server: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"❌ {error_msg}"


def get_status_message() -> str:
    """Get the current server status message."""
    is_running, msg = check_server_status()
    return f"🟢 Server status: {msg}" if is_running else f"🔴 Server status: {msg}"


with gr.Blocks() as demo:
    gr.Markdown("### 🧠 AI OTT Configurator Chatbot (Local Agent Demo)")
    gr.Markdown(
        "This is a demo of the AI OTT Configurator. Please ensure the FastAPI server is running at http://127.0.0.1:8001")

    status = gr.Markdown(get_status_message())

    refresh_btn = gr.Button("🔄 Refresh Status")

    def update_status():
        return get_status_message()

    refresh_btn.click(update_status, outputs=status)
    demo.load(update_status, outputs=status)

    gr.ChatInterface(
        fn=chat,
        title="Agentic AI Configurator",
        description="Ask me anything about OTT configuration!"
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
