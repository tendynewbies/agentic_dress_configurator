from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import logging
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentInitializationError(Exception):
    """Custom exception for agent initialization errors."""
    pass

def create_dress_configurator_agent(temperature: float = 0.1, api_key: Optional[str] = None):
    """
    Creates and returns a LangChain Agent for video recommendations.

    Args:
        temperature (float): Controls randomness in the LLM's output (0.0 to 1.0).
        api_key (str, optional): OpenAI API key. If not provided, will try to get from environment variable.

    Returns:
        An agent instance ready to process recommendation requests.

    Raises:
        AgentInitializationError: If there's an error initializing the agent or LLM.
    """
    # Check if API key is provided or available in environment
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise AgentInitializationError(
            "OpenAI API key not found. Please create a .env file with OPENAI_API_KEY="
            " or set the environment variable directly."
        )

    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=temperature,
            timeout=30
        )

        system_instruction = ("You are an expert theme and occasion based dress configuration assistant. "
                              "Help users design and plan their wardrobe contents according to their theme and occasion. "
                              "So user's geography plays an important role in the recommendations. "
                              "For example, if the user is from India, then the recommendations should be based on Indian fashion. "
                              "If the user is from the US, then the recommendations should be based on US fashion.")
        agent = create_agent(
            model=llm,
            debug=True,
            tools=[],
            system_prompt=system_instruction
        )

        return agent

    except Exception as e:
        if "Incorrect API key" in str(e):
            raise AgentInitializationError("Invalid OpenAI API key provided.") from e
        raise AgentInitializationError(f"Failed to initialize agent: {str(e)}") from e
