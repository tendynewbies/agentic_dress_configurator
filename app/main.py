from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from app.agent import create_dress_configurator_agent, AgentInitializationError
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agentic Dress Configurator")

class Query(BaseModel):
    prompt: str

@app.post("/agent/run")
def run_agent(query: Query):
    try:
        logging.debug("Running agent with query: %s", query.prompt)
        # Try to get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        agent = create_dress_configurator_agent(api_key=api_key)

        # Handle ping request for status check
        if query.prompt.lower() == "ping":
            return {"output": "pong"}

        # Run the agent using the new invoke API
        response = agent.invoke({"input": query.prompt})

        # Parse the response to extract the message content
        if isinstance(response, dict):
            if 'messages' in response:
                # If response has messages array, extract the content from the last message
                if response['messages'] and hasattr(response['messages'][-1], 'content'):
                    return {"output": response['messages'][-1].content}
            elif 'output' in response:
                return {"output": response['output']}
            elif 'response' in response:
                return {"output": response['response']}

        # If we can't find the expected format, return the string representation
        return {"output": str(response)}
    except AgentInitializationError as e:
        logger.error(f"Agent initialization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Agent initialization failed",
                "message": str(e),
                "solution": "Please ensure the OPENAI_API_KEY environment variable is set correctly"
            }
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred while processing your request"
            }
        )