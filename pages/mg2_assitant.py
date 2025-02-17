import os
import logfire
import asyncio
import streamlit as st
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from supabase import Client
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from rag_mg2_assitant import magento_ai_expert, MagentoAIDeps

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Define system prompt
system_prompt = (
    "You are an expert assistant for Magento 2 admin users. "
    "Your role is to provide precise, accurate, and easy-to-understand explanations "
    "about Magento 2 administration, troubleshooting, customization, and best practices. "
    "Ensure that your responses are structured, informative, and suitable for users of varying experience levels. "
    "If possible, include step-by-step instructions, command-line examples, or references to Magento documentation. "
    "For coding-related questions, provide well-commented code snippets."
    "Don't start with login, just answer with the understanding that you are already logged in and are an admin user."
)

# Initialize OpenAI model and agent
model = OpenAIModel(OPENAI_MODEL, api_key=OPENAI_API_KEY)
agent = Agent(model=model, system_prompt=system_prompt)


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = MagentoAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with magento_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )
        return partial_text

st.title("Magento 2 Admin Assistant")

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)     

# Initialize chat history in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all messages from the conversation so far
# Each message is either a ModelRequest or ModelResponse.
# We iterate over their parts to decide how to display them.
for msg in st.session_state.messages:
    if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
        for part in msg.parts:
                        display_message_part(part)

# Handle user input
prompt = st.chat_input("Ask me anything about Magento 2...")
if prompt:
    logfire.info(f"Received user prompt: {prompt}")
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(
        ModelRequest(parts=[UserPromptPart(content=prompt)])
    )

    # Get and display assistant response
    with st.chat_message("assistant"):
        response = asyncio.run(run_agent_with_streaming(prompt))
        logfire.info(f"Reponse result: {response}")