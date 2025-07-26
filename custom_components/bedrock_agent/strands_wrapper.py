"""Wrapper for strands.Agent to make it easier to test and implement."""

import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError
from strands import Agent
from strands.models import BedrockModel
from strands.session.file_session_manager import FileSessionManager

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.llm import API, LLMContext

# Configure the root strands logger
logging.getLogger("strands").setLevel(logging.DEBUG)

_LOGGER = logging.getLogger(__name__)

class StrandsAgentWrapper:
    """Wrapper for strands.Agent to make it easier to test and implement."""

    def __init__(
        self,
        hass: HomeAssistant,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        model_id: str,
        apis: list[API],
        system_prompt: str | None = ""
    ) -> None:
        """Initialize the wrapper."""
        self.hass = hass
        self.tools = []
        self.apis = apis
        self.api_instances = {}
        self.llm_context = None
        self.modules = {}
        self.system_prompt = system_prompt

        # Create a boto3 session
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        # Create a Bedrock model with the custom session
        self.bedrock_model = BedrockModel(
            model_id=model_id,
            boto_session=session
        )

        self.session_manager = FileSessionManager(session_id="enno-123", storage_dir="/tmp/strands")  # noqa: S108

        self.agent = Agent(model=self.bedrock_model, session_manager=self.session_manager, system_prompt=self.system_prompt)

    async def generate_response(self, prompt: Any, llm_context: LLMContext | None = None) -> str:
        """Generate a response from the agent."""
        try:
            response = await self.hass.async_add_executor_job(self.agent, prompt)
            return response.__str__()
        except ClientError as error:
            raise HomeAssistantError(
                f"Amazon Bedrock Error: `{error.response.get('Error').get('Message')}`"
            ) from error
