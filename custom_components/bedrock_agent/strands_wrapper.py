"""Wrapper for strands.Agent to make it easier to test and implement."""

from functools import partial
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
logging.getLogger("strands").setLevel(logging.ERROR)

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
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.system_prompt = system_prompt

        self.tools = []
        self.apis = apis
        self.api_instances = {}
        self.llm_context = None
        self.modules = {}
        # self.system_prompt = system_prompt

        # # Create a boto3 session
        # session = boto3.Session(
        #     aws_access_key_id=aws_access_key_id,
        #     aws_secret_access_key=aws_secret_access_key,
        #     region_name=region_name
        # )

        # # Create a Bedrock model with the custom session
        # self.bedrock_model = BedrockModel(
        #     model_id=model_id,
        #     boto_session=session
        # )

        # self.user = str(hass.auth.async_get_user)

        # self.session_manager = FileSessionManager(session_id="enno-123", storage_dir="/tmp/strands")

        # self.agent = Agent(model=self.bedrock_model, session_manager=self.session_manager, system_prompt=self.system_prompt)

        self.agent = self.get_agent(model_id, True, True)


    def get_agent(self,
        model_id: str,
        withSession: bool,
        withSystemPrompt: bool) -> Agent:
        """Initalize Agent."""

        # Create a boto3 session
        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )

        # Create a Bedrock model with the custom session
        bedrock_model = BedrockModel(
            model_id=model_id,
            boto_session=session,
            streaming=False,
        )

        session_manager = FileSessionManager(session_id="enno-123", storage_dir="/tmp/strands")  # noqa: S108
        if(withSession and withSystemPrompt):
            return Agent(model=bedrock_model, session_manager=session_manager, system_prompt=self.system_prompt, callback_handler=None)
        if(withSession):
            return Agent(model=bedrock_model, session_manager=session_manager, callback_handler=None)
        if(withSystemPrompt):
            return Agent(model=bedrock_model, system_prompt=self.system_prompt, callback_handler=None)

        return Agent(model=bedrock_model, callback_handler=None)

    async def generate_response(self, prompt: Any, llm_context: LLMContext | None = None) -> str:
        """Generate a response from the agent."""
        try:
            response = await self.hass.async_add_executor_job(self.agent, prompt)
            # return response.__str__()
            return str(response)
        except ClientError as error:
            raise HomeAssistantError(
                f"Amazon Bedrock Error: `{error.response.get('Error').get('Message')}`"
            ) from error

    async def async_call_llm(self, prompt: str, llm_context: LLMContext) -> str:
        """Call the agent with the given prompt."""
        _LOGGER.debug("Calling LLM with prompt: %s", prompt)
        try:
            response = await self.hass.async_add_executor_job(self.agent, prompt)
            # return response.__str__()
            return str(response)
        except ClientError as error:
            raise HomeAssistantError(
                f"Amazon Bedrock Error: `{error.response.get('Error').get('Message')}`"
            ) from error
