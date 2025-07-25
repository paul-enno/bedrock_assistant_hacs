"""The Bedrock Agent integration."""

from __future__ import annotations

from functools import partial
from io import BytesIO
import logging
import mimetypes
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.request import urlopen
import uuid

import boto3
from botocore.exceptions import ClientError
import PIL.Image
from PIL.Image import Image
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.components.conversation import agent_manager
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.intent import IntentResponse, IntentResponseErrorCode

from .const import (
    CONST_AGENT_ALIAS_ID,
    CONST_AGENT_ID,
    CONST_KEY_ID,
    CONST_KEY_SECRET,
    CONST_KNOWLEDGEBASE_ID,
    CONST_MODEL_ID,
    CONST_MODEL_LIST,
    CONST_PROMPT_CONTEXT,
    CONST_REGION,
    CONST_SERVICE_PARAM_FILENAMES,
    CONST_SERVICE_PARAM_IMAGE_URLS,
    CONST_SERVICE_PARAM_MODEL_ID,
    CONST_SERVICE_PARAM_PROMPT,
    DOMAIN,
)
from .strands_wrapper import StrandsAgentWrapper

_LOGGER = logging.getLogger(__name__)
__all__ = [
    "BedrockAgent",
    "async_process",
    "async_setup_entry",
    "async_unload_entry",
    "options_update_listener",
]


# Example migration function
async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug("Migrating from version %s", config_entry.version)
    if config_entry.version == 1:
        hass.config_entries.async_update_entry(
            config_entry, data=config_entry.data, minor_version=1, version=2
        )

    return True


async def build_converse_prompt_content(image: Image) -> Any:
    buffered = BytesIO()
    image.save(buffered, format=image.format)
    file_image_byte = buffered.getvalue()
    file_image_format = (
        image.format if image.format in ["jpeg", "png", "gif", "webp"] else "jpeg"
    )
    return {
        "image": {
            "format": file_image_format,
            "source": {"bytes": file_image_byte},
        },
    }


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Bedrock Agent from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    conversation.async_set_agent(hass, entry, BedrockAgent(hass, entry))

    hass_data = dict(entry.data)
    unsub_options_update_listener = entry.add_update_listener(options_update_listener)
    # Store a reference to the unsubscribe function to cleanup if an entry is unloaded.
    hass_data["unsub_options_update_listener"] = unsub_options_update_listener
    hass.data[DOMAIN][entry.entry_id] = hass_data

    async def async_cognitive_task(call: ServiceCall) -> ServiceResponse:
        """Return answer to prompt and description of image."""
        param_model_id = call.data.get(
            CONST_SERVICE_PARAM_MODEL_ID, "us.anthropic.claude-sonnet-4-20250514-v1:0"
        )

        strands_agent_wrapper = StrandsAgentWrapper(
            hass=hass,
            aws_access_key_id=entry.data[CONST_KEY_ID],
            aws_secret_access_key=entry.data[CONST_KEY_SECRET],
            region_name=entry.data[CONST_REGION],
            model_id=param_model_id,
            apis = llm.async_get_apis(hass)
        )

        param_prompt = call.data.get(CONST_SERVICE_PARAM_PROMPT)
        prompt_content = [{"text": param_prompt}]
        image_filenames = call.data.get(CONST_SERVICE_PARAM_FILENAMES)
        for image_filename in image_filenames or []:
            if not hass.config.is_allowed_path(image_filename):
                raise HomeAssistantError(
                    f"Cannot read `{image_filename}`, no access to path; "
                    "`allowlist_external_dirs` may need to be adjusted in "
                    "`configuration.yaml`"
                )
            if not Path(image_filename).exists():
                raise HomeAssistantError(f"`{image_filename}` does not exist")
            mime_type, _ = mimetypes.guess_type(image_filename)
            if mime_type is None or not mime_type.startswith("image"):
                raise HomeAssistantError(f"`{image_filename}` is not an image")
            file_image = await hass.async_add_executor_job(
                PIL.Image.open, image_filename
            )
            prompt_content.append(await build_converse_prompt_content(file_image))

        param_image_urls = call.data.get(CONST_SERVICE_PARAM_IMAGE_URLS)
        for param_image_url in param_image_urls or []:
            try:
                mime_type, _ = mimetypes.guess_type(param_image_url)
                if mime_type is None or not mime_type.startswith("image"):
                    raise HomeAssistantError(f"`{param_image_url}` is not an image")
                opened_url = await hass.async_add_executor_job(urlopen, param_image_url)
                url_image = PIL.Image.open(opened_url)
                prompt_content.append(await build_converse_prompt_content(url_image))
            except HTTPError as error:  # status reason
                raise HomeAssistantError(
                    f"Cannot access file from `{param_image_url}`."
                    f"Status: `{error.status}`, Reason: `{error.reason}`"
                ) from error

        try:
            result = await strands_agent_wrapper.generate_response(prompt_content)
        except ClientError as error:
            raise HomeAssistantError(
                f"Bedrock Error: `{error.response.get('Error').get('Message')}`"
            ) from error

        description = (
            result
            #bedrock_response["output"]["message"].get("content")[0].get("text")
        )

        return {"text": f"{description}"}

    COGNITIVE_TASK_SCHEMA = vol.Schema(
        {
            vol.Required(CONST_SERVICE_PARAM_PROMPT): str,
            vol.Optional(CONST_SERVICE_PARAM_MODEL_ID): str,
            vol.Optional(CONST_SERVICE_PARAM_IMAGE_URLS, default=[]): vol.All(
                cv.ensure_list, [cv.string]
            ),
            vol.Optional(CONST_SERVICE_PARAM_FILENAMES, default=[]): vol.All(
                cv.ensure_list, [cv.string]
            ),
        }
    )

    hass.services.async_register(
        DOMAIN,
        "cognitive_task",
        async_cognitive_task,
        schema=COGNITIVE_TASK_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    conversation.async_unset_agent(hass, entry)
    return True


async def options_update_listener(hass: HomeAssistant, config_entry: ConfigEntry):
    """Handle options update."""
    # await hass.config_entries.async_reload(config_entry.entry_id)


class BedrockAgent(conversation.AbstractConversationAgent):
    """Bedrock conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}

        # self.bedrock_agent = self.hass.async_add_executor_job(
        #         partial(boto3.client,
        #                 service_name="bedrock-agent-runtime",
        #                 region_name=self.entry.data[CONST_REGION],
        #                 aws_access_key_id=self.entry.data[CONST_KEY_ID],
        #                 aws_secret_access_key=self.entry.data[CONST_KEY_SECRET]))

        self.bedrock_agent = boto3.client(
            service_name="bedrock-agent-runtime",
            region_name=self.entry.data[CONST_REGION],
            aws_access_key_id=self.entry.data[CONST_KEY_ID],
            aws_secret_access_key=self.entry.data[CONST_KEY_SECRET],
        )

        # Initialize the strands agent wrapper

        self.strands_agent_wrapper = StrandsAgentWrapper(
            hass=self.hass,
            aws_access_key_id=self.entry.data[CONST_KEY_ID],
            aws_secret_access_key=self.entry.data[CONST_KEY_SECRET],
            region_name=self.entry.data[CONST_REGION],
            model_id=self.entry.options[CONST_MODEL_ID],
            apis = llm.async_get_apis(self.hass),
            system_prompt = self.entry.options[CONST_PROMPT_CONTEXT]
        )


    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @staticmethod
    def supported_models() -> list[str]:
        """Return a list of supported models."""
        return CONST_MODEL_LIST

    async def async_call_bedrock(
        self, userInput: agent_manager.ConversationInput
    ) -> str:
        """Return result from Amazon Bedrock."""
        inital_question = userInput.text
        question = self.entry.options[CONST_PROMPT_CONTEXT] + userInput.text

        # modelId = self.entry.options[CONST_MODEL_ID]
        # knowledgebaseId = self.entry.options.get(CONST_KNOWLEDGEBASE_ID) or ""
        configAgentId = self.entry.options.get(CONST_AGENT_ID) or ""
        configAgentAliasId = (
            self.entry.options.get(CONST_AGENT_ALIAS_ID) or "TSTALIASID"
        )

        if configAgentId != "":
            bedrock_agent_response = await self.hass.async_add_executor_job(
                partial(
                    self.bedrock_agent.invoke_agent,
                    agentId=configAgentId,
                    agentAliasId=configAgentAliasId,
                    sessionId=userInput.conversation_id,
                    inputText=question,
                ),
            )

            completion = ""

            for event in bedrock_agent_response.get("completion"):
                chunk = event["chunk"]
                completion = completion + chunk["bytes"].decode()

            return completion

        return await self.strands_agent_wrapper.generate_response(inital_question,  userInput.as_llm_context(DOMAIN))

    async def async_process(
        self,
        user_input: agent_manager.ConversationInput,
    ) -> agent_manager.ConversationResult:
        """Process a sentence."""
        response = IntentResponse(language=user_input.language)

        user_input.conversation_id = user_input.conversation_id or uuid.uuid4().hex

        try:
            answer = await self.async_call_bedrock(user_input)
            # Add the response to the chat log.

            response.async_set_speech(answer)
        except HomeAssistantError as error:
            response.async_set_error(
                IntentResponseErrorCode.FAILED_TO_HANDLE, error.args[0]
            )

        return agent_manager.ConversationResult(
            conversation_id=user_input.conversation_id,
            response=response,
        )
