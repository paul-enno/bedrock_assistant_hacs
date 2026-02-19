"""Service handlers for Bedrock integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from botocore.exceptions import ClientError
from strands.types.content import ContentBlock
import voluptuous as vol

from homeassistant.core import ServiceCall, ServiceResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .const import (
    CONST_SERVICE_PARAM_FILENAMES,
    CONST_SERVICE_PARAM_IMAGE_URLS,
    CONST_SERVICE_PARAM_MODEL_ID,
    CONST_SERVICE_PARAM_PROMPT,
)
from .image_processor import ImageProcessor, build_converse_prompt_content

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .agent import BedrockAgent


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


class CognitiveTaskService:
    """Handle cognitive task service calls."""

    def __init__(self, hass: HomeAssistant, bedrock_agent: BedrockAgent) -> None:
        """Initialize the service handler."""
        self.hass = hass
        self.bedrock_agent = bedrock_agent
        self.image_processor = ImageProcessor(hass)

    async def async_handle_cognitive_task(self, call: ServiceCall) -> ServiceResponse:
        """Return answer to prompt and description of image."""
        param_model_id = call.data.get(
            CONST_SERVICE_PARAM_MODEL_ID, "us.anthropic.claude-sonnet-4-20250514-v1:0"
        )

        agent = await self.bedrock_agent.strands_agent_wrapper.get_simple_agent(
            param_model_id
        )

        param_prompt = str(call.data.get(CONST_SERVICE_PARAM_PROMPT))
        prompt_content: list[ContentBlock] = [{"text": param_prompt}]

        # Process image files
        image_filenames = call.data.get(CONST_SERVICE_PARAM_FILENAMES)
        for image_filename in image_filenames or []:
            file_image = await self.image_processor.load_image_from_file(image_filename)
            image_content = await build_converse_prompt_content(file_image)
            prompt_content.append(image_content)  # type: ignore[arg-type]

        # Process image URLs
        param_image_urls = call.data.get(CONST_SERVICE_PARAM_IMAGE_URLS)
        for param_image_url in param_image_urls or []:
            url_image = await self.image_processor.load_image_from_url(param_image_url)
            url_content = await build_converse_prompt_content(url_image)
            prompt_content.append(url_content)  # type: ignore[arg-type]

        try:
            result = agent(prompt_content)
        except ClientError as error:
            raise HomeAssistantError(
                f"Bedrock Error: `{error.response.get('Error').get('Message')}`"
            ) from error

        return {"text": f"{result}"}
