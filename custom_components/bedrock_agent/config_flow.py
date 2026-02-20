"""Config flow for Amazon Bedrock Agent integration."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import logging
from typing import Any

import boto3
from botocore.exceptions import EndpointConnectionError
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import selector

from .const import (
    CONST_ENABLE_HA_CONTROL,
    CONST_ENABLE_MEMORY,
    CONST_KEY_ID,
    CONST_KEY_SECRET,
    CONST_MEMORY_GUIDELINES,
    CONST_MEMORY_STORAGE_PATH,
    CONST_MODEL_ID,
    CONST_MODEL_LIST,
    CONST_PROMPT_CONTEXT,
    CONST_REGION,
    CONST_TITLE,
    DEFAULT_MEMORY_GUIDELINES,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONST_TITLE): str,
        vol.Required(CONST_REGION): str,
        vol.Required(CONST_KEY_ID): str,
        vol.Required(CONST_KEY_SECRET): str,
    }
)

STEP_MODELCONFIG_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(
            CONST_PROMPT_CONTEXT,
            default="Provide me a short answer to the following question: ",
        ): str,
        vol.Required(CONST_MODEL_ID): selector.SelectSelector(
            selector.SelectSelectorConfig(options=CONST_MODEL_LIST),
        ),
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""

    bedrock = boto3.client(
        service_name="bedrock",
        region_name=data.get(CONST_REGION),
        aws_access_key_id=data.get(CONST_KEY_ID),
        aws_secret_access_key=data.get(CONST_KEY_SECRET),
    )

    response = None

    try:
        response = await hass.async_add_executor_job(bedrock.list_foundation_models)
    except EndpointConnectionError as err:
        _LOGGER.exception("Unable to connect to AWS Endpoint")
        raise CannotConnect from err
    except bedrock.exceptions.ClientError as err:
        _LOGGER.exception("Unable to authenticate against AWS Endpoint")
        raise InvalidAuth from err
    except Exception as err:  # pylint: disable=broad-except
        _LOGGER.exception("Unexpected exception")
        raise HomeAssistantError from err

    if response is None or response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        raise CannotConnect

    return {"title": "Bedrock"}


async def get_foundation_models_select_option_dict(
    hass: HomeAssistant, data: dict[str, Any]
) -> Sequence[selector.SelectOptionDict]:
    """Load available foundation models."""
    bedrock = boto3.client(
        service_name="bedrock",
        region_name=data.get(CONST_REGION),
        aws_access_key_id=data.get(CONST_KEY_ID),
        aws_secret_access_key=data.get(CONST_KEY_SECRET),
    )

    model_response = await hass.async_add_executor_job(
        partial(
            bedrock.list_foundation_models,
            byOutputModality="TEXT",
            byInferenceType="ON_DEMAND",
        )
    )

    models = model_response.get("modelSummaries")
    models.sort(
        key=lambda m: (
            m.get("providerName", "").lower(),
            m.get("modelName", "").lower(),
        )
    )
    template = "{model_provider} - {model_name}"

    model_select_options = [
        selector.SelectOptionDict(
            {
                "value": m.get("modelId"),
                "label": template.format(
                    model_provider=m.get("providerName"), model_name=m.get("modelName")
                ),
            }
        )
        for m in models
    ]

    profiles_response = await hass.async_add_executor_job(
        partial(bedrock.list_inference_profiles)
    )
    profiles = profiles_response.get("inferenceProfileSummaries")
    profiles.sort(key=lambda p: p.get("inferenceProfileName").lower())
    template = "{profileName}"
    profile_select_options = [
        selector.SelectOptionDict(
            {
                "value": p.get("inferenceProfileId"),
                "label": template.format(profileName=p.get("inferenceProfileName")),
            }
        )
        for p in profiles
    ]

    return model_select_options + profile_select_options


class BedrockAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Amazon Bedrock Agent."""

    VERSION = 2
    MINOR_VERSION = 1

    def __init__(self) -> None:
        """Initialize options flow."""
        self.config_data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                await validate_input(self.hass, user_input)
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except HomeAssistantError:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                self.config_data.update(user_input)
                return await self.async_step_modelconfig()

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    async def async_step_modelconfig(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""

        foundation_models = await get_foundation_models_select_option_dict(
            self.hass, self.config_data
        )

        modelconfig_schema = vol.Schema(
            {
                vol.Optional(
                    CONST_PROMPT_CONTEXT,
                    default="Provide me a short answer to the following question: ",
                ): str,
                vol.Required(CONST_MODEL_ID): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=foundation_models),
                ),
                vol.Optional(
                    CONST_ENABLE_HA_CONTROL,
                    default=True,
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONST_ENABLE_MEMORY,
                    default=True,
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONST_MEMORY_STORAGE_PATH,
                    default="",
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.TEXT, multiline=False
                    )
                ),
            }
        )

        errors: dict[str, str] = {}
        if user_input is not None:
            return self.async_create_entry(
                title=self.config_data.get(CONST_TITLE, "Bedrock"),
                data=self.config_data,
                options=user_input,
            )

        return self.async_show_form(
            step_id="modelconfig",
            data_schema=modelconfig_schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OptionsFlowHandler()


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class InvalidAuth(HomeAssistantError):
    """Error to indicate there is invalid auth."""


class OptionsFlowHandler(OptionsFlow):
    """Handle a options flow for Amazon Bedrock Agent."""

    def __init__(self) -> None:
        """Initialize options flow."""
        self._options: dict[str, Any] = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options - show menu."""
        # If user selected "done", close the flow
        if user_input is not None and user_input.get("next_step_id") == "done":
            return self.async_create_entry(title="", data=self.config_entry.options)
        
        return self.async_show_menu(
            step_id="init",
            menu_options=["aws_config", "ai_config", "memory_config", "tools_config"],
        )

    async def async_step_aws_config(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure AWS credentials and region."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Validate the new credentials
            try:
                await validate_input(self.hass, user_input)
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except HomeAssistantError:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                # Update config entry data with new credentials
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={
                        **self.config_entry.data,
                        CONST_KEY_ID: user_input[CONST_KEY_ID],
                        CONST_KEY_SECRET: user_input[CONST_KEY_SECRET],
                        CONST_REGION: user_input[CONST_REGION],
                    },
                )
                
                # Reload the integration to apply new credentials
                await self.hass.config_entries.async_reload(self.config_entry.entry_id)
                
                # Navigate to AI config to ensure model is valid for new region
                return await self.async_step_ai_config()

        # Get current values
        current_key_id = self.config_entry.data.get(CONST_KEY_ID, "")
        current_region = self.config_entry.data.get(CONST_REGION, "")

        # Region first, then credentials
        aws_schema = vol.Schema(
            {
                vol.Required(
                    CONST_REGION,
                    default=current_region,
                ): str,
                vol.Required(
                    CONST_KEY_ID,
                    default=current_key_id,
                ): str,
                vol.Required(
                    CONST_KEY_SECRET,
                ): str,
            }
        )

        return self.async_show_form(
            step_id="aws_config",
            data_schema=aws_schema,
            errors=errors,
            description_placeholders={
                "aws_info": "Update AWS credentials and region. After saving, you'll need to verify/update the AI model for the new region."
            },
        )

    async def async_step_ai_config(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure AI model settings."""
        if user_input is not None:
            # Update options with AI config
            self._options.update(user_input)
            # Save and return to menu
            return await self._update_options()

        foundation_models = await get_foundation_models_select_option_dict(
            self.hass, self.config_entry.data.copy()
        )

        ai_schema = vol.Schema(
            {
                vol.Required(
                    CONST_MODEL_ID,
                    default=self.config_entry.options.get(CONST_MODEL_ID),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=foundation_models),
                ),
                vol.Required(
                    CONST_PROMPT_CONTEXT,
                    default=self.config_entry.options.get(CONST_PROMPT_CONTEXT),
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.TEXT, multiline=True
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="ai_config",
            data_schema=ai_schema,
            description_placeholders={
                "model_info": "Select the AI model to use for conversations. The model list has been updated based on your AWS region."
            },
        )

    async def async_step_memory_config(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure memory settings."""
        if user_input is not None:
            # Update options with memory config
            self._options.update(user_input)
            # Save and return to menu
            return await self._update_options()

        # Get current memory enabled state
        current_memory_enabled = self.config_entry.options.get(CONST_ENABLE_MEMORY, True)

        # Build schema
        schema_dict = {
            vol.Optional(
                CONST_ENABLE_MEMORY,
                default=current_memory_enabled,
            ): selector.BooleanSelector(),
            vol.Optional(
                CONST_MEMORY_STORAGE_PATH,
                default=self.config_entry.options.get(
                    CONST_MEMORY_STORAGE_PATH, ""
                ),
            ): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.TEXT, multiline=False
                )
            ),
        }

        # Add memory guidelines field only if memory is enabled
        if current_memory_enabled:
            schema_dict[vol.Optional(
                CONST_MEMORY_GUIDELINES,
                default=self.config_entry.options.get(
                    CONST_MEMORY_GUIDELINES, DEFAULT_MEMORY_GUIDELINES
                ),
            )] = selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.TEXT, multiline=True
                )
            )

        memory_schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="memory_config",
            data_schema=memory_schema,
            description_placeholders={
                "memory_info": "Configure memory and enhancement settings"
            },
        )

    async def async_step_tools_config(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure tool settings."""
        if user_input is not None:
            # Update options with tools config
            self._options.update(user_input)
            # Save and return to menu
            return await self._update_options()

        tools_schema = vol.Schema(
            {
                vol.Optional(
                    CONST_ENABLE_HA_CONTROL,
                    default=self.config_entry.options.get(
                        CONST_ENABLE_HA_CONTROL, True
                    ),
                ): selector.BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="tools_config",
            data_schema=tools_schema,
            description_placeholders={
                "tools_info": "Enable or disable Home Assistant control tools"
            },
        )

    async def _update_options(self) -> ConfigFlowResult:
        """Update config entry options and return to menu."""
        # Merge current options with new options
        new_options = {**self.config_entry.options, **self._options}
        
        # Update the config entry
        self.hass.config_entries.async_update_entry(
            self.config_entry, options=new_options
        )
        
        # Reload the integration to apply new options
        await self.hass.config_entries.async_reload(self.config_entry.entry_id)
        
        # Clear the temporary options
        self._options = {}
        
        # Return to menu so user can configure other sections
        return await self.async_step_init()
