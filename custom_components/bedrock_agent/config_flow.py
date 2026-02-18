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
    CONST_MEMORY_STORAGE_PATH,
    CONST_MODEL_ID,
    CONST_MODEL_LIST,
    CONST_PROMPT_CONTEXT,
    CONST_REGION,
    CONST_TITLE,
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

    # def __init__(self, config_entry: ConfigEntry) -> None:
    #     """Initialize options flow."""
    #     self.config_entry = config_entry

    def __init__(self) -> None:
        """Initialize options flow."""
        self._conf_app_id: str | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Optionsflow to edit model configuration."""
        foundation_models = await get_foundation_models_select_option_dict(
            self.hass, self.config_entry.data.copy()
        )

        options_schema = vol.Schema(
            {
                vol.Required(
                    CONST_PROMPT_CONTEXT,
                    default=self.config_entry.options.get(CONST_PROMPT_CONTEXT),
                ): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.TEXT, multiline=True
                    )
                ),
                vol.Required(
                    CONST_MODEL_ID,
                    default=self.config_entry.options.get(CONST_MODEL_ID),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=foundation_models),
                ),
                vol.Optional(
                    CONST_ENABLE_HA_CONTROL,
                    default=self.config_entry.options.get(
                        CONST_ENABLE_HA_CONTROL, True
                    ),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONST_ENABLE_MEMORY,
                    default=self.config_entry.options.get(CONST_ENABLE_MEMORY, True),
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
        )

        if user_input is not None:
            return self.async_create_entry(
                title=self.config_entry.title, data=user_input
            )

        return self.async_show_form(step_id="init", data_schema=options_schema)
