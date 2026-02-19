"""Home Assistant control tool for Strands Agent."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from strands.tools.decorator import tool

from homeassistant.helpers.llm import ToolInput as HAToolInput

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.llm import API, APIInstance, LLMContext, Tool

_LOGGER = logging.getLogger(__name__)


class HAToolRegistry:
    """Registry for Home Assistant tools."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self.api_instances: dict[str, APIInstance] = {}
        self.tools_by_name: dict[
            str, tuple[Tool, str]
        ] = {}  # tool_name -> (tool, api_id)

    async def async_load_apis(
        self,
        hass: HomeAssistant,
        apis: list[API],
        llm_context: LLMContext,
    ) -> None:
        """Load all API instances and their tools.

        Args:
            hass: Home Assistant instance
            apis: List of Home Assistant LLM APIs
            llm_context: LLM context
        """
        self.api_instances.clear()
        self.tools_by_name.clear()

        for api in apis:
            try:
                api_instance = await api.async_get_api_instance(llm_context)
                self.api_instances[api.id] = api_instance

                # Register all tools from this API
                for ha_tool in api_instance.tools:
                    self.tools_by_name[ha_tool.name] = (ha_tool, api.id)
                    _LOGGER.debug(
                        "Registered HA tool: %s from API: %s", ha_tool.name, api.id
                    )

                _LOGGER.debug(
                    "Loaded API: %s with %d tools", api.id, len(api_instance.tools)
                )

            except Exception as err:  # noqa: BLE001
                _LOGGER.warning("Failed to load API %s: %s", api.id, err)

        _LOGGER.info(
            "Loaded %d Home Assistant tools from %d APIs",
            len(self.tools_by_name),
            len(self.api_instances),
        )

    def get_tool_descriptions(self) -> str:
        """Get descriptions of all available tools.

        Returns:
            Formatted string with tool descriptions
        """
        if not self.tools_by_name:
            return "No Home Assistant tools available"

        descriptions = ["Available Home Assistant tools:"]
        for tool_name, (ha_tool, _api_id) in sorted(self.tools_by_name.items()):
            desc = ha_tool.description or "No description"
            descriptions.append(f"- {tool_name}: {desc}")

        return "\n".join(descriptions)

    async def async_call_tool(
        self,
        hass: HomeAssistant,
        tool_name: str,
        tool_args: dict[str, Any],
        llm_context: LLMContext,
    ) -> dict[str, Any]:
        """Call a Home Assistant tool by name.

        Args:
            hass: Home Assistant instance
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            llm_context: LLM context

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools_by_name:
            available = ", ".join(sorted(self.tools_by_name.keys()))
            return {
                "error": f"Tool '{tool_name}' not found. Available tools: {available}"
            }

        ha_tool, api_id = self.tools_by_name[tool_name]

        try:
            _LOGGER.debug(
                "Calling HA tool %s from API %s with args: %s",
                tool_name,
                api_id,
                tool_args,
            )

            # Create tool input
            tool_input = HAToolInput(
                tool_name=tool_name,
                tool_args=tool_args,
            )

            # Call the HA tool
            result = await ha_tool.async_call(hass, tool_input, llm_context)

            _LOGGER.debug("HA tool %s result: %s", tool_name, result)

            # Return result
            if isinstance(result, dict):
                return result
            return {"result": str(result)}

        except Exception as err:
            _LOGGER.exception("Error calling HA tool %s", tool_name)

            # Provide helpful error messages for common issues
            error_msg = str(err)
            if "Failed to call turn_on" in error_msg and "scene." in error_msg:
                # Extract scene name from error
                scene_match = re.search(r"scene\.(\w+)", error_msg)
                scene_name = scene_match.group(1) if scene_match else "unknown"

                # Check if there's a direct tool for this scene
                if scene_name in self.tools_by_name:
                    return {
                        "error": f"Cannot use HassTurnOn for scenes. Use tool_name='{scene_name}' directly instead."
                    }
                return {
                    "error": "Scenes cannot be activated with HassTurnOn. Try using the scene name as tool_name directly, or check available tools with GetLiveContext."
                }

            return {"error": error_msg}


async def create_ha_control_tool(
    hass: HomeAssistant,
    apis: list[API],
    llm_context: LLMContext,
) -> Any:
    """Create a single Strands tool that dispatches to Home Assistant tools.

    Args:
        hass: Home Assistant instance
        apis: List of Home Assistant LLM APIs
        llm_context: LLM context

    Returns:
        Strands-compatible tool function
    """
    # Create and load the registry
    registry = HAToolRegistry()
    await registry.async_load_apis(hass, apis, llm_context)

    # Build tool description
    available_tools = "\n".join(
        f"- {tool_name}: {tool.description or 'No description'}"
        for tool_name, (tool, _) in sorted(registry.tools_by_name.items())
    )

    @tool(
        name="homeassistant_control",
        description=f"""Control Home Assistant devices and query their state.

This tool dispatches to specific Home Assistant intents. You MUST provide:
1. tool_name: The intent name (e.g., 'HassTurnOn', 'HassGetState', 'HassListAddItem')
2. name: The device/list name (REQUIRED for most intents)
3. Additional parameters based on the intent type

Available tools:
{available_tools}

Examples:
- Turn on: tool_name='HassTurnOn', name='kitchen light', domain='light'
- Get state: tool_name='HassGetState', name='bedroom temperature'
- Add to list: tool_name='HassListAddItem', name='Shopping List', item='milk'
- List all: tool_name='GetLiveContext'""",
    )
    async def homeassistant_control(
        tool_name: str,
        name: str = "",
        domain: str = "",
        brightness: int | None = None,
        color: str = "",
        item: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a Home Assistant tool.

        Args:
            tool_name: Name of the HA tool/intent (e.g., 'HassTurnOn', 'HassGetState', 'HassListAddItem')
            name: Device/list name to control (REQUIRED for most intents)
            domain: Device domain (e.g., 'light', 'switch', 'fan')
            brightness: Light brightness 0-100 (for HassLightSet)
            color: Color name or value (for HassLightSet)
            item: Item to add/remove (for HassListAddItem, HassListRemoveItem)
            **kwargs: Additional parameters for specific intents
        """
        _LOGGER.info(
            "Homeassistant_control called: tool_name=%s, name='%s', domain='%s', brightness=%s, color='%s', kwargs=%s",
            tool_name,
            name,
            domain,
            brightness,
            color,
            kwargs,
        )

        # Validate that name is provided for intents that require it
        intents_requiring_name = {
            # Device control intents
            "HassTurnOn",
            "HassTurnOff",
            "HassToggle",
            "HassGetState",
            "HassLightSet",
            "HassSetPosition",
            "HassMediaUnpause",
            "HassMediaPause",
            "HassMediaNext",
            "HassMediaPrevious",
            "HassSetVolume",
            # Shopping list / todo intents
            "HassListAddItem",
            "HassListRemoveItem",
        }

        if tool_name in intents_requiring_name and not name:
            error_msg = f"Intent '{tool_name}' requires a 'name' parameter. "

            # Provide specific guidance based on intent type
            if tool_name.startswith("HassList"):
                error_msg += "For shopping list intents, provide the list name (e.g., name='Shopping List')."
            else:
                error_msg += "For device control, provide the device name (e.g., name='kitchen light')."

            _LOGGER.error(error_msg)
            return {"error": error_msg}

        # Build tool args
        tool_args = {**kwargs}
        if name:
            tool_args["name"] = name
        if domain:
            tool_args["domain"] = domain
        if brightness is not None:
            tool_args["brightness"] = brightness
        if color:
            tool_args["color"] = color
        if item:
            tool_args["item"] = item

        _LOGGER.info("Calling HA tool %s with args: %s", tool_name, tool_args)

        return await registry.async_call_tool(
            hass,
            tool_name,
            tool_args,
            llm_context,
        )

    return homeassistant_control


# Tool specification for documentation
TOOL_SPEC = {
    "name": "homeassistant_control",
    "description": (
        "Control Home Assistant devices and query their state. "
        "This tool provides access to all Home Assistant intents and actions."
    ),
}
