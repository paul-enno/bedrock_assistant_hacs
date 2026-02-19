"""Wrapper for strands.Agent to make it easier to test and implement."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import BedrockModel
from strands.session import FileSessionManager

from homeassistant.exceptions import HomeAssistantError

from .ha_control_tool import create_ha_control_tool

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.llm import API, LLMContext

    from .aws_client import AWSClientFactory

# Configure the root strands logger
logging.getLogger("strands").setLevel(logging.ERROR)

_LOGGER = logging.getLogger(__name__)

# Try to import mem0_memory tool and check dependencies
_mem0_available = False
_mem0_error_message = None

try:
    from strands_tools import mem0_memory

    # Check if faiss is available (required by mem0)
    try:
        import faiss  # noqa: F401

        _mem0_available = True
    except ImportError:
        _mem0_error_message = (
            "Faiss-cpu not available. Install with: pip install faiss-cpu"
        )
        _LOGGER.warning(
            "Mem0_memory tool requires faiss-cpu. Install with: pip install faiss-cpu"
        )
except ImportError:
    _mem0_error_message = "Mem0_memory tool not available. Install with: pip install 'strands-agents-tools[mem0_memory]'"
    _LOGGER.warning(
        "Mem0_memory tool not available. Install with: pip install 'strands-agents-tools[mem0_memory]'"
    )


class StrandsAgentWrapper:
    """Wrapper for strands.Agent to make it easier to test and implement."""

    def __init__(
        self,
        hass: HomeAssistant,
        aws_factory: AWSClientFactory,
        model_id: str,
        apis: list[API],
        system_prompt: str | None = "",
        enable_memory: bool = True,
        enable_ha_control: bool = True,
        memory_storage_path: str = "",
        memory_guidelines: str = "",
        user_id: str | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            hass: Home Assistant instance
            aws_factory: AWS client factory
            model_id: Bedrock model ID
            apis: List of Home Assistant APIs
            system_prompt: System prompt for the agent
            enable_memory: Enable long-term memory with mem0
            enable_ha_control: Enable Home Assistant device control
            memory_storage_path: Custom path for memory storage (empty = use default)
            memory_guidelines: Custom guidelines for memory storage (empty = use default)
            user_id: User ID for memory isolation
        """
        self.hass = hass
        self.aws_factory = aws_factory
        self.system_prompt = system_prompt
        self.model_id = model_id
        self.enable_memory = enable_memory and _mem0_available
        self.enable_ha_control = enable_ha_control
        self.memory_guidelines = memory_guidelines
        self.user_id = user_id or "default_user"

        # Set memory storage path - use default if not provided
        if memory_storage_path:
            self.memory_storage_path = memory_storage_path
        else:
            # Default to Home Assistant's storage directory
            self.memory_storage_path = os.path.join(
                hass.config.path(".storage"), "bedrock_agent_memory"
            )

        # Set session storage path (separate from memory storage)
        self.session_storage_path = os.path.join(
            hass.config.path(".storage"), "bedrock_agent_sessions"
        )

        self.tools = []
        self.apis = apis
        self.api_instances: dict[str, Any] = {}
        self.llm_context = None
        self.modules: dict[str, Any] = {}

        # Configure AWS credentials for mem0 if memory is enabled
        if self.enable_memory:
            self._configure_mem0_credentials()
            self.tools.append(mem0_memory)
            _LOGGER.info("Mem0 memory enabled for long-term semantic memory")
        elif not _mem0_available:
            _LOGGER.warning("Memory disabled: mem0_memory tool not available")

        # Log Home Assistant control status
        if self.enable_ha_control:
            if self.apis:
                _LOGGER.info(
                    "Home Assistant control enabled with %d APIs", len(self.apis)
                )
            else:
                _LOGGER.warning("Home Assistant control enabled but no APIs available")
        else:
            _LOGGER.info("Home Assistant control disabled by configuration")

        # Ensure session storage directory exists
        try:
            os.makedirs(self.session_storage_path, exist_ok=True)
            _LOGGER.info("Session storage directory: %s", self.session_storage_path)
        except OSError as err:
            _LOGGER.error(
                "Failed to create session storage directory %s: %s",
                self.session_storage_path,
                err,
            )

        # Cache of agents per conversation ID and user ID
        self._agent_cache: dict[str, Agent] = {}

        # Cache of session managers per user ID
        self._session_managers: dict[str, FileSessionManager] = {}

        # Default agent (created on first use)
        self.agent: Agent | None = None

    def _configure_mem0_credentials(self) -> None:
        """Configure AWS credentials and storage for mem0 via environment variables.

        Mem0 uses environment variables for AWS credentials and FAISS configuration.
        We set them here to ensure mem0 can access Bedrock for embeddings and LLM operations,
        and stores data in the configured location.
        
        IMPORTANT: We always override environment variables to ensure consistency
        between dev and production environments.
        """
        # Always set AWS credentials (override any existing values)
        os.environ["AWS_ACCESS_KEY_ID"] = self.aws_factory.aws_access_key_id
        _LOGGER.debug("Set AWS_ACCESS_KEY_ID for mem0")

        os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_factory.aws_secret_access_key
        _LOGGER.debug("Set AWS_SECRET_ACCESS_KEY for mem0")

        os.environ["AWS_REGION"] = self.aws_factory.region_name
        _LOGGER.debug("Set AWS_REGION for mem0: %s", self.aws_factory.region_name)

        # Always configure mem0 to use Bedrock for embeddings and LLM
        os.environ["MEM0_EMBEDDER_PROVIDER"] = "aws_bedrock"

        # Use Titan Embed Text v1 which is more widely available
        os.environ["MEM0_EMBEDDER_MODEL"] = "amazon.titan-embed-text-v1"

        os.environ["MEM0_LLM_PROVIDER"] = "aws_bedrock"

        # Use the same model as configured for the main agent
        # This ensures consistency and that the model is available
        os.environ["MEM0_LLM_MODEL"] = self.model_id
        _LOGGER.debug("Using configured model for mem0 LLM: %s", self.model_id)

        # Configure FAISS storage path
        os.environ["MEM0_VECTOR_STORE_PATH"] = self.memory_storage_path
        _LOGGER.debug("Set MEM0_VECTOR_STORE_PATH: %s", self.memory_storage_path)

        # Ensure the storage directory exists
        try:
            os.makedirs(self.memory_storage_path, exist_ok=True)
            _LOGGER.info("Memory storage directory: %s", self.memory_storage_path)
        except OSError as err:
            _LOGGER.error(
                "Failed to create memory storage directory %s: %s",
                self.memory_storage_path,
                err,
            )

        _LOGGER.info(
            "Configured mem0 with AWS Bedrock: embedder=%s, llm=%s, region=%s, storage=%s",
            os.environ.get("MEM0_EMBEDDER_MODEL"),
            os.environ.get("MEM0_LLM_MODEL"),
            os.environ.get("AWS_REGION"),
            self.memory_storage_path,
        )

        _LOGGER.info(
            "Note: Mem0 environment variables are set from integration config. "
            "If you need different models, update the integration configuration."
        )

    async def _create_bedrock_model(self) -> BedrockModel:
        """Create a Bedrock model instance.

        Runs in executor because BedrockModel initialization loads botocore
        service definitions from disk (blocking I/O).
        """
        def _create_model() -> BedrockModel:
            """Create model in executor."""
            session = self.aws_factory.create_boto3_session()
            return BedrockModel(
                model_id=self.model_id,
                boto_session=session,
                streaming=False,
            )

        return await self.hass.async_add_executor_job(_create_model)

    async def _get_session_manager(self, user_id: str) -> FileSessionManager:
        """Get or create a FileSessionManager for a specific user.

        Each user gets their own session identified by user_id.
        All conversations for that user share the same agent within the session,
        so conversation history persists across all interactions.

        This structure allows:
        - User isolation: Each user's data is completely separate
        - Persistent conversation history: All messages persist across conversations
        - Proper session persistence: Messages are automatically loaded by the session manager

        Args:
            user_id: Home Assistant user ID (used as session_id)

        Returns:
            FileSessionManager instance for this user
        """
        # Use user_id as the session_id for proper user isolation
        if user_id not in self._session_managers:
            _LOGGER.debug(
                "Creating FileSessionManager for user %s (session_id=%s) in %s",
                user_id,
                user_id,
                self.session_storage_path,
            )

            # Create the session manager in executor to avoid blocking I/O
            # FileSessionManager reads session.json during initialization
            def _create_session_manager() -> FileSessionManager:
                """Create session manager in executor."""
                return FileSessionManager(
                    session_id=user_id,
                    storage_dir=self.session_storage_path,
                )

            session_manager = await self.hass.async_add_executor_job(
                _create_session_manager
            )

            self._session_managers[user_id] = session_manager

        return self._session_managers[user_id]

    async def get_simple_agent(self, model_id: str | None = None) -> Agent:
        """Get or create a simple agent without session persistence.

        This is used for one-off tasks like the cognitive task service.

        Args:
            model_id: Optional model ID to use (defaults to wrapper's model_id)

        Returns:
            Agent instance without session persistence
        """
        # Create a simple agent without session persistence
        bedrock_model = await self._create_bedrock_model()

        return Agent(
            model=bedrock_model,
            system_prompt=self.system_prompt or "",
            callback_handler=None,
        )

    def _get_enhanced_system_prompt(
        self, user_id: str | None = None, has_ha_control: bool = False
    ) -> str:
        """Get system prompt enhanced with memory and HA control instructions.

        Args:
            user_id: Optional user ID to include in memory instructions
            has_ha_control: Whether Home Assistant control is available

        Returns:
            Enhanced system prompt with memory and HA control instructions
        """
        base_prompt = self.system_prompt or ""

        enhancements = []

        # Add memory instructions if enabled
        if self.enable_memory:
            effective_user_id = user_id or self.user_id

            # Build memory instructions
            memory_instruction = f"""

You have access to a long-term memory system that persists across conversations. Use the memory tool to:
- Store important information about the user (preferences, facts, context)
- Retrieve relevant memories to provide personalized responses
- Remember user preferences and past interactions

IMPORTANT: When using the memory tool, always use user_id="{effective_user_id}" to ensure memories are stored and retrieved for the correct user.
"""

            # Add custom memory guidelines if provided
            if self.memory_guidelines:
                memory_instruction += f"\n{self.memory_guidelines}"

            enhancements.append(memory_instruction)

        # Add Home Assistant control instructions if available
        if has_ha_control:
            enhancements.append("""

You have access to Home Assistant smart home control through the homeassistant_control tool.

CRITICAL RULES FOR USING homeassistant_control:
1. The tool requires TWO parameters for most operations:
   - tool_name: The intent name (e.g., "HassTurnOn", "HassGetState")
   - name: The device name (e.g., "kitchen light", "bedroom fan")

2. ALWAYS provide the 'name' parameter when using these intents:
   - Device control: HassTurnOn, HassTurnOff, HassToggle, HassGetState, HassLightSet, HassSetPosition
   - Media control: HassMediaUnpause, HassMediaPause, HassMediaNext, HassMediaPrevious, HassSetVolume
   - Shopping lists: HassListAddItem, HassListRemoveItem (name = list name, e.g., "Shopping List")

3. Only these intents work WITHOUT a 'name' parameter:
   - GetLiveContext (shows all devices)
   - GetDateTime (shows current time)

4. Optional parameters:
   - domain: Device type (e.g., "light", "switch", "fan") - helps identify the right device
   - brightness: For lights, 0-100
   - color: For lights, color name or value

5. SPECIAL CASES:
   - Scenes: Use the scene name directly as a tool if available, OR use HassTurnOn with the full entity_id (e.g., name="scene.ha_new")
   - Scripts: Use the script name directly as a tool if available
   - If a scene/script tool exists with the exact name, prefer using that tool directly

CORRECT EXAMPLES:
✓ homeassistant_control(tool_name="HassTurnOn", name="kitchen light", domain="light")
✓ homeassistant_control(tool_name="HassGetState", name="living room temperature")
✓ homeassistant_control(tool_name="HassListAddItem", name="Shopping List", item="milk")
✓ homeassistant_control(tool_name="GetLiveContext")
✓ homeassistant_control(tool_name="ha_new") - if ha_new is a scene/script tool
✓ homeassistant_control(tool_name="HassTurnOn", name="scene.ha_new") - activate scene by entity_id

WRONG EXAMPLES:
✗ homeassistant_control(tool_name="HassTurnOn") - Missing 'name' parameter!
✗ homeassistant_control(tool_name="HassTurnOn", domain="light") - Still missing 'name'!

If you get an error about "cannot target all devices", it means you forgot to provide the 'name' parameter.
If you get an error about "Failed to call turn_on", the device might not support that action - try checking available tools with GetLiveContext.""")

        return base_prompt + "".join(enhancements)

    async def get_agent_with_memory(
        self, conversation_id: str, user_id: str, llm_context: LLMContext | None = None
    ) -> Agent:
        """Get or create an agent with dual memory system and session persistence.

        This agent uses a sophisticated dual-memory architecture:

        1. SHORT-TERM MEMORY (Conversation Manager):
           - SlidingWindowConversationManager keeps last 40 interactions in context
           - Provides immediate conversational context for the LLM
           - Automatically managed by the agent

        2. LONG-TERM MEMORY (mem0):
           - Semantic memory that persists across ALL conversations
           - Stores important facts, preferences, and context
           - Accessible via mem0_memory tool (if enabled)

        3. PERSISTENT STORAGE (FileSessionManager):
           - Full conversation history saved to disk per user
           - User-isolated storage (session_id = user_id)
           - Agent identified by user_id (agent_id = user_id)
           - Survives Home Assistant restarts
           - All conversations for a user share the same history

        Args:
            conversation_id: Unique identifier for the conversation (not used for persistence)
            user_id: Home Assistant user ID for isolation and persistence
            llm_context: LLM context for Home Assistant control

        Returns:
            Agent instance with dual memory, session persistence, and HA control
        """
        # Use user_id as cache key since all conversations for a user share the same agent
        cache_key = user_id

        # Return cached agent if it exists
        if cache_key in self._agent_cache:
            _LOGGER.debug(
                "Using cached agent for user: %s (conversation: %s)",
                user_id,
                conversation_id,
            )
            return self._agent_cache[cache_key]

        # Create new agent with session manager, mem0 memory, and HA control
        _LOGGER.debug(
            "Creating new agent with session persistence for user: %s (conversation: %s)",
            user_id,
            conversation_id,
        )

        bedrock_model = await self._create_bedrock_model()

        # Get or create session manager for this user
        # session_id = user_id, agent_id = user_id
        # This means all conversations for a user share the same persistent history
        session_manager = await self._get_session_manager(user_id)

        # Create conversation manager with sliding window (defaults to 40 messages)
        # This keeps only the last 40 interactions in the agent's context
        # while FileSessionManager persists the full history to disk
        conversation_manager = SlidingWindowConversationManager()
        _LOGGER.debug(
            "Created SlidingWindowConversationManager for user %s (default 40 message window)",
            user_id,
        )

        # Build tools list
        agent_tools = list(self.tools)  # Start with mem0_memory if enabled

        # Add Home Assistant control tool if enabled, APIs available, and llm_context provided
        if self.enable_ha_control and self.apis and llm_context:
            ha_tool = await create_ha_control_tool(self.hass, self.apis, llm_context)
            agent_tools.append(ha_tool)
            _LOGGER.debug("Added Home Assistant control tool to agent")

        # Create agent with:
        # - agent_id: Use user_id as the agent identifier (persistent across all conversations)
        # - session_manager: FileSessionManager for persistent storage (automatically loads messages)
        # - conversation_manager: SlidingWindowConversationManager for 40-message context window
        # - tools: mem0_memory (long-term semantic memory) + HA control
        #
        # IMPORTANT: The session manager's initialize() method is called automatically
        # when the agent is created, and it loads existing messages from disk if they exist.
        # We do NOT need to manually load messages - the session manager handles this.
        # All conversations for this user will share the same message history.
        #
        # NOTE: Agent creation is done in executor because FileSessionManager does blocking
        # I/O operations (os.listdir) during initialization to load existing messages.
        system_prompt = self._get_enhanced_system_prompt(
            user_id,
            has_ha_control=bool(self.enable_ha_control and self.apis and llm_context),
        )

        def _create_agent() -> Agent:
            """Create agent in executor to avoid blocking I/O."""
            return Agent(
                agent_id=user_id,  # Use user_id as agent_id for persistent history
                model=bedrock_model,
                tools=agent_tools,
                system_prompt=system_prompt,
                session_manager=session_manager,
                conversation_manager=conversation_manager,
                callback_handler=None,
            )

        agent = await self.hass.async_add_executor_job(_create_agent)

        # Cache the agent by user_id only (not conversation_id)
        self._agent_cache[cache_key] = agent

        _LOGGER.info(
            "Created agent for user %s: "
            "session_id=%s, agent_id=%s, storage=%s, context window=40 messages, mem0=%s, tools=%d",
            user_id,
            user_id,
            user_id,
            session_manager.storage_dir,
            self.enable_memory,
            len(agent_tools),
        )

        return agent

    def clear_conversation_cache(self, conversation_id: str) -> None:
        """Clear cached agents and session managers for a specific conversation.

        Note: Since agents are now cached by user_id only (not conversation_id),
        this method has limited effect. To clear a user's agent, use clear_user_cache().

        This clears the in-memory cache but does NOT delete the persisted session data.
        Session data remains on disk and will be loaded when the user interacts again.

        Note: This also does not clear mem0 memories, which persist across all conversations.

        Args:
            conversation_id: Unique identifier for the conversation (deprecated for agent clearing)
        """
        _LOGGER.debug(
            "clear_conversation_cache called for conversation %s, but agents are cached by user_id. "
            "Use clear_user_cache() to clear a specific user's agent",
            conversation_id,
        )

    def clear_user_cache(self, user_id: str) -> None:
        """Clear cached agent and session manager for a specific user.

        This clears the in-memory cache but does NOT delete the persisted session data.
        Session data remains on disk and will be loaded when the user interacts again.

        Note: This also does not clear mem0 memories, which persist across all conversations.

        Args:
            user_id: Home Assistant user ID
        """
        if user_id in self._agent_cache:
            del self._agent_cache[user_id]
            _LOGGER.debug("Cleared agent cache for user: %s", user_id)

        if user_id in self._session_managers:
            del self._session_managers[user_id]
            _LOGGER.debug("Cleared session manager cache for user: %s", user_id)

        if user_id in self._agent_cache or user_id in self._session_managers:
            _LOGGER.info("Cleared cache for user %s", user_id)

    def clear_all_cache(self) -> None:
        """Clear all cached agents and session managers.

        This clears the in-memory cache but does NOT delete persisted session data.
        Session data remains on disk and will be loaded when conversations resume.

        Note: This also does not clear mem0 memories.
        """
        agent_count = len(self._agent_cache)
        manager_count = len(self._session_managers)

        _LOGGER.debug(
            "Clearing all caches: %d agents, %d session managers",
            agent_count,
            manager_count,
        )

        self._agent_cache.clear()
        self._session_managers.clear()

        _LOGGER.info(
            "Cleared all caches: %d agents, %d session managers",
            agent_count,
            manager_count,
        )

    async def generate_response(
        self,
        prompt: Any,
        llm_context: LLMContext | None = None,
        conversation_id: str | None = None,
        context_user_id: str | None = None,
    ) -> str:
        """Generate a response from the agent.

        When memory is enabled, the agent will automatically:
        - Store important information from the conversation
        - Retrieve relevant memories to provide context
        - Maintain long-term memory across all conversations

        Args:
            prompt: The prompt to send to the agent
            llm_context: Optional LLM context
            conversation_id: Optional conversation ID for agent caching and session persistence
            context_user_id: Optional user ID from Home Assistant context

        Returns:
            The agent's response as a string
        """
        try:
            # Use agent with session persistence if conversation_id provided
            # This includes mem0 memory if enabled, plus FileSessionManager and sliding window
            if conversation_id:
                # Use context user_id if available, otherwise fall back to wrapper user_id
                effective_user_id = context_user_id or self.user_id
                agent = await self.get_agent_with_memory(
                    conversation_id, effective_user_id, llm_context
                )
                _LOGGER.debug(
                    "Using agent with session persistence for user: %s, conversation: %s, mem0: %s",
                    effective_user_id,
                    conversation_id,
                    self.enable_memory,
                )
            else:
                # Create default agent without session persistence if not cached
                if self.agent is None:
                    bedrock_model = await self._create_bedrock_model()

                    # Build tools list for default agent
                    agent_tools = []
                    if self.enable_ha_control and self.apis and llm_context:
                        ha_tool = await create_ha_control_tool(
                            self.hass, self.apis, llm_context
                        )
                        agent_tools.append(ha_tool)

                    self.agent = Agent(
                        model=bedrock_model,
                        tools=agent_tools,
                        system_prompt=self._get_enhanced_system_prompt(
                            has_ha_control=bool(
                                self.enable_ha_control and self.apis and llm_context
                            )
                        ),
                        callback_handler=None,
                    )
                agent = self.agent
                _LOGGER.debug("Using agent without session persistence")

            # Call the agent asynchronously using invoke_async
            # For agents with memory (mem0), run in executor to avoid blocking
            # Mem0 operations can be slow and may block the event loop
            if self.enable_memory and conversation_id:
                _LOGGER.debug("Running agent with memory in executor to avoid blocking")
                try:
                    response = await self.hass.async_add_executor_job(
                        agent, prompt
                    )
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    error_msg = e.response.get("Error", {}).get("Message", "")

                    # Handle specific Bedrock validation errors
                    if error_code == "ValidationException" and "cannot be provided in the same turn" in error_msg:
                        _LOGGER.warning(
                            "Session history has invalid message structure. Clearing cache and retrying. "
                            "This can happen after SDK updates or when restoring old sessions"
                        )
                        # Clear the problematic agent from cache
                        if effective_user_id in self._agent_cache:
                            del self._agent_cache[effective_user_id]
                        if effective_user_id in self._session_managers:
                            del self._session_managers[effective_user_id]

                        # Create a fresh agent and retry
                        agent = await self.get_agent_with_memory(
                            conversation_id, effective_user_id, llm_context
                        )
                        response = await self.hass.async_add_executor_job(
                            agent, prompt
                        )
                    else:
                        raise
                except Exception:
                    _LOGGER.exception("Error during agent execution with memory")
                    raise
            else:
                # This keeps us in the same event loop as Home Assistant
                response = await agent.invoke_async(prompt)

            # Extract text from the response message
            # AgentResult.message.content is a list of ContentBlock objects
            if hasattr(response, "message") and response.message:
                content_blocks = response.message.get("content", [])
                text_parts = [
                    block.get("text", "")
                    for block in content_blocks
                    if isinstance(block, dict) and "text" in block
                ]
                return "".join(text_parts) if text_parts else str(response)

            # Fallback to string conversion
            return str(response)
        except ClientError as error:
            raise HomeAssistantError(
                f"Amazon Bedrock Error: `{error.response.get('Error').get('Message')}`"
            ) from error

    async def async_call_llm(
        self,
        prompt: str,
        llm_context: LLMContext,
        conversation_id: str | None = None,
        context_user_id: str | None = None,
    ) -> str:
        """Call the agent with the given prompt.

        Args:
            prompt: The prompt to send to the agent
            llm_context: LLM context
            conversation_id: Optional conversation ID for agent caching
            context_user_id: Optional user ID from Home Assistant context

        Returns:
            The agent's response as a string
        """
        _LOGGER.debug("Calling LLM with prompt: %s", prompt)
        return await self.generate_response(
            prompt, llm_context, conversation_id, context_user_id
        )

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory and session statistics.

        Returns:
            Dictionary with memory and session statistics
        """
        stats = {
            "memory_enabled": self.enable_memory,
            "mem0_available": _mem0_available,
            "user_id": self.user_id,
            "cached_agents": len(self._agent_cache),
            "cached_session_managers": len(self._session_managers),
            "tools_count": len(self.tools),
            "session_storage_path": self.session_storage_path,
            "memory_storage_path": self.memory_storage_path,
        }

        # Add error message if mem0 is not available
        if not _mem0_available and _mem0_error_message:
            stats["error"] = _mem0_error_message

        return stats
