"""Constants for the Amazon Bedrock Agent integration."""

from typing import Final

DOMAIN: Final = "bedrock_agent"

CONST_KEY_ID: Final = "key_id"
CONST_KEY_SECRET: Final = "key_secret"
CONST_REGION: Final = "region"
CONST_MODEL_ID: Final = "model_id"
CONST_TITLE: Final = "title"

CONST_PROMPT_CONTEXT: Final = "prompt_context"
CONST_ENABLE_MEMORY: Final = "enable_memory"
CONST_ENABLE_HA_CONTROL: Final = "enable_ha_control"
CONST_MEMORY_STORAGE_PATH: Final = "memory_storage_path"
CONST_MEMORY_GUIDELINES: Final = "memory_guidelines"
CONST_MODEL_LIST: Final = [
    "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1",
    "anthropic.claude-v2",
    "anthropic.claude-v2:1",
    "anthropic.claude-instant-v1",
    "ai21.j2-mid-v1",
    "ai21.j2-ultra-v1",
    "cohere.command-text-v14",
    "cohere.command-light-text-v14",
    "cohere.command-r-v1:0",
    "cohere.command-r-plus-v1:0",
    "meta.llama2-13b-chat-v1",
    "meta.llama2-70b-chat-v1",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-small-2402-v1:0",
]

CONST_SERVICE_PARAM_PROMPT: Final = "prompt"
CONST_SERVICE_PARAM_MODEL_ID: Final = "model_id"
CONST_SERVICE_PARAM_IMAGE_URLS: Final = "image_urls"
CONST_SERVICE_PARAM_FILENAMES: Final = "image_filenames"

# Default memory guidelines for selective storage
DEFAULT_MEMORY_GUIDELINES: Final = """MEMORY STORAGE GUIDELINES - Be Selective:
STORE in memory:
- User preferences (e.g., "prefers dark mode", "likes Italian food")
- Important facts about the user (e.g., "has a dog named Max", "lives in Seattle")
- Significant decisions or plans (e.g., "planning trip to Japan in spring")
- Recurring requests or patterns (e.g., "always asks for weather at 7am")
- Home automation preferences (e.g., "bedroom lights should be 50% brightness at night")
- Important context that should persist (e.g., "works from home on Mondays")

DO NOT STORE in memory:
- Greetings and casual conversation (e.g., "hello", "how are you", "thanks")
- Simple questions without context (e.g., "what's the weather?")
- One-time requests (e.g., "turn on the kitchen light")
- Temporary information (e.g., "I'm going to the store now")
- General knowledge questions (e.g., "what is the capital of France?")
- System status checks (e.g., "what devices are available?")

When users share important information, proactively store it in memory. When answering questions, retrieve relevant memories to provide contextual, personalized responses."""

