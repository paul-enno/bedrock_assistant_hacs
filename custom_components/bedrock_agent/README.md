# Amazon Bedrock Agent Integration

The Amazon Bedrock Agent integration brings powerful AI conversation capabilities to Home Assistant using Amazon Bedrock foundation models. Built on the Strands Agents SDK, it provides a sophisticated multi-tier memory system and seamless smart home control.

## Features

### 🤖 AI Conversation Agent
- Integrates with Home Assistant's conversation system and Assist Pipeline
- Supports multiple Amazon Bedrock foundation models
- Customizable system prompts for personalized behavior
- Streaming responses for real-time interaction

### 🧠 Three-Tier Memory System

1. **Short-term Memory (Conversation Manager)**
   - Sliding window of last 40 messages in context
   - Automatic context management
   - Optimized for LLM token limits

2. **Persistent Storage (FileSessionManager)**
   - Full conversation history saved to disk
   - User-isolated storage
   - Survives Home Assistant restarts
   - All conversations per user share the same history

3. **Long-term Semantic Memory (Mem0)**
   - Stores important facts, preferences, and context
   - Semantic search across all conversations
   - Personalized responses based on learned information
   - Powered by FAISS vector database
   - **Configurable memory guidelines** to control what gets stored
   - **Multi-language support** for memory instructions

### 🏠 Home Assistant Control
- Turn devices on/off through natural language
- Query device states and sensor values
- Control lights (brightness, color)
- Manage media players
- Add items to shopping lists
- Execute scenes and scripts
- Full integration with Home Assistant's intent system

### 🖼️ Cognitive Task Service
- Multimodal AI service for image analysis and understanding
- Process multiple images in a single request
- Support for both local files and remote URLs
- Perfect for automation scenarios:
  - Security camera analysis ("Is there a person at the door?")
  - Package delivery detection ("Has a package been delivered?")
  - Pet monitoring ("Is the dog in the backyard?")
  - Plant health assessment ("Does my plant need water?")
  - Visual quality control ("Is the garage door closed?")
- Powered by Claude vision models for accurate image understanding
- Seamlessly integrates with Home Assistant automations
- Returns structured responses for easy automation logic

## Installation

### Requirements
- Home Assistant 2024.1 or later
- AWS account with Bedrock access
- Bedrock model access enabled in AWS console

### Dependencies
The integration automatically installs:
- `boto3==1.39.9` - AWS SDK
- `botocore==1.39.9` - AWS SDK core
- `strands-agents==1.26.0` - Agent framework
- `strands-agents-tools[mem0_memory]==0.1.19` - Memory tools
- `faiss-cpu==1.9.0` - Vector database for semantic memory

## Configuration

### Initial Setup

1. **Add Integration**
   - Go to Settings → Devices & Services
   - Click "Add Integration"
   - Search for "Amazon Bedrock Agent"

2. **AWS Connection**
   - **Name**: Friendly name for your agent
   - **Key ID**: AWS Access Key ID
   - **Key Secret**: AWS Secret Access Key
   - **Region**: AWS region (e.g., us-east-1)

3. **Model Configuration**
   - **Model**: Select from available Bedrock models
   - **Prompt Context**: Custom system prompt (optional)
   - **Enable Home Assistant Control**: Allow device control
   - **Enable Long-term Memory**: Enable Mem0 semantic memory
   - **Memory Storage Path**: Custom path for memory data (optional)
   - **Memory Storage Guidelines**: Customize what information gets stored (optional, only shown when memory is enabled)

### Supported Models

The integration supports all Amazon Bedrock foundation models that are available in your AWS region and enabled in your AWS account. Models include offerings from:
- Amazon Titan
- Anthropic Claude
- AI21 Labs
- Cohere
- Meta Llama
- Mistral AI

To see available models, check the model dropdown during configuration or visit the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).

### Options

After setup, you can modify settings through the integration's options:
- Change the selected model
- Update system prompt
- Toggle Home Assistant control
- Toggle long-term memory
- Change memory storage path
- **Customize memory storage guidelines** (supports any language)

## Usage

### Conversation Agent

The integration registers as a conversation agent in Home Assistant:

```yaml
# Example automation
automation:
  - alias: "Ask Bedrock Agent"
    trigger:
      - platform: state
        entity_id: binary_sensor.front_door
        to: "on"
    action:
      - service: conversation.process
        data:
          agent_id: <your_bedrock_agent_id>
          text: "The front door just opened"
```

### Voice Assistants

Use with Assist Pipeline for voice control:
1. Go to Settings → Voice Assistants
2. Select your voice assistant
3. Choose the Bedrock Agent as the conversation agent

### Home Assistant Control Examples

When Home Assistant control is enabled, you can:

```
"Turn on the kitchen lights"
"Set bedroom lights to 50% brightness"
"What's the temperature in the living room?"
"Turn off all lights in the house"
"Add milk to my shopping list"
"Activate movie night scene"
```

### Cognitive Task Service

Analyze images with AI:

```yaml
service: bedrock_agent.cognitive_task
data:
  prompt: "Describe what you see in these images"
  model_id: "anthropic.claude-3-haiku-20240307-v1:0"
  image_filenames:
    - /config/www/image1.jpg
    - /config/www/image2.jpg
  image_urls:
    - https://example.com/image.jpg
```

### Memory Management Services

#### Clear Conversation Cache
```yaml
service: bedrock_agent.clear_conversation_cache
data:
  conversation_id: "conversation_123"
```

#### Clear All Cache
```yaml
service: bedrock_agent.clear_all_cache
```

#### Get Memory Statistics
```yaml
service: bedrock_agent.get_memory_stats
```

## Architecture

### Session Management

The integration uses a sophisticated session management system:

```
Storage Structure:
/config/.storage/bedrock_agent_sessions/
└── session_<user_id>/
    ├── session.json              # Session metadata
    └── agents/
        └── agent_<user_id>/
            ├── agent.json         # Agent state
            └── messages/
                ├── message_0.json
                ├── message_1.json
                └── ...
```

### Memory Storage

When Mem0 is enabled:

```
/config/.storage/bedrock_agent_memory/
└── bedrock_agent_user_<user_id>/
    ├── index.faiss               # Vector index
    └── metadata.json             # Memory metadata
```

### User Isolation

- Each Home Assistant user gets their own session
- Conversations are isolated per user
- Memory is user-specific
- All conversations for a user share the same history

## Troubleshooting

### Common Issues

**"Cannot connect" error**
- Verify AWS credentials are correct
- Check that the region is valid
- Ensure network connectivity to AWS

**"Invalid auth" error**
- Verify AWS Access Key ID and Secret Access Key
- Check IAM permissions for Bedrock access
- Ensure credentials haven't expired

**"Conversation blocks and tool result blocks cannot be provided in the same turn"**
- This indicates corrupted session history
- The integration will automatically clear cache and retry
- If it persists, manually clear session storage:
  ```bash
  rm -rf /config/.storage/bedrock_agent_sessions/*
  ```
- Then restart Home Assistant
- This can happen after SDK updates or when restoring old sessions

**Agent storing too much in memory (greetings, casual conversation)**
- Edit the "Memory Storage Guidelines" in integration options
- Customize what should and shouldn't be stored
- Default guidelines prevent storing greetings and one-time requests
- Guidelines can be written in any language

**Memory not working**
- Verify `faiss-cpu` is installed
- Check memory storage path is writable
- Review logs for memory-related errors
- Ensure embedder model is available in your region

**Home Assistant control not working**
- Ensure "Enable Home Assistant Control" is checked
- Verify devices are exposed to conversation agent
- Check device names match what you're saying

### Logging

Enable debug logging for troubleshooting:

```yaml
logger:
  default: info
  logs:
    homeassistant.components.bedrock_agent: debug
    strands: debug
```

### Performance Tips

1. **Model Selection**: Lighter models (Titan Lite, Claude Instant) respond faster
2. **Memory**: Disable if not needed to reduce overhead
3. **Context Window**: The 40-message sliding window balances context and performance
4. **Storage**: Use SSD storage for better session/memory performance
5. **Memory Guidelines**: Customize to store only important information, reducing memory overhead
6. **Async Operations**: All blocking I/O runs in executor threads for optimal performance

## Security Considerations

- AWS credentials are stored encrypted in Home Assistant
- Memory data is stored locally, not sent to external services
- User conversations are isolated
- Image processing respects Home Assistant's allowlist

## Support

- **Documentation**: https://www.home-assistant.io/integrations/bedrock_agent
- **Issues**: Report bugs through Home Assistant GitHub
- **Codeowner**: @paul-enno

## License

This integration is part of Home Assistant and follows the Apache 2.0 license.
