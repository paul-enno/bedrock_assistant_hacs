# Amazon Bedrock Conversation Agent for HACS

# v1.5
Added support for Bedrock Agents. 

If there is an Agent with AgentAlias it will use this to answer questions. If there is no Alias definied, it will use the latest draft.

If there is no Agent defined, it will use the selected knowledge base. 

If there is no knowledge base selected, it will use the selected LLM directly.