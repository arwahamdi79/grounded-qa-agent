"""Factory for the chat model, so provider choice is a single switch."""
import config


def get_chat_model(temperature: float = 0.0):
    if config.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.CHAT_MODEL_ANTHROPIC,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    # default: openai
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.CHAT_MODEL_OPENAI,
        api_key=config.OPENAI_API_KEY,
        temperature=temperature,
    )
