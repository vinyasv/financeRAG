"""LLM client abstraction for flexible provider support."""

from abc import ABC, abstractmethod
from typing import Any
import os

# Ensure .env is loaded
from .config import config as _config  # noqa: F401


# OpenRouter popular models
OPENROUTER_MODELS = {
    # Fast & cheap
    "gemini-flash": "google/gemini-2.0-flash-001",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-haiku": "anthropic/claude-3-haiku",
    "llama-8b": "meta-llama/llama-3.1-8b-instruct",
    
    # Balanced
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet": "anthropic/claude-3.5-sonnet",
    "gemini-pro": "google/gemini-pro-1.5",
    "llama-70b": "meta-llama/llama-3.1-70b-instruct",
    
    # Best quality
    "claude-opus": "anthropic/claude-3-opus",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "llama-405b": "meta-llama/llama-3.1-405b-instruct",
    
    # Free (with rate limits)
    "free": "meta-llama/llama-3.1-8b-instruct:free",
}


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass
    
    async def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        image_media_type: str = "image/png"
    ) -> str:
        """
        Generate a response using a vision-capable model with an image.
        
        Args:
            prompt: Text prompt describing what to do with the image
            image_base64: Base64-encoded image data
            image_media_type: MIME type of the image (default: image/png)
            
        Returns:
            LLM response text
        """
        raise NotImplementedError("Vision not supported by this client")


class OpenRouterClient(LLMClient):
    """
    OpenRouter API client - access many models with one API key.
    
    OpenRouter uses OpenAI-compatible API, so we use the OpenAI client
    with a different base URL.
    
    Get your API key at: https://openrouter.ai/keys
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        site_url: str | None = None,
        app_name: str = "Finance RAG"
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model name - can be short name (e.g., "gpt-4o-mini") or 
                   full OpenRouter name (e.g., "openai/gpt-4o-mini")
            site_url: Optional URL of your site for rankings
            app_name: Name of your app for rankings
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.site_url = site_url
        self.app_name = app_name
        self._client = None
        
        # Resolve model name
        if model in OPENROUTER_MODELS:
            self.model = OPENROUTER_MODELS[model]
        else:
            self.model = model
    
    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.BASE_URL
            )
    
    async def generate(self, prompt: str) -> str:
        self._ensure_client()
        
        # Build extra headers for OpenRouter
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            extra_headers["X-Title"] = self.app_name
        
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            extra_headers=extra_headers if extra_headers else None
        )
        
        return response.choices[0].message.content or ""
    
    async def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        image_media_type: str = "image/png",
        model: str | None = None
    ) -> str:
        """
        Generate a response using a vision-capable model with an image.
        
        Args:
            prompt: Text prompt describing what to do with the image
            image_base64: Base64-encoded image data
            image_media_type: MIME type of the image (default: image/png)
            model: Optional model override (must be vision-capable)
            
        Returns:
            LLM response text
        """
        self._ensure_client()
        
        # Use specified model or default vision model
        vision_model = model or os.getenv("VISION_MODEL", "openai/gpt-4o")
        
        # Build extra headers for OpenRouter
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            extra_headers["X-Title"] = self.app_name
        
        # Build message with image using OpenAI vision format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_media_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        response = await self._client.chat.completions.create(
            model=vision_model,
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
            extra_headers=extra_headers if extra_headers else None
        )
        
        return response.choices[0].message.content or ""
    
    @classmethod
    def list_models(cls) -> dict[str, str]:
        """List available model shortcuts."""
        return OPENROUTER_MODELS.copy()


class OpenAIClient(LLMClient):
    """Direct OpenAI API client."""
    
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None
    
    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate(self, prompt: str) -> str:
        self._ensure_client()
        
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    """Direct Anthropic API client."""
    
    def __init__(self, api_key: str | None = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None
    
    def _ensure_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
    
    async def generate(self, prompt: str) -> str:
        self._ensure_client()
        
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text if response.content else ""


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls."""
    
    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = None
    
    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        
        # Check for matching response
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # Default response
        return '{"query": "test", "reasoning": "mock", "steps": []}'


def get_llm_client(
    provider: str = "auto",
    model: str | None = None
) -> LLMClient | None:
    """
    Get an LLM client based on available credentials.
    
    Args:
        provider: "openrouter", "openai", "anthropic", "auto", or "none"
        model: Optional model name override
        
    Returns:
        LLMClient instance or None
    """
    from .config import config
    
    if provider == "none":
        return None
    
    # Use model from config if not explicitly provided
    default_model = model or config.llm_model
    
    if provider == "auto":
        # Try OpenRouter first (most flexible)
        if os.getenv("OPENROUTER_API_KEY"):
            return OpenRouterClient(model=default_model)
        # Then OpenAI
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIClient(model=default_model)
        # Then Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicClient(model=default_model)
        # No API keys available
        return None
    
    if provider == "openrouter":
        return OpenRouterClient(model=default_model)
    
    if provider == "openai":
        return OpenAIClient(model=default_model)
    
    if provider == "anthropic":
        return AnthropicClient(model=default_model)
    
    return None

