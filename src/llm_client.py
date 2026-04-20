import os
import logging
from time import sleep
from typing import Optional
import google.genai as genai
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClientGemini:
    """A wrapper class for handling Gemini LLM inference."""
    
    def __init__(self, 
                 model_checkpoint: str = 'gemini-2.5-pro',
                 max_tokens: int = 32768,
                 thinking_budget: int = 1000,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 sleep_between_calls: float = 60,
                 api_key: Optional[str] = None,
                 debug: bool = False):
        """Initialize the Gemini LLM client."""
        self.model_checkpoint = model_checkpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.top_p = top_p
        self.sleep_between_calls = sleep_between_calls
        self.debug = debug
        self._last_thought = None
        
        if not self.debug:
            if api_key is None:
                api_key = os.environ.get('GOOGLE_API_KEY')
            assert api_key is not None, "Missing GOOGLE_API_KEY"
            
            self.api_key = api_key
            self.client, self.generation_config = self._configure_client()
        else:
            self.client = None

        logger.info(f"Gemini LLM Client initialized with model: {self.model_checkpoint} (debug={self.debug})")

    def _configure_client(self):
        """Set up and return the LLM with the provided configuration."""
        generation_config = genai.types.GenerateContentConfig(
            top_k=50,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            thinking_config=genai.types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=self.thinking_budget > 0
            )
        )
        client = genai.Client(api_key=self.api_key)
        return client, generation_config
    
    def generate(self, prompt: str, do_sleep: bool = True, **kwargs) -> str:
        """Generate a response from the Gemini LLM."""
        if self.debug:
            logger.info("Debug mode: returning dummy response")
            return "Dummy content for testing pipeline functionality."

        logger.info("Running Gemini inference...")
        
        response = self.client.models.generate_content(
            model=self.model_checkpoint,
            contents=prompt,
            config=self.generation_config
        )
        
        thought = None
        response_text = None
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought = part.text.strip()
            else:
                response_text = part.text.strip()
                logger.info(f"LLM response: {response_text}")
        
        self._last_thought = thought
        
        # DEBUG
        if response_text is None:
            if thought:
                logger.warning("Model returned only thought content; using thought as response_text.")
                response_text = thought
            else:
                logger.error("Model returned no textual content at all; using empty string.")
                response_text = ""

        
        if do_sleep and self.sleep_between_calls > 0:
            logger.info(f"Sleeping for: {self.sleep_between_calls}s")
            sleep(self.sleep_between_calls)
            
        return response_text
    
    def generate_and_extract(self, prompt: str, do_sleep: bool = True, **kwargs) -> tuple[str, str]:
        """Generate a response and return both full response and extracted content."""
        response = self.generate(prompt, do_sleep, **kwargs)
        thought = self._last_thought or ""
        
        # DEBUG
        if response is None:
            response = ""
        
        if self.debug:
            full_response = response
            extracted_content = response
        else:
            full_response = f"<think>\n{thought}\n</think>\n\n{response}" if thought else response
            extracted_content = response
        
        return full_response, extracted_content


class LLMClientDeepseek:
    """A wrapper class for handling DeepSeek LLM inference via Together API."""
    
    def __init__(self, 
                 model_checkpoint: str = 'deepseek-ai/DeepSeek-R1',
                 max_tokens: int = 32768,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 sleep_between_calls: float = 60,
                 api_key: Optional[str] = None,
                 debug: bool = False):
        """Initialize the DeepSeek LLM client."""
        self.model_checkpoint = model_checkpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.sleep_between_calls = sleep_between_calls
        self.debug = debug
        
        if not self.debug:
            if api_key is None:
                api_key = os.environ.get('TOGETHER_API_KEY')
            assert api_key is not None, "Missing TOGETHER_API_KEY"
            
            try:
                from together import Together
                self.client = Together(api_key=api_key)
            except ImportError:
                raise ImportError("Please install the 'together' package: pip install together")
        else:
            self.client = None

        logger.info(f"DeepSeek LLM Client initialized with model: {self.model_checkpoint} (debug={self.debug})")
    
    def generate(self, prompt: str, do_sleep: bool = True, **kwargs) -> str:
        """Generate a response from the DeepSeek LLM."""
        if self.debug:
            logger.info("Debug mode: returning dummy response")
            return "<think>\nThis is a dummy response for debugging.\n</think>\n\nDummy content for testing."

        logger.info("Running DeepSeek inference...")
        
        response = self.client.chat.completions.create(
            model=self.model_checkpoint,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"LLM response: {response_text}")
        
        if do_sleep and self.sleep_between_calls > 0:
            logger.info(f"Sleeping for: {self.sleep_between_calls}s")
            sleep(self.sleep_between_calls)
            
        return response_text
    
    def extract_content_after_think(self, response: str) -> str:
        """Extract content after the </think> tag."""
        return response.split('</think>')[-1].strip()
    
    def generate_and_extract(self, prompt: str, do_sleep: bool = True, **kwargs) -> tuple[str, str]:
        """Generate a response and extract content after </think> tag."""
        full_response = self.generate(prompt, do_sleep, **kwargs)
        extracted_content = self.extract_content_after_think(full_response)
        return full_response, extracted_content


class PromptManager:
    """Helper class for managing prompts and templates."""
    
    @staticmethod
    def load_prompt(prompt_path: str) -> str:
        """Load a prompt from file."""
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_prompts(prompt_dir: str, prompt_files: list[str]) -> dict[str, str]:
        """Load multiple prompts from a directory."""
        prompts = {}
        for filename in prompt_files:
            filepath = os.path.join(prompt_dir, filename)
            key = os.path.splitext(filename)[0]  # Remove extension for key
            prompts[key] = PromptManager.load_prompt(filepath)
        return prompts
    
    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """Format a prompt template with the given kwargs."""
        return template.format(**kwargs)


class LLMClientOpenAI:
    """A wrapper class for handling OpenAI LLM inference with common configuration and retry logic."""
    
    def __init__(self, 
                 model_checkpoint: str = 'o4-mini',
                 max_tokens: int = 32768,
                 reasoning_effort: str = "medium",
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 sleep_between_calls: float = 60,
                 api_key: Optional[str] = None,
                 debug: bool = False):
        """
        Initialize the OpenAI LLM client.
        
        Args:
            model_checkpoint: The model to use for inference (e.g., 'o4-mini', 'gpt-4o')
            max_tokens: Maximum tokens for generation
            reasoning_effort: Reasoning effort level for o-series models ("low", "medium", "high")
            temperature: Temperature for sampling (optional for o-series models)
            top_p: Top-p for nucleus sampling (optional for o-series models)
            sleep_between_calls: Sleep time between API calls
            api_key: API key (if None, will try to get from environment)
            debug: Whether to run in debug mode
        """
        self.model_checkpoint = model_checkpoint
        self.max_output_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.sleep_between_calls = sleep_between_calls
        self.debug = debug
        self.verbose = False  # Add verbose attribute for compatibility
        self._last_thought = None  # Store last thought for generate_and_extract
        
        # Check if this is a reasoning model to determine parameter support
        self.is_reasoning_model = any(model_prefix in model_checkpoint.lower() for model_prefix in ['o1', 'o4', 'o3'])
        
        # Only store temperature and top_p for non-reasoning models
        if not self.is_reasoning_model:
            self.temperature = temperature
            self.top_p = top_p
        else:
            self.temperature = None
            self.top_p = None
        
        if not self.debug:
            # Get API key from parameter or environment
            if api_key is None:
                api_key = os.environ.get('OPENAI_API_KEY')
            assert api_key is not None, "Missing OPENAI_API_KEY"
            
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None  # No need for real client in debug mode

        logger.info(f"OpenAI LLM Client initialized with model: {self.model_checkpoint} (debug={self.debug})")
    
    def generate(self, 
                 prompt: str, 
                 do_sleep: bool = True,
                 custom_max_tokens: Optional[int] = None,
                 custom_temperature: Optional[float] = None,
                 custom_top_p: Optional[float] = None,
                 custom_reasoning_effort: Optional[str] = None) -> str:
        """
        Generate a response from the OpenAI LLM.
        
        Args:
            prompt: The input prompt
            do_sleep: Whether to sleep after the call
            custom_max_tokens: Override max_tokens for this call
            custom_temperature: Override temperature for this call
            custom_top_p: Override top_p for this call
            custom_reasoning_effort: Override reasoning effort for this call
            
        Returns:
            The generated response content
        """

        if self.debug:
            logger.info("Debug mode: returning dummy response")
            return "<think>\nThis is a dummy response for debugging purposes.\n</think>\n\nDummy content for testing pipeline functionality."

        logger.info("Running OpenAI inference...")
        
        # Use custom parameters if provided, otherwise use defaults
        max_output_tokens = custom_max_tokens if custom_max_tokens is not None else self.max_output_tokens
        reasoning_effort = custom_reasoning_effort if custom_reasoning_effort is not None else self.reasoning_effort
        
        # For reasoning models, ignore temperature and top_p parameters completely
        if self.is_reasoning_model:
            temperature = None
            top_p = None
        else:
            temperature = custom_temperature if custom_temperature is not None else self.temperature
            top_p = custom_top_p if custom_top_p is not None else self.top_p
        
        # Build the request parameters
        request_params = {
            "model": self.model_checkpoint,
            "input": [{"role": "user", "content": prompt}],
        }
        
        # Add reasoning config only for o-series models
        if self.is_reasoning_model and reasoning_effort:
            request_params["reasoning"] = {"effort": reasoning_effort}
        
        # Add generation parameters if specified
        if max_output_tokens is not None:
            request_params["max_output_tokens"] = max_output_tokens
        
        # Only add temperature and top_p for non-reasoning models
        if not self.is_reasoning_model:
            if temperature is not None:
                request_params["temperature"] = temperature
            if top_p is not None:
                request_params["top_p"] = top_p
        
        response = self.client.responses.create(**request_params)
        
        # Extract response text and reasoning (if available)
        response_text = response.output_text
        reasoning_text = getattr(response, 'reasoning_text', None)
        
        # Store reasoning for potential use in generate_and_extract
        self._last_thought = reasoning_text
        
        logger.info(f"OpenAI LLM response: {response_text}")
        
        if do_sleep and self.sleep_between_calls > 0:
            logger.info(f"Sleeping for: {self.sleep_between_calls}s")
            sleep(self.sleep_between_calls)
            
        return response_text
    
    def extract_content_after_think(self, response: str) -> str:
        """
        Extract content after the </think> tag for compatibility with DeepSeek interface
        
        Args:
            response: The full LLM response
            
        Returns:
            Content after the </think> tag, stripped of whitespace
        """
        return response.split('</think>')[-1].strip()
    
    def generate_and_extract(self, 
                           prompt: str, 
                           do_sleep: bool = True,
                           **kwargs) -> tuple[str, str]:
        """
        Generate a response and extract content after </think> tag.
        
        Args:
            prompt: The input prompt
            do_sleep: Whether to sleep after the call
            **kwargs: Additional parameters for generate()
            
        Returns:
            Tuple of (full_response, extracted_content)
        """
        response = self.generate(prompt, do_sleep, **kwargs)
        reasoning = self._last_thought or ""
        
        if self.debug:
            # In debug mode, create the full response format for consistency
            full_response = response  # Already contains <think> tags in debug mode
            extracted_content = self.extract_content_after_think(response)
        else:
            # In real mode, reconstruct full response if we have reasoning
            full_response = f"<think>\n{reasoning}\n</think>\n\n{response}" if reasoning else response
            extracted_content = response  # Response is already the extracted content
        
        return full_response, extracted_content