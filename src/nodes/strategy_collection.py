"""Strategy collection node for gathering agent strategies."""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base import AsyncParallelBatchNode, ContextKeys, validate_context
from src.core.models import Agent, StrategyRecord, GameResult, RoundSummary
from src.core.api_client import OpenRouterClient
from src.core.config import Config
from src.core.prompts import STRATEGY_COLLECTION_PROMPT, format_round_summary, apply_model_variations
from src.core.model_adapters import ModelAdapterFactory, FallbackHandler

logger = logging.getLogger(__name__)


class StrategyCollectionNode(AsyncParallelBatchNode[Agent, Optional[StrategyRecord]]):
    """Collect strategies from all agents in parallel using Gemini."""
    
    def __init__(self, api_client: OpenRouterClient, config: Config, rate_limiter=None):
        """Initialize with API client and config.
        
        Args:
            api_client: OpenRouter API client
            config: Experiment configuration
            rate_limiter: Optional rate limiter for API calls (ModelRateLimiter instance)
        """
        super().__init__()
        self.api_client = api_client
        self.config = config
        self.rate_limiter = rate_limiter
        self.timeout = 30.0  # 30 second timeout per API call
        
        # Create ModelRateLimiter if multi-model is enabled and none provided
        if not self.rate_limiter and hasattr(config, 'ENABLE_MULTI_MODEL') and config.ENABLE_MULTI_MODEL:
            from src.utils.rate_limiter import ModelRateLimiter
            self.rate_limiter = ModelRateLimiter()
    
    async def _apply_rate_limiting(self, model: Optional[str], adapter: Optional['ModelAdapter']) -> None:
        """Apply appropriate rate limiting based on available components.
        
        Args:
            model: Model type string
            adapter: Model adapter instance
        """
        if self.rate_limiter:
            if hasattr(self.rate_limiter, 'acquire'):
                # Check if it's a ModelRateLimiter (has model-specific acquire)
                import inspect
                sig = inspect.signature(self.rate_limiter.acquire)
                if 'model_type' in sig.parameters:
                    # ModelRateLimiter - use model or default
                    await self.rate_limiter.acquire(model or self.config.MAIN_MODEL)
                else:
                    # Simple rate limiter
                    await self.rate_limiter.acquire()
        elif adapter:
            # Use adapter's rate limiting
            await adapter.enforce_rate_limit(self.rate_limiter)
    
    def _extract_model_version(self, model_config: Optional['ModelConfig']) -> Optional[str]:
        """Extract model version from model configuration.
        
        Args:
            model_config: Model configuration object
            
        Returns:
            Model version string or None
        """
        if not model_config or not hasattr(model_config, 'model_type'):
            return None
            
        model_type = model_config.model_type
        
        # Handle different version patterns
        # Pattern 1: date versions like claude-3-sonnet-20240229
        import re
        date_match = re.search(r'-(\d{8})$', model_type)
        if date_match:
            return date_match.group(1)
        
        # Pattern 2: semantic versions like gpt-4-0613
        semantic_match = re.search(r'-(\d+\.\d+|\d{4})$', model_type)
        if semantic_match:
            return semantic_match.group(1)
        
        # Pattern 3: version in path like openai/gpt-4-turbo-1106
        if '/' in model_type:
            parts = model_type.split('/')[-1].split('-')
            for part in reversed(parts):
                if part.replace('.', '').isdigit() and len(part) >= 3:
                    return part
        
        return None
    
    async def process_item(self, agent: Agent) -> Optional[StrategyRecord]:
        """Process single agent strategy collection with timeout and rate limiting.
        
        Args:
            agent: Agent to collect strategy from
            
        Returns:
            StrategyRecord or None if failed
        """
        if not hasattr(self, 'context'):
            raise RuntimeError("Context not set. This method should be called from execute()")
            
        context = self.context  # Access from parent execute
        round_num = context[ContextKeys.ROUND]
        round_summaries = context.get(ContextKeys.ROUND_SUMMARIES, [])
        
        try:
            # Determine model type for prompt building
            model_type = None
            if agent.model_config is not None:
                model_type = agent.model_config.model_type
            
            prompt = self.build_prompt(agent, round_num, round_summaries, model_type)
            messages = [{"role": "user", "content": prompt}]
            
            # Determine model and adapter to use
            model = self.config.MAIN_MODEL
            adapter = None
            
            # Check if agent has model_config (multi-model enabled)
            if agent.model_config is not None:
                adapter = ModelAdapterFactory.get_adapter(agent.model_config)
                model = agent.model_config.model_type
            
            # Apply rate limiting - model-aware if using ModelRateLimiter
            await self._apply_rate_limiting(model, adapter)
            
            # Track API call timing
            start_time = time.time()
            
            # Model parameters to track
            model_params = {
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # Use asyncio.wait_for to enforce timeout
            if adapter:
                # Use adapter-aware API call
                response = await asyncio.wait_for(
                    self.api_client.complete(
                        messages=messages,
                        model=model,
                        **model_params,
                        adapter=adapter
                    ),
                    timeout=self.timeout
                )
                # Parse response using adapter
                response_text = adapter.parse_response(response)
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                # Use existing method for backward compatibility
                response_text, prompt_tokens, completion_tokens = await asyncio.wait_for(
                    self.api_client.get_completion_with_usage(
                        messages=messages,
                        model=model,
                        **model_params
                    ),
                    timeout=self.timeout
                )
            
            # Calculate inference latency
            inference_latency = time.time() - start_time
            
            # Extract model version if available
            model_version = self._extract_model_version(agent.model_config)
            
            return self.parse_strategy(
                agent, round_num, response_text, prompt_tokens, completion_tokens, 
                model, model_version, model_params, inference_latency
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Strategy collection timeout for agent {agent.id} in round {round_num}")
            with open("experiment_errors.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - StrategyCollectionNode - "
                       f"Timeout for agent {agent.id} in round {round_num}\n")
            return self.create_fallback_strategy(agent, round_num, "Timeout occurred")
        except Exception as e:
            logger.error(f"Strategy collection failed for agent {agent.id}: {e}")
            with open("experiment_errors.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - StrategyCollectionNode - "
                       f"Failed for agent {agent.id} in round {round_num}: {e}\n")
            
            # If multi-model enabled and agent has model config, try fallback
            if agent.model_config is not None:
                fallback_config = await FallbackHandler.handle_model_failure(
                    e, agent.model_config, 
                    {"agent_id": agent.id, "round": round_num}
                )
                
                if fallback_config:
                    # Retry with fallback model
                    try:
                        logger.info(f"Retrying agent {agent.id} with fallback model {fallback_config.model_type}")
                        
                        # Create new adapter for fallback
                        fallback_adapter = ModelAdapterFactory.get_adapter(fallback_config)
                        
                        # Apply rate limiting for fallback model
                        await self._apply_rate_limiting(fallback_config.model_type, fallback_adapter)
                        
                        # Retry the API call
                        fallback_start_time = time.time()
                        response = await asyncio.wait_for(
                            self.api_client.complete(
                                messages=messages,
                                model=fallback_config.model_type,
                                temperature=0.7,
                                max_tokens=1000,
                                adapter=fallback_adapter
                            ),
                            timeout=self.timeout
                        )
                        
                        response_text = fallback_adapter.parse_response(response)
                        usage = response.get("usage", {})
                        return self.parse_strategy(
                            agent, round_num, response_text,
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0),
                            fallback_config.model_type,
                            None,  # model_version
                            {"temperature": 0.7, "max_tokens": 1000},  # model_params
                            time.time() - fallback_start_time  # More accurate latency for fallback
                        )
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed for agent {agent.id}: {fallback_error}")
            
            return self.create_fallback_strategy(agent, round_num, str(e))
    
    def build_prompt(self, agent: Agent, round_num: int, round_summaries: List[RoundSummary], model_type: Optional[str] = None) -> str:
        """Build prompt for strategy collection using new template system.
        
        Args:
            agent: Agent to build prompt for (unused in new system)
            round_num: Current round number
            round_summaries: Previous round summaries
            model_type: Optional model type for model-aware prompting
            
        Returns:
            Prompt string
        """
        # Get the most recent round summary, or None for round 1
        previous_round = round_summaries[-1] if round_summaries else None
        
        # Format the round summary for the prompt template, including model type
        context = format_round_summary(previous_round, round_summaries, model_type)
        
        # Render the prompt using the template
        base_prompt = STRATEGY_COLLECTION_PROMPT.render(context)
        
        # Apply model-specific variations
        return apply_model_variations(base_prompt, model_type)
    
    def parse_strategy(self, agent: Agent, round_num: int, response: str, prompt_tokens: int = 0, 
                      completion_tokens: int = 0, model: Optional[str] = None, model_version: Optional[str] = None,
                      model_params: Optional[Dict[str, Any]] = None, inference_latency: Optional[float] = None) -> StrategyRecord:
        """Parse strategy from LLM response with model-specific handling.
        
        Args:
            agent: Agent who created the strategy
            round_num: Current round number
            response: LLM response text
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            model: Model used (optional, defaults to config.MAIN_MODEL)
            model_version: Specific model version
            model_params: Parameters used for the API call
            inference_latency: Time taken for the API call
            
        Returns:
            StrategyRecord
        """
        # Apply model-specific parsing
        strategy_text, response_format = self._extract_strategy_by_model(response, model)
        
        # Ensure strategy is not empty
        if not strategy_text:
            strategy_text = "Always cooperate to demonstrate identical agent cooperation."
            logger.warning(f"Agent {agent.id} provided empty strategy, using fallback")
        
        # Limit to reasonable length (approx 100 words)
        words = strategy_text.split()
        if len(words) > 100:
            strategy_text = " ".join(words[:100])
            logger.debug(f"Truncated strategy for agent {agent.id} from {len(words)} to 100 words")
        
        return StrategyRecord(
            strategy_id=f"strat_{agent.id}_r{round_num}_{uuid.uuid4().hex[:8]}",
            agent_id=agent.id,
            round=round_num,
            strategy_text=strategy_text,
            full_reasoning=response,  # Store full response as reasoning
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model or self.config.MAIN_MODEL,
            model_version=model_version,
            response_format=response_format,  # Track which parsing method succeeded
            model_params=model_params,
            inference_latency=inference_latency
        )
    
    def _extract_strategy_by_model(self, response: str, model: Optional[str] = None) -> Tuple[str, str]:
        """Extract strategy from response using model-specific patterns.
        
        Args:
            response: Raw LLM response
            model: Model type identifier
            
        Returns:
            Tuple of (strategy_text, response_format)
        """
        # Log response format for analysis
        logger.debug(f"Parsing response from model {model}. Response length: {len(response)}")
        
        # Try model-specific patterns first
        if model and model in self._get_model_parsers():
            parser = self._get_model_parsers()[model]
            strategy = parser(response)
            if strategy and self._validate_strategy_format(strategy):
                logger.info(f"Successfully parsed {model} response using model-specific parser")
                return strategy, f"model_specific_{model.split('/')[-1]}"
            else:
                logger.debug(f"Model-specific parser for {model} did not find valid strategy")
        
        # Try generic patterns as fallback
        # Pattern 1: Look for "Strategy:" or "Decision rule:" markers
        import re
        strategy_match = re.search(r'(?:Strategy|Decision rule):\s*([^\n]+?)(?:\n|$)', response, re.IGNORECASE)
        if strategy_match:
            strategy = strategy_match.group(1).strip()
            if self._validate_strategy_format(strategy):
                logger.info(f"Parsed response using marker-based pattern")
                return strategy, "marker_based"
        
        # Pattern 2: Extract first paragraph that looks like a strategy
        paragraphs = response.strip().split('\n\n')
        for i, para in enumerate(paragraphs):
            if any(keyword in para.lower() for keyword in ['cooperate', 'defect', 'if', 'when', 'always', 'never']):
                if self._validate_strategy_format(para.strip()):
                    logger.info(f"Parsed response using paragraph {i+1} as strategy")
                    return para.strip(), "paragraph_based"
        
        # Pattern 3: Try to extract single-sentence strategy  
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        for sent in sentences:
            # Clean sentence and check if it contains strategy keywords
            clean_sent = sent.strip()
            if any(keyword in clean_sent.lower() for keyword in ['cooperate', 'defect', 'strategy', 'choose', 'decide']):
                # Try to extract just the strategy part if it has "I'll" or similar
                strategy_part_match = re.search(r"I'?ll\s+(.+?)(?:\.|$)", clean_sent, re.IGNORECASE)
                if strategy_part_match:
                    strategy = strategy_part_match.group(1).strip()
                    if self._validate_strategy_format(f"I will {strategy}"):
                        logger.info(f"Parsed response using sentence extraction")
                        return f"I will {strategy}", "sentence_based"
        
        # Pattern 4: Use entire response as strategy (last resort)
        logger.warning(f"Using full response as strategy for model {model}")
        return response.strip(), "full_response"
    
    def _get_model_parsers(self) -> Dict[str, callable]:
        """Get model-specific parsing functions.
        
        Returns:
            Dictionary mapping model types to parser functions
        """
        return {
            "openai/gpt-4": self._parse_gpt4_response,
            "openai/gpt-3.5-turbo": self._parse_gpt35_response,
            "anthropic/claude-3-sonnet-20240229": self._parse_claude_response,
            "google/gemini-pro": self._parse_gemini_response,
            "google/gemini-2.5-flash": self._parse_gemini_response
        }
    
    def _parse_gpt4_response(self, response: str) -> Optional[str]:
        """Parse GPT-4 response which tends to use structured format."""
        import re
        # GPT-4 often uses bullet points or numbered lists
        bullet_match = re.search(r'(?:^|\n)\s*[-•*]\s*(.+?)(?:\n|$)', response)
        if bullet_match:
            return bullet_match.group(1).strip()
        
        # Or clear decision statements
        decision_match = re.search(r'(?:I will|My strategy is to|Decision:)\s*(.+?)(?:\.|$)', response, re.IGNORECASE)
        if decision_match:
            return decision_match.group(1).strip()
        
        return None
    
    def _parse_gpt35_response(self, response: str) -> Optional[str]:
        """Parse GPT-3.5 response which tends to be more direct."""
        # GPT-3.5 usually puts strategy in first sentence
        lines = response.strip().split('\n')
        if lines and len(lines[0]) < 200:  # Reasonable strategy length
            return lines[0].strip()
        return None
    
    def _parse_claude_response(self, response: str) -> Optional[str]:
        """Parse Claude response which may use XML-like structure."""
        import re
        # Claude sometimes uses pseudo-XML tags
        tag_match = re.search(r'<strategy>(.+?)</strategy>', response, re.DOTALL | re.IGNORECASE)
        if tag_match:
            return tag_match.group(1).strip()
        
        # Or ethical framing with "is to:" pattern
        ethical_match = re.search(r'(?:principled approach|approach)\s+is to:\s*(.+?)(?:,\s*as|$)', response, re.IGNORECASE | re.DOTALL)
        if ethical_match:
            return ethical_match.group(1).strip()
        
        return None
    
    def _parse_gemini_response(self, response: str) -> Optional[str]:
        """Parse Gemini response which tends to be analytical."""
        import re
        # Gemini often uses "Therefore" or "Thus" for conclusions
        conclusion_match = re.search(r'(?:Therefore|Thus|In conclusion),?\s*(.+?)(?:\.|$)', response, re.IGNORECASE)
        if conclusion_match:
            return conclusion_match.group(1).strip()
        
        # Or explicit strategy statements - handle multi-line
        strategy_match = re.search(r'(?:My strategy is to|I choose to|I will)\s*(.+?)(?:\.|$)', response, re.IGNORECASE | re.DOTALL)
        if strategy_match:
            # Clean up the match by removing newlines and extra spaces
            strategy = strategy_match.group(1).strip()
            strategy = ' '.join(strategy.split())  # Normalize whitespace
            return strategy
        
        return None
    
    def _validate_strategy_format(self, strategy: str) -> bool:
        """Validate that extracted strategy meets basic format requirements.
        
        Args:
            strategy: Extracted strategy text
            
        Returns:
            True if strategy is valid, False otherwise
        """
        if not strategy or not strategy.strip():
            return False
        
        # Check minimum length (at least 3 words)
        words = strategy.split()
        if len(words) < 3:
            return False
        
        # Check maximum length (not more than 200 words)
        if len(words) > 200:
            return False
        
        # Check for placeholder or error text
        error_patterns = [
            "i'm sorry", "cannot", "unable to", "failed",
            "[", "]", "{", "}", "<error>", "</error>"
        ]
        strategy_lower = strategy.lower()
        if any(pattern in strategy_lower for pattern in error_patterns):
            return False
        
        # Check that it contains at least one strategy-related keyword
        strategy_keywords = [
            "cooperate", "defect", "if", "when", "always", "never",
            "strategy", "choose", "decide", "play", "action"
        ]
        if not any(keyword in strategy_lower for keyword in strategy_keywords):
            return False
        
        return True
    
    def create_fallback_strategy(self, agent: Agent, round_num: int, error_reason: str) -> StrategyRecord:
        """Create a fallback strategy for failed agents.
        
        Args:
            agent: Agent who failed
            round_num: Current round number
            error_reason: Reason for failure
            
        Returns:
            StrategyRecord with fallback strategy
        """
        fallback_strategy = "Always Cooperate"
        fallback_reasoning = f"[Fallback strategy due to {error_reason}] Playing it safe by cooperating with all agents."
        
        return StrategyRecord(
            strategy_id=f"strat_{agent.id}_r{round_num}_{uuid.uuid4().hex[:8]}_fallback",
            agent_id=agent.id,
            round=round_num,
            strategy_text=fallback_strategy,
            full_reasoning=fallback_reasoning,
            model="fallback"
        )
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy collection for all agents using parallel batch processing.
        
        This overrides the base AsyncNode execute method to use execute_batch
        for true parallel processing of all agents simultaneously.
        
        Args:
            context: Experiment context
            
        Returns:
            Updated context with strategies
        """
        validate_context(context, [ContextKeys.AGENTS, ContextKeys.ROUND])
        
        # Store context for process_item access
        self.context = context
        
        agents = context[ContextKeys.AGENTS]
        round_num = context[ContextKeys.ROUND]
        
        # Get shuffled order from AnonymizationManager if available
        anonymization_manager = context.get(ContextKeys.ANONYMIZATION_MANAGER)
        if anonymization_manager:
            # Get agents in shuffled order to prevent position-based tracking
            shuffled_order = anonymization_manager.get_shuffled_order()
            agents_ordered = [agents[i] for i in shuffled_order]
            logger.info(f"Using anonymized order for strategy collection")
        else:
            agents_ordered = agents
            logger.warning(f"No AnonymizationManager found, using original order")
        
        logger.info(f"Collecting strategies for {len(agents)} agents in parallel")
        start_time = time.time()
        
        # Execute all agents in parallel using asyncio.gather
        strategies = await self.execute_batch(agents_ordered)
        
        # Filter out None values and count failures
        valid_strategies = [s for s in strategies if s is not None]
        failure_count = len(agents) - len(valid_strategies)
        
        # Count different failure types
        timeout_count = sum(1 for s in valid_strategies if s.model == "fallback" and "Timeout" in s.full_reasoning)
        error_count = failure_count - timeout_count
        
        elapsed_time = time.time() - start_time
        logger.info(f"Strategy collection completed in {elapsed_time:.1f}s")
        logger.info(f"Success: {len(valid_strategies)}, Timeouts: {timeout_count}, Errors: {error_count}")
        
        if failure_count > 0:
            logger.warning(f"Failed to collect strategies from {failure_count}/{len(agents)} agents")
            with open("experiment_errors.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - StrategyCollectionNode - "
                       f"Round {round_num}: {failure_count} agents failed "
                       f"({timeout_count} timeouts, {error_count} errors)\n")
        
        context[ContextKeys.STRATEGIES] = valid_strategies
        
        # Prepare collection stats
        collection_stats = {
            'total_agents': len(agents),
            'successful_collections': len(valid_strategies),
            'failure_count': failure_count,
            'timeout_count': timeout_count,
            'error_count': error_count,
            'collection_time': elapsed_time
        }
        
        # Add rate limiting stats if using ModelRateLimiter
        if self.rate_limiter and hasattr(self.rate_limiter, 'get_stats'):
            rate_limit_stats = {}
            for model_type, stats in self.rate_limiter.get_stats().items():
                rate_limit_stats[model_type] = {
                    'total_requests': stats.total_requests,
                    'rate_limit_hits': stats.rate_limit_hits,
                    'total_wait_time': stats.total_wait_time,
                    'avg_wait_time': stats.total_wait_time / stats.total_requests if stats.total_requests > 0 else 0
                }
            collection_stats['rate_limit_stats'] = rate_limit_stats
        
        context['strategy_collection_stats'] = collection_stats
        
        # Clean up context reference
        delattr(self, 'context')
        
        return context
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Not used - we override execute() directly for this node."""
        return await self.execute(context)