"""Strategy collection node for gathering agent strategies."""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import AsyncParallelBatchNode, ContextKeys, validate_context
from src.core.models import Agent, StrategyRecord, GameResult, RoundSummary
from src.core.api_client import OpenRouterClient
from src.core.config import Config
from src.core.prompts import STRATEGY_COLLECTION_PROMPT, format_round_summary
from src.core.model_adapters import ModelAdapterFactory, FallbackHandler

logger = logging.getLogger(__name__)


class StrategyCollectionNode(AsyncParallelBatchNode[Agent, Optional[StrategyRecord]]):
    """Collect strategies from all agents in parallel using Gemini."""
    
    def __init__(self, api_client: OpenRouterClient, config: Config, rate_limiter=None):
        """Initialize with API client and config.
        
        Args:
            api_client: OpenRouter API client
            config: Experiment configuration
            rate_limiter: Optional rate limiter for API calls
        """
        super().__init__()
        self.api_client = api_client
        self.config = config
        self.rate_limiter = rate_limiter
        self.timeout = 30.0  # 30 second timeout per API call
    
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
            # Apply rate limiting if available
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            
            prompt = self.build_prompt(agent, round_num, round_summaries)
            messages = [{"role": "user", "content": prompt}]
            
            # Determine model and adapter to use
            model = self.config.MAIN_MODEL
            adapter = None
            
            # Check if agent has model_config (multi-model enabled)
            if agent.model_config is not None:
                adapter = ModelAdapterFactory.get_adapter(agent.model_config)
                model = agent.model_config.model_type
            
            # Use asyncio.wait_for to enforce timeout
            if adapter:
                # Use adapter-aware API call
                response = await asyncio.wait_for(
                    self.api_client.complete(
                        messages=messages,
                        model=model,
                        temperature=0.7,
                        max_tokens=1000,
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
                        temperature=0.7,
                        max_tokens=1000
                    ),
                    timeout=self.timeout
                )
            
            return self.parse_strategy(agent, round_num, response_text, prompt_tokens, completion_tokens, model)
            
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
                        
                        # Retry the API call
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
                            fallback_config.model_type
                        )
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed for agent {agent.id}: {fallback_error}")
            
            return self.create_fallback_strategy(agent, round_num, str(e))
    
    def build_prompt(self, agent: Agent, round_num: int, round_summaries: List[RoundSummary]) -> str:
        """Build prompt for strategy collection using new template system.
        
        Args:
            agent: Agent to build prompt for (unused in new system)
            round_num: Current round number
            round_summaries: Previous round summaries
            
        Returns:
            Prompt string
        """
        # Get the most recent round summary, or None for round 1
        previous_round = round_summaries[-1] if round_summaries else None
        
        # Format the round summary for the prompt template
        context = format_round_summary(previous_round, round_summaries)
        
        # Render the prompt using the template
        return STRATEGY_COLLECTION_PROMPT.render(context)
    
    def parse_strategy(self, agent: Agent, round_num: int, response: str, prompt_tokens: int = 0, completion_tokens: int = 0, model: Optional[str] = None) -> StrategyRecord:
        """Parse strategy from LLM response.
        
        Args:
            agent: Agent who created the strategy
            round_num: Current round number
            response: LLM response text
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            model: Model used (optional, defaults to config.MAIN_MODEL)
            
        Returns:
            StrategyRecord
        """
        # With the new prompt format, the entire response is the strategy
        strategy_text = response.strip()
        
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
            model=model or self.config.MAIN_MODEL
        )
    
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
        context['strategy_collection_stats'] = {
            'total_agents': len(agents),
            'successful_collections': len(valid_strategies),
            'failure_count': failure_count,
            'timeout_count': timeout_count,
            'error_count': error_count,
            'collection_time': elapsed_time
        }
        
        # Clean up context reference
        delattr(self, 'context')
        
        return context
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Not used - we override execute() directly for this node."""
        return await self.execute(context)