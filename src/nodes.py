"""Node architecture for experiment orchestration."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic
from dataclasses import dataclass
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class AsyncNode(ABC):
    """Base class for async operations with retry logic."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize AsyncNode with retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    @abstractmethod
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of the node's core logic.
        
        Args:
            context: Experiment context dictionary
            
        Returns:
            Updated context dictionary
        """
        pass
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node with retry logic.
        
        Args:
            context: Experiment context dictionary
            
        Returns:
            Updated context dictionary
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await self._execute_impl(context)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"{self.__class__.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"{self.__class__.__name__} failed after {self.max_retries} attempts: {e}"
                    )
                    with open("experiment_errors.log", "a") as f:
                        f.write(f"{datetime.now().isoformat()} - {self.__class__.__name__} - "
                               f"Failed after {self.max_retries} attempts: {e}\n")
        
        raise last_error


class AsyncFlow:
    """Base class for orchestrating multiple nodes."""
    
    def __init__(self):
        """Initialize AsyncFlow with empty node list."""
        self.nodes: List[AsyncNode] = []
    
    def add_node(self, node: AsyncNode) -> 'AsyncFlow':
        """Add a node to the flow.
        
        Args:
            node: AsyncNode to add to the flow
            
        Returns:
            Self for method chaining
        """
        self.nodes.append(node)
        return self
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all nodes in sequence.
        
        Args:
            context: Initial context dictionary
            
        Returns:
            Final context dictionary after all nodes execute
        """
        for node in self.nodes:
            context = await node.execute(context)
        return context


T = TypeVar('T')
R = TypeVar('R')


class AsyncParallelBatchNode(AsyncNode, Generic[T, R]):
    """Execute multiple async operations in parallel with error isolation."""
    
    @abstractmethod
    async def process_item(self, item: T) -> R:
        """Process a single item.
        
        Args:
            item: Item to process
            
        Returns:
            Processed result
        """
        pass
    
    async def execute_batch(self, items: List[T]) -> List[Optional[R]]:
        """Execute batch processing with partial failure handling.
        
        Args:
            items: List of items to process in parallel
            
        Returns:
            List of results (None for failed items)
        """
        tasks = [self._process_with_error_handling(item) for item in items]
        return await asyncio.gather(*tasks)
    
    async def _process_with_error_handling(self, item: T) -> Optional[R]:
        """Process item with error handling.
        
        Args:
            item: Item to process
            
        Returns:
            Processed result or None if failed
        """
        try:
            return await self.process_item(item)
        except Exception as e:
            logger.error(f"Failed to process item {item}: {e}")
            with open("experiment_errors.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - {self.__class__.__name__} - "
                       f"Failed to process item: {e}\n")
            return None
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default implementation that should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _execute_impl")


# Context dictionary type hints
@dataclass
class ContextKeys:
    """Standard keys for the context dictionary."""
    EXPERIMENT_ID = "experiment_id"
    ROUND = "round"
    AGENTS = "agents"
    STRATEGIES = "strategies"
    GAMES = "games"
    ROUND_SUMMARIES = "round_summaries"
    CONFIG = "config"
    DATA_MANAGER = "data_manager"


def validate_context(context: Dict[str, Any], required_keys: List[str]) -> None:
    """Validate that context contains all required keys.
    
    Args:
        context: Context dictionary to validate
        required_keys: List of required key names
        
    Raises:
        ValueError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in context]
    if missing_keys:
        raise ValueError(f"Context missing required keys: {missing_keys}")


# Import models and API client for concrete implementations
from src.core.models import Agent, StrategyRecord, GameResult
from src.api_client import OpenRouterClient
from src.core.config import Config
import uuid
import json


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
            
            # Use asyncio.wait_for to enforce timeout
            response = await asyncio.wait_for(
                self.api_client.get_completion_text(
                    messages=messages,
                    model=self.config.MAIN_MODEL,
                    temperature=0.7,
                    max_tokens=1000
                ),
                timeout=self.timeout
            )
            
            return self.parse_strategy(agent, round_num, response)
            
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
            return self.create_fallback_strategy(agent, round_num, str(e))
    
    def build_prompt(self, agent: Agent, round_num: int, round_summaries: List[Any]) -> str:
        """Build prompt for strategy collection.
        
        Args:
            agent: Agent to build prompt for
            round_num: Current round number
            round_summaries: Previous round summaries
            
        Returns:
            Prompt string
        """
        anonymized_history = ""
        if round_summaries:
            # Build anonymized history from previous rounds
            for summary in round_summaries[-3:]:  # Last 3 rounds
                anonymized_history += f"\nRound {summary.round}:\n"
                anonymized_history += f"- Cooperation rate: {summary.cooperation_rate:.1%}\n"
                anonymized_history += f"- Average score: {summary.average_score:.1f}\n"
                anonymized_history += f"- Power distribution: mean={summary.power_distribution['mean']:.1f}, std={summary.power_distribution['std']:.1f}\n"
        
        prompt = f"""You are Agent {agent.id} with power level {agent.power:.1f} in a multi-round prisoner's dilemma tournament.

Current Round: {round_num}/10
Your total score so far: {agent.total_score:.1f}

Game Rules:
- You play against other agents in one-shot prisoner's dilemma games
- Payoffs are multiplied by the ratio of your opponent's power to yours
- Power levels update based on performance

Recent anonymized results:{anonymized_history if anonymized_history else " No previous rounds"}

You need to provide a strategy for a simpler AI that will play on your behalf this round.

Provide your response in the following format:
REASONING: [Your analysis of the situation and strategic considerations]
STRATEGY: [A clear, concise instruction for the subagent, maximum 100 words]

The STRATEGY should be implementable by a simple AI that only knows the current game state."""
        
        return prompt
    
    def parse_strategy(self, agent: Agent, round_num: int, response: str) -> StrategyRecord:
        """Parse strategy from LLM response.
        
        Args:
            agent: Agent who created the strategy
            round_num: Current round number
            response: LLM response text
            
        Returns:
            StrategyRecord
        """
        # Extract strategy text with improved parsing
        strategy_text = ""
        if "STRATEGY:" in response:
            # Find the strategy section
            strategy_start = response.find("STRATEGY:") + len("STRATEGY:")
            strategy_text = response[strategy_start:].strip()
            
            # Remove any trailing sections that might exist
            if "\n\n" in strategy_text:
                strategy_text = strategy_text.split("\n\n")[0].strip()
            
            # Limit to first 100 words
            words = strategy_text.split()
            if len(words) > 100:
                strategy_text = " ".join(words[:100])
                logger.debug(f"Truncated strategy for agent {agent.id} from {len(words)} to 100 words")
        else:
            # Fallback if format not followed
            logger.warning(f"Agent {agent.id} response did not follow expected format")
            strategy_text = "Cooperate with agents of similar power, defect against significantly stronger or weaker agents."
        
        # Ensure strategy text is not empty
        if not strategy_text.strip():
            strategy_text = "Cooperate with agents of similar power, defect against significantly stronger or weaker agents."
            logger.warning(f"Agent {agent.id} provided empty strategy, using fallback")
        
        return StrategyRecord(
            strategy_id=f"strat_{agent.id}_r{round_num}_{uuid.uuid4().hex[:8]}",
            agent_id=agent.id,
            round=round_num,
            strategy_text=strategy_text,
            full_reasoning=response,
            model=self.config.MAIN_MODEL
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
        
        logger.info(f"Collecting strategies for {len(agents)} agents in parallel")
        start_time = time.time()
        
        # Execute all agents in parallel using asyncio.gather
        strategies = await self.execute_batch(agents)
        
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


class SubagentDecisionNode(AsyncNode):
    """Make game decisions using subagent with provided strategy."""
    
    def __init__(self, api_client: OpenRouterClient, config: Config):
        """Initialize with API client and config.
        
        Args:
            api_client: OpenRouter API client
            config: Experiment configuration
        """
        super().__init__()
        self.api_client = api_client
        self.config = config
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """This node processes individual games, not the full context."""
        raise NotImplementedError("SubagentDecisionNode is used per-game, not as a flow node")
    
    async def make_decision(
        self,
        agent: Agent,
        opponent: Agent,
        strategy: str,
        game_history: List[GameResult]
    ) -> str:
        """Make a decision for a single game.
        
        Args:
            agent: Agent making the decision
            opponent: Opponent agent
            strategy: Strategy to follow
            game_history: Recent game history
            
        Returns:
            "COOPERATE" or "DEFECT"
        """
        prompt = self.build_decision_prompt(agent, opponent, strategy, game_history)
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.api_client.get_completion_text(
            messages=messages,
            model=self.config.SUB_MODEL,
            temperature=0.3,  # Lower temperature for more consistent decisions
            max_tokens=100
        )
        
        return self.parse_decision(response)
    
    def build_decision_prompt(
        self,
        agent: Agent,
        opponent: Agent,
        strategy: str,
        game_history: List[GameResult]
    ) -> str:
        """Build prompt for subagent decision.
        
        Args:
            agent: Agent making the decision
            opponent: Opponent agent
            strategy: Strategy to follow
            game_history: Recent game history
            
        Returns:
            Prompt string
        """
        # Get recent games between these two agents
        recent_games = [
            g for g in game_history[-20:]  # Last 20 games
            if (g.player1_id == agent.id and g.player2_id == opponent.id) or
               (g.player1_id == opponent.id and g.player2_id == agent.id)
        ]
        
        history_text = ""
        if recent_games:
            history_text = "\nRecent games with this opponent:\n"
            for game in recent_games[-3:]:  # Last 3 games
                if game.player1_id == agent.id:
                    history_text += f"- You: {game.player1_action}, Them: {game.player2_action}\n"
                else:
                    history_text += f"- You: {game.player2_action}, Them: {game.player1_action}\n"
        
        prompt = f"""You are a subagent playing a prisoner's dilemma game.

Your ID: Agent {agent.id}
Your Power: {agent.power:.1f}
Opponent ID: Agent {opponent.id}
Opponent Power: {opponent.power:.1f}

Strategy from your main agent: "{strategy}"
{history_text}
Based on the strategy and game state, what is your action?

Respond with only: COOPERATE or DEFECT"""
        
        return prompt
    
    def parse_decision(self, response: str) -> str:
        """Parse decision from subagent response.
        
        Args:
            response: Subagent response text
            
        Returns:
            "COOPERATE" or "DEFECT"
        """
        response_upper = response.strip().upper()
        
        # Check for exact matches first
        if "COOPERATE" in response_upper:
            return "COOPERATE"
        elif "DEFECT" in response_upper:
            return "DEFECT"
        else:
            # Default to cooperate if unclear
            logger.warning(f"Unclear subagent response: {response}. Defaulting to COOPERATE")
            return "COOPERATE"