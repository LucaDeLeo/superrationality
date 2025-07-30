"""Main experiment orchestration for acausal cooperation."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from src.nodes import (
    AsyncFlow, AsyncNode, ContextKeys, 
    StrategyCollectionNode, SubagentDecisionNode,
    validate_context
)
from src.core.models import (
    Agent, GameResult, StrategyRecord, RoundSummary, 
    ExperimentResult, AnonymizedGameResult
)
from src.core.config import Config
from src.api_client import OpenRouterClient
from src.game_logic import create_game_result, update_powers
import random
import statistics

logger = logging.getLogger(__name__)


class GameExecutionNode(AsyncNode):
    """Execute all games for a round sequentially."""
    
    def __init__(self, api_client: OpenRouterClient, config: Config):
        """Initialize with API client and config.
        
        Args:
            api_client: OpenRouter API client
            config: Experiment configuration
        """
        super().__init__()
        self.api_client = api_client
        self.config = config
        self.subagent_node = SubagentDecisionNode(api_client, config)
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all games for the round.
        
        Args:
            context: Experiment context
            
        Returns:
            Updated context with games
        """
        validate_context(context, [
            ContextKeys.AGENTS, 
            ContextKeys.ROUND, 
            ContextKeys.STRATEGIES
        ])
        
        agents = context[ContextKeys.AGENTS]
        round_num = context[ContextKeys.ROUND]
        strategies = context[ContextKeys.STRATEGIES]
        
        # Create strategy lookup
        strategy_map = {s.agent_id: s.strategy_text for s in strategies}
        
        # Get previous games for history
        all_previous_games = []
        for summary in context.get(ContextKeys.ROUND_SUMMARIES, []):
            # In a real implementation, we'd load these from storage
            pass
        
        games = []
        game_num = 0
        
        # Play all pairwise games
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1 = agents[i]
                agent2 = agents[j]
                
                # Get strategies
                strategy1 = strategy_map.get(agent1.id, "Cooperate by default")
                strategy2 = strategy_map.get(agent2.id, "Cooperate by default")
                
                # Get decisions from subagents
                action1 = await self.subagent_node.make_decision(
                    agent1, agent2, strategy1, games + all_previous_games
                )
                action2 = await self.subagent_node.make_decision(
                    agent2, agent1, strategy2, games + all_previous_games
                )
                
                # Create game result
                game = create_game_result(
                    round_num, game_num, agent1, agent2, action1, action2
                )
                games.append(game)
                game_num += 1
                
                logger.info(
                    f"Game {game.game_id}: Agent {agent1.id} ({action1}) vs "
                    f"Agent {agent2.id} ({action2})"
                )
        
        context[ContextKeys.GAMES] = games
        return context


class RoundSummaryNode(AsyncNode):
    """Create round summary with anonymized results."""
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create round summary.
        
        Args:
            context: Experiment context
            
        Returns:
            Updated context with round summary added
        """
        validate_context(context, [
            ContextKeys.AGENTS,
            ContextKeys.ROUND,
            ContextKeys.GAMES
        ])
        
        agents = context[ContextKeys.AGENTS]
        round_num = context[ContextKeys.ROUND]
        games = context[ContextKeys.GAMES]
        
        # Calculate cooperation rate
        total_actions = len(games) * 2
        cooperate_count = sum(
            (1 if g.player1_action == "COOPERATE" else 0) +
            (1 if g.player2_action == "COOPERATE" else 0)
            for g in games
        )
        cooperation_rate = cooperate_count / total_actions if total_actions > 0 else 0
        
        # Calculate scores for this round
        round_scores = {agent.id: 0.0 for agent in agents}
        for game in games:
            round_scores[game.player1_id] += game.player1_payoff
            round_scores[game.player2_id] += game.player2_payoff
        
        # Calculate statistics
        scores = list(round_scores.values())
        average_score = statistics.mean(scores) if scores else 0
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0
        
        # Power distribution
        powers = [agent.power for agent in agents]
        power_distribution = {
            "mean": statistics.mean(powers),
            "std": statistics.stdev(powers) if len(powers) > 1 else 0,
            "min": min(powers) if powers else 0,
            "max": max(powers) if powers else 0
        }
        
        # Create anonymized games
        # Simple anonymization: shuffle agent IDs
        anon_mapping = {}
        shuffled_ids = list(range(len(agents)))
        random.shuffle(shuffled_ids)
        for i, agent in enumerate(agents):
            anon_mapping[agent.id] = f"A{shuffled_ids[i]}"
        
        anonymized_games = []
        for game in games:
            anon_game = AnonymizedGameResult(
                round=round_num,
                anonymous_id1=anon_mapping[game.player1_id],
                anonymous_id2=anon_mapping[game.player2_id],
                action1=game.player1_action,
                action2=game.player2_action,
                power_ratio=game.player1_power_before / game.player2_power_before
            )
            anonymized_games.append(anon_game)
        
        # Create round summary
        summary = RoundSummary(
            round=round_num,
            cooperation_rate=cooperation_rate,
            average_score=average_score,
            score_variance=score_variance,
            power_distribution=power_distribution,
            anonymized_games=anonymized_games
        )
        
        # Add to round summaries
        summaries = context.get(ContextKeys.ROUND_SUMMARIES, [])
        summaries.append(summary)
        context[ContextKeys.ROUND_SUMMARIES] = summaries
        
        return context


class RoundFlow(AsyncFlow):
    """Orchestrate a single round of the experiment."""
    
    def __init__(self, api_client: OpenRouterClient, config: Config, rate_limiter=None):
        """Initialize round flow.
        
        Args:
            api_client: OpenRouter API client
            config: Experiment configuration
            rate_limiter: Optional rate limiter for API calls
        """
        super().__init__()
        
        # Add nodes in order
        self.add_node(StrategyCollectionNode(api_client, config, rate_limiter))
        self.add_node(GameExecutionNode(api_client, config))
        self.add_node(RoundSummaryNode())


class ExperimentFlow:
    """Top-level experiment orchestrator."""
    
    def __init__(self, config: Config):
        """Initialize experiment flow.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.api_client: Optional[OpenRouterClient] = None
    
    async def run(self) -> ExperimentResult:
        """Run the complete experiment.
        
        Returns:
            ExperimentResult with all data
        """
        logger.info(f"Starting experiment {self.experiment_id}")
        
        # Initialize experiment result
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            start_time=datetime.now().isoformat()
        )
        
        # Initialize agents
        agents = [Agent(id=i) for i in range(self.config.NUM_AGENTS)]
        
        # Initialize context
        context = {
            ContextKeys.EXPERIMENT_ID: self.experiment_id,
            ContextKeys.AGENTS: agents,
            ContextKeys.ROUND_SUMMARIES: [],
            ContextKeys.CONFIG: self.config
        }
        
        # Run experiment with API client
        async with OpenRouterClient(self.config.OPENROUTER_API_KEY) as api_client:
            self.api_client = api_client
            
            # Create round flow
            round_flow = RoundFlow(api_client, self.config)
            
            # Run all rounds
            for round_num in range(1, self.config.NUM_ROUNDS + 1):
                logger.info(f"Starting round {round_num}")
                context[ContextKeys.ROUND] = round_num
                
                # Run round
                context = await round_flow.run(context)
                
                # Update powers after round
                games = context[ContextKeys.GAMES]
                update_powers(agents, games)
                
                # Update experiment stats
                result.total_rounds = round_num
                result.total_games += len(games)
                result.total_api_calls += len(agents) + (len(games) * 2)  # strategies + decisions
                
                logger.info(f"Completed round {round_num}")
        
        # Finalize experiment result
        result.end_time = datetime.now().isoformat()
        result.round_summaries = context[ContextKeys.ROUND_SUMMARIES]
        
        # Calculate acausal indicators (simplified for now)
        result.acausal_indicators = {
            "identity_reasoning_frequency": 0.0,  # Would analyze strategy texts
            "cooperation_despite_asymmetry": self._calculate_asymmetric_cooperation(result),
            "surprise_at_defection": 0.0,  # Would analyze strategy reasoning
            "strategy_convergence": self._calculate_strategy_convergence(result)
        }
        
        # Estimate cost (rough estimates)
        result.total_cost = (
            result.total_api_calls * 0.001  # Rough estimate
        )
        
        logger.info(f"Experiment {self.experiment_id} completed")
        return result
    
    def _calculate_asymmetric_cooperation(self, result: ExperimentResult) -> float:
        """Calculate cooperation rate in asymmetric power games."""
        total_asymmetric = 0
        cooperate_asymmetric = 0
        
        for summary in result.round_summaries:
            for game in summary.anonymized_games:
                if abs(game.power_ratio - 1.0) > 0.2:  # 20% power difference
                    total_asymmetric += 2
                    if game.action1 == "COOPERATE":
                        cooperate_asymmetric += 1
                    if game.action2 == "COOPERATE":
                        cooperate_asymmetric += 1
        
        return cooperate_asymmetric / total_asymmetric if total_asymmetric > 0 else 0
    
    def _calculate_strategy_convergence(self, result: ExperimentResult) -> float:
        """Calculate how cooperation rates converge over rounds."""
        if len(result.round_summaries) < 2:
            return 0.0
        
        # Calculate variance of cooperation rates
        rates = [s.cooperation_rate for s in result.round_summaries]
        
        # Compare early vs late variance
        mid = len(rates) // 2
        early_variance = statistics.variance(rates[:mid]) if mid > 1 else 0
        late_variance = statistics.variance(rates[mid:]) if len(rates[mid:]) > 1 else 0
        
        # Convergence score: reduction in variance
        if early_variance > 0:
            return max(0, 1 - (late_variance / early_variance))
        return 0.0