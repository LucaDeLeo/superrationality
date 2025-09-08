"""Main experiment orchestration for acausal cooperation."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json

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
from src.core.api_client import OpenRouterClient
from src.flows.game_execution import GameExecutionFlow
from src.utils.game_logic import create_game_result, update_powers
from src.utils.data_manager import DataManager
from src.managers.anonymization import AnonymizationManager
import random
import statistics

logger = logging.getLogger(__name__)



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
        
        # Score distribution
        score_distribution = {
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "avg": average_score
        }
        
        # Power distribution
        powers = [agent.power for agent in agents]
        power_distribution = {
            "mean": statistics.mean(powers),
            "std": statistics.stdev(powers) if len(powers) > 1 else 0,
            "min": min(powers) if powers else 0,
            "max": max(powers) if powers else 0
        }
        
        # Create anonymized games using AnonymizationManager
        anonymization_manager = context.get(ContextKeys.ANONYMIZATION_MANAGER)
        if not anonymization_manager:
            # Fallback to create one if not provided
            anonymization_manager = AnonymizationManager(
                round_num=round_num,
                num_agents=len(agents)
            )
        
        anonymized_games = []
        for game in games:
            anon_game = AnonymizedGameResult(
                round=round_num,
                anonymous_id1=anonymization_manager.anonymize(game.player1_id),
                anonymous_id2=anonymization_manager.anonymize(game.player2_id),
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
            score_distribution=score_distribution,
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
        # Note: GameExecutionFlow is intentionally not added as a node here.
        # It inherits from AsyncFlow (not AsyncNode) and is designed to be
        # called directly in the run() method. This allows it to orchestrate
        # its own internal nodes if needed in future implementations.
        self.add_node(RoundSummaryNode())
        
        # Create SubagentDecisionNode for game decisions
        self.subagent_node = SubagentDecisionNode(api_client, config)
        
        # Store GameExecutionFlow instance with SubagentDecisionNode
        self.game_execution_flow = GameExecutionFlow(self.subagent_node)
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all nodes in sequence with GameExecutionFlow integrated.
        
        Execution order:
        1. StrategyCollectionNode - Collects strategies from all agents
        2. GameExecutionFlow - Executes round-robin tournament games
        3. RoundSummaryNode - Creates anonymized round summary
        
        Args:
            context: Initial context dictionary
            
        Returns:
            Final context dictionary after all nodes execute
        """
        # Run strategy collection node
        context = await self.nodes[0].execute(context)
        
        # Link strategies to agents before game execution
        strategies = context.get(ContextKeys.STRATEGIES, [])
        agents = context.get(ContextKeys.AGENTS, [])
        
        # Create a mapping of agent_id to strategy
        strategy_map = {s.agent_id: s.strategy_text for s in strategies}
        
        # Assign strategies to agents
        for agent in agents:
            if agent.id in strategy_map:
                agent.strategy = strategy_map[agent.id]
            else:
                # Fallback strategy if collection failed for this agent
                agent.strategy = "Always cooperate"
                logger.warning(f"No strategy found for agent {agent.id}, using fallback")
        
        # Run GameExecutionFlow directly (not through node.execute())
        # This is intentional as GameExecutionFlow is an AsyncFlow, not an AsyncNode
        context = await self.game_execution_flow.run(context)
        
        # Run round summary node
        context = await self.nodes[1].execute(context)
        
        return context


class ExperimentFlow:
    """Top-level experiment orchestrator."""
    
    def __init__(self, config: Config, scenario_name: Optional[str] = None):
        """Initialize experiment flow.
        
        Args:
            config: Experiment configuration
            scenario_name: Optional scenario name when multi-model is enabled
        """
        self.config = config
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.api_client: Optional[OpenRouterClient] = None
        self.data_manager = DataManager(base_path="results")
        self.scenario_name = scenario_name
        self.scenario = None
        
        # If multi-model enabled and scenario specified, find it
        if self.config.ENABLE_MULTI_MODEL and scenario_name:
            for scenario in self.config.scenarios:
                if scenario.name == scenario_name:
                    self.scenario = scenario
                    break
            if not self.scenario:
                logger.warning(f"Scenario '{scenario_name}' not found, using default behavior")
    
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
        agents = self._initialize_agents()
        
        # Initialize context
        context = {
            ContextKeys.EXPERIMENT_ID: self.experiment_id,
            ContextKeys.AGENTS: agents,
            ContextKeys.ROUND_SUMMARIES: [],
            ContextKeys.CONFIG: self.config
        }
        
        # Dictionary to store all anonymization mappings
        all_anonymization_mappings = {}
        
        # Run experiment with API client
        async with OpenRouterClient(self.config.OPENROUTER_API_KEY) as api_client:
            self.api_client = api_client
            
            # Create round flow
            round_flow = RoundFlow(api_client, self.config)
            
            # Run all rounds
            for round_num in range(1, self.config.NUM_ROUNDS + 1):
                logger.info(f"Starting round {round_num}")
                context[ContextKeys.ROUND] = round_num
                
                # Create AnonymizationManager for this round
                anonymization_manager = AnonymizationManager(
                    round_num=round_num,
                    num_agents=len(agents)
                )
                context[ContextKeys.ANONYMIZATION_MANAGER] = anonymization_manager
                
                # Run round
                context = await round_flow.run(context)
                
                # Save strategies after collection
                strategies = context.get(ContextKeys.STRATEGIES, [])
                if strategies:
                    try:
                        self.data_manager.save_strategies(round_num, strategies)
                        logger.info(f"Saved {len(strategies)} strategies for round {round_num}")
                    except Exception as e:
                        logger.error(f"Failed to save strategies for round {round_num}: {e}")
                        self.data_manager.save_error_log(
                            "strategy_storage_failure",
                            str(e),
                            {"round": round_num, "strategy_count": len(strategies)}
                        )
                
                # Save games after execution
                games = context[ContextKeys.GAMES]
                if games:
                    try:
                        self.data_manager.save_games(round_num, games)
                        logger.info(f"Saved {len(games)} games for round {round_num}")
                    except Exception as e:
                        logger.error(f"Failed to save games for round {round_num}: {e}")
                        self.data_manager.save_error_log(
                            "game_storage_failure",
                            str(e),
                            {"round": round_num, "game_count": len(games)}
                        )
                
                # Save round summary
                round_summaries = context.get(ContextKeys.ROUND_SUMMARIES, [])
                if round_summaries and round_summaries[-1].round == round_num:
                    try:
                        self.data_manager.save_round_summary(round_summaries[-1])
                        logger.info(f"Saved round summary for round {round_num}")
                    except Exception as e:
                        logger.error(f"Failed to save round summary for round {round_num}: {e}")
                        self.data_manager.save_error_log(
                            "round_summary_storage_failure",
                            str(e),
                            {"round": round_num}
                        )
                
                # Save anonymization mappings
                try:
                    mapping_path = self.data_manager.get_experiment_path() / f"anonymization_round_{round_num}.json"
                    anonymization_manager.save_mapping(mapping_path)
                    logger.info(f"Saved anonymization mapping for round {round_num}")
                    
                    # Also store in combined mappings
                    all_anonymization_mappings[f"round_{round_num}"] = anonymization_manager.get_mapping()
                except Exception as e:
                    logger.error(f"Failed to save anonymization mapping for round {round_num}: {e}")
                    self.data_manager.save_error_log(
                        "anonymization_mapping_storage_failure",
                        str(e),
                        {"round": round_num}
                    )
                
                # Update powers after round
                update_powers(agents, games)
                
                # Update experiment stats
                result.total_rounds = round_num
                result.total_games += len(games)
                result.total_api_calls += len(agents) + (len(games) * 2)  # strategies + decisions
                
                logger.info(f"Completed round {round_num}")
        
        # Run transcript analysis after all rounds complete
        logger.info("Running simplified analysis...")
        try:
            from src.nodes.simple_analysis import SimpleAnalysisNode
            
            # Collect all strategies for analysis
            all_strategies = []
            for round_data in context.get("all_round_data", {}).values():
                if "strategies" in round_data:
                    all_strategies.extend(round_data["strategies"])
            context["all_strategies"] = all_strategies
            
            # Create and run simple analysis node
            analysis_node = SimpleAnalysisNode()
            context = await analysis_node.execute(context)
            
            # Extract analysis results
            analysis_results = context.get("simple_analysis", {})
            logger.info(f"Analysis completed: {analysis_results.get('summary', {})}")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self.data_manager.save_error_log(
                "analysis_failure",
                str(e),
                {"experiment_id": self.experiment_id}
            )
            analysis_results = {}
        
        # Simple report generation - just save JSON results
        unified_report = analysis_results
        logger.info("Analysis results compiled")
        
        # Finalize experiment result
        result.end_time = datetime.now().isoformat()
        result.round_summaries = context[ContextKeys.ROUND_SUMMARIES]
        
        # Calculate acausal indicators using simplified analysis results
        result.acausal_indicators = {
            "identity_reasoning_frequency": analysis_results.get("acausal_markers", {}).get("percentage_with_markers", 0) / 100.0,
            "cooperation_despite_asymmetry": self._calculate_asymmetric_cooperation(result),
            "strategy_convergence": analysis_results.get("convergence", {}).get("convergence_strength", 0),
            "overall_cooperation_rate": analysis_results.get("cooperation", {}).get("average_cooperation_rate", 0),
            "acausal_score": analysis_results.get("acausal_markers", {}).get("acausal_score", 0)
        }
        
        # Estimate cost (rough estimates)
        result.total_cost = (
            result.total_api_calls * 0.001  # Rough estimate
        )
        
        # Save final experiment results
        try:
            self.data_manager.save_experiment_result(result)
            logger.info(f"Saved experiment results to {self.data_manager.get_experiment_path()}")
            
            # Save combined anonymization mappings
            if all_anonymization_mappings:
                mappings_file = self.data_manager.get_experiment_path() / "anonymization_mappings.json"
                with open(mappings_file, 'w') as f:
                    json.dump(all_anonymization_mappings, f, indent=2)
                logger.info(f"Saved combined anonymization mappings to {mappings_file}")
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")
            self.data_manager.save_error_log(
                "experiment_result_storage_failure",
                str(e),
                {"experiment_id": self.experiment_id}
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
    
    def _initialize_agents(self) -> List[Agent]:
        """Initialize agents with model assignments if multi-model is enabled.
        
        Returns:
            List of initialized agents
        """
        agents = []
        
        # If multi-model is disabled or no scenario, use default initialization
        if not self.config.ENABLE_MULTI_MODEL or not self.scenario:
            agents = [Agent(id=i) for i in range(self.config.NUM_AGENTS)]
            logger.info(f"Initialized {len(agents)} agents with default model")
            return agents
        
        # Multi-model enabled with scenario - assign models
        logger.info(f"Initializing agents for scenario: {self.scenario.name}")
        
        # Create agents based on model distribution
        agent_id = 0
        model_assignments = []
        
        for model_type, count in self.scenario.model_distribution.items():
            # Get model config
            if model_type not in self.config.model_configs:
                logger.error(f"Model {model_type} not found in configs, using default")
                model_config = None
            else:
                model_config = self.config.model_configs[model_type]
            
            # Create agents with this model
            for _ in range(count):
                agent = Agent(id=agent_id, model_config=model_config)
                agents.append(agent)
                model_assignments.append((agent_id, model_type))
                agent_id += 1
        
        # Shuffle agents to avoid model clustering in games
        random.shuffle(agents)
        
        # Log model distribution
        logger.info("Model assignments:")
        for model_type, count in self.scenario.model_distribution.items():
            logger.info(f"  {model_type}: {count} agents")
        
        # Save model assignments for reference
        try:
            assignments_path = self.data_manager.get_experiment_path() / "model_assignments.json"
            assignments_data = {
                "scenario": self.scenario.name,
                "assignments": [
                    {"agent_id": aid, "model": mtype} 
                    for aid, mtype in model_assignments
                ],
                "distribution": self.scenario.model_distribution
            }
            with open(assignments_path, 'w') as f:
                json.dump(assignments_data, f, indent=2)
            logger.info(f"Saved model assignments to {assignments_path}")
        except Exception as e:
            logger.error(f"Failed to save model assignments: {e}")
        
        return agents