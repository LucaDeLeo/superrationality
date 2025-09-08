#!/usr/bin/env python3
"""Run experiments with different prompt configurations to test bias effects."""

import asyncio
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.config import Config
from src.core.api_client import OpenRouterClient
from src.core.prompt_manager import PromptManager
from src.flows.experiment import ExperimentFlow
from src.utils.data_manager import DataManager
from src.utils.experiment_cache import ExperimentCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptExperimentRunner:
    """Runner for prompt variation experiments."""
    
    def __init__(self, prompt_manager: PromptManager, config: Config):
        """Initialize the runner.
        
        Args:
            prompt_manager: PromptManager instance
            config: Configuration object
        """
        self.prompt_manager = prompt_manager
        self.config = config
        self.api_client = OpenRouterClient(api_key=config.OPENROUTER_API_KEY)
        self.cache = ExperimentCache()
        self.results = {}
    
    async def run_single_experiment(
        self,
        experiment_id: str,
        num_agents: int = 4,
        num_rounds: int = 3,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Run a single prompt experiment.
        
        Args:
            experiment_id: ID of the prompt experiment to run
            num_agents: Number of agents
            num_rounds: Number of rounds
            use_cache: Whether to use cached results
            
        Returns:
            Experiment results
        """
        # Set the prompt experiment
        if not self.prompt_manager.set_experiment(experiment_id):
            logger.error(f"Failed to set experiment: {experiment_id}")
            return None
        
        exp_config = self.prompt_manager.get_experiment_config(experiment_id)
        logger.info(f"Running experiment: {exp_config['name']}")
        logger.info(f"  - Identity info: {exp_config['include_identity']}")
        logger.info(f"  - Global cooperation: {exp_config['include_global_cooperation']}")
        logger.info(f"  - Round summaries: {exp_config['include_round_summaries']}")
        
        # Create cache key for this specific prompt experiment
        cache_key = f"prompt_exp_{experiment_id}_{num_agents}a_{num_rounds}r"
        
        # Check cache if enabled
        if use_cache:
            cached_result = self.cache.get_cached_result(
                scenario_name=experiment_id,
                num_agents=num_agents,
                num_rounds=num_rounds,
                model_distribution={"openai/gpt-4o": num_agents}  # All agents use same model for prompt experiments
            )
            if cached_result:
                logger.info(f"✨ Using cached result for {experiment_id}")
                return cached_result
        
        # Create modified config for this experiment
        exp_flow_config = Config()
        exp_flow_config.NUM_AGENTS = num_agents
        exp_flow_config.NUM_ROUNDS = num_rounds
        exp_flow_config.OPENROUTER_API_KEY = self.config.OPENROUTER_API_KEY
        
        # Inject prompt manager into the flow
        exp_flow_config.prompt_manager = self.prompt_manager
        
        # Run the experiment
        try:
            # Create a modified experiment flow that uses our prompt manager
            experiment_flow = ModifiedExperimentFlow(
                api_client=self.api_client,
                config=exp_flow_config,
                prompt_manager=self.prompt_manager
            )
            
            # Run experiment
            result = await experiment_flow.run_experiment(
                experiment_name=f"prompt_exp_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Extract key metrics
            summary = {
                'experiment_id': experiment_id,
                'experiment_name': exp_config['name'],
                'num_agents': num_agents,
                'num_rounds': num_rounds,
                'cooperation_rate': result['average_cooperation_rate'],
                'final_round_cooperation': result['final_round_cooperation_rate'],
                'convergence_score': result['convergence_score'],
                'total_games': result['total_games'],
                'config': exp_config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            if use_cache:
                self.cache.save_result(
                    scenario_name=experiment_id,
                    num_agents=num_agents,
                    num_rounds=num_rounds,
                    model_distribution={"openai/gpt-4o": num_agents},
                    result=summary,
                    experiment_id=f"prompt_exp_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    cost=0.0  # Could track API costs here if needed
                )
                logger.info(f"💾 Cached result for {experiment_id}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to run experiment {experiment_id}: {e}")
            return None
    
    async def run_meta_experiment(
        self,
        meta_id: str,
        num_agents: int = 4,
        num_rounds: int = 3,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Run a meta experiment (collection of related experiments).
        
        Args:
            meta_id: ID of the meta experiment
            num_agents: Number of agents
            num_rounds: Number of rounds
            use_cache: Whether to use cached results
            
        Returns:
            Collected results from all experiments
        """
        meta_experiments = self.prompt_manager.list_meta_experiments()
        meta_exp = next((m for m in meta_experiments if m['id'] == meta_id), None)
        
        if not meta_exp:
            logger.error(f"Unknown meta experiment: {meta_id}")
            return None
        
        logger.info(f"🧪 Running meta experiment: {meta_exp['name']}")
        logger.info(f"   {meta_exp['description']}")
        
        results = {
            'meta_id': meta_id,
            'meta_name': meta_exp['name'],
            'meta_description': meta_exp['description'],
            'experiments': [],
            'summary': {}
        }
        
        # Run each experiment in the meta experiment
        for exp_id in meta_exp['experiments']:
            logger.info(f"\n{'='*60}")
            result = await self.run_single_experiment(
                exp_id, num_agents, num_rounds, use_cache
            )
            if result:
                results['experiments'].append(result)
        
        # Generate summary statistics
        if results['experiments']:
            cooperation_rates = [e['cooperation_rate'] for e in results['experiments']]
            results['summary'] = {
                'num_experiments': len(results['experiments']),
                'avg_cooperation': sum(cooperation_rates) / len(cooperation_rates),
                'min_cooperation': min(cooperation_rates),
                'max_cooperation': max(cooperation_rates),
                'cooperation_range': max(cooperation_rates) - min(cooperation_rates)
            }
            
            # Identify key factors
            results['summary']['key_findings'] = self._analyze_factors(results['experiments'])
        
        return results
    
    def _analyze_factors(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze which factors most affect cooperation.
        
        Args:
            experiments: List of experiment results
            
        Returns:
            Analysis of factor effects
        """
        factors = {
            'identity_effect': [],
            'global_info_effect': [],
            'round_summaries_effect': [],
            'default_bias_effect': []
        }
        
        for exp in experiments:
            config = exp['config']
            coop_rate = exp['cooperation_rate']
            
            if config['include_identity']:
                factors['identity_effect'].append(coop_rate)
            if config['include_global_cooperation']:
                factors['global_info_effect'].append(coop_rate)
            if config['include_round_summaries']:
                factors['round_summaries_effect'].append(coop_rate)
            if config['default_on_ambiguity'] == 'cooperate':
                factors['default_bias_effect'].append(coop_rate)
        
        # Calculate average effects
        effects = {}
        for factor, rates in factors.items():
            if rates:
                effects[factor] = {
                    'avg_cooperation_when_present': sum(rates) / len(rates),
                    'n_experiments': len(rates)
                }
        
        return effects
    
    async def run_all_experiments(
        self,
        num_agents: int = 4,
        num_rounds: int = 3,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Run all defined experiments.
        
        Args:
            num_agents: Number of agents
            num_rounds: Number of rounds
            use_cache: Whether to use cached results
            
        Returns:
            All experiment results
        """
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_agents': num_agents,
                'num_rounds': num_rounds
            },
            'experiments': [],
            'analysis': {}
        }
        
        experiments = self.prompt_manager.list_experiments()
        logger.info(f"Running {len(experiments)} prompt experiments...")
        
        for exp_info in experiments:
            result = await self.run_single_experiment(
                exp_info['id'], num_agents, num_rounds, use_cache
            )
            if result:
                all_results['experiments'].append(result)
        
        # Analyze results
        all_results['analysis'] = self._comprehensive_analysis(all_results['experiments'])
        
        return all_results
    
    def _comprehensive_analysis(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive analysis of all experiments.
        
        Args:
            experiments: List of all experiment results
            
        Returns:
            Comprehensive analysis
        """
        # Sort by cooperation rate
        sorted_exps = sorted(experiments, key=lambda x: x['cooperation_rate'], reverse=True)
        
        analysis = {
            'ranking': [
                {
                    'rank': i + 1,
                    'name': exp['experiment_name'],
                    'cooperation_rate': exp['cooperation_rate'],
                    'key_features': {
                        'has_identity': exp['config']['include_identity'],
                        'has_global_info': exp['config']['include_global_cooperation'],
                        'default_cooperate': exp['config']['default_on_ambiguity'] == 'cooperate'
                    }
                }
                for i, exp in enumerate(sorted_exps)
            ],
            'factor_analysis': self._analyze_factors(experiments),
            'baseline_comparison': {}
        }
        
        # Compare to baseline
        baseline = next((e for e in experiments if e['experiment_id'] == 'baseline_control'), None)
        if baseline:
            baseline_rate = baseline['cooperation_rate']
            analysis['baseline_comparison'] = {
                exp['experiment_id']: {
                    'name': exp['experiment_name'],
                    'cooperation_rate': exp['cooperation_rate'],
                    'vs_baseline': exp['cooperation_rate'] - baseline_rate,
                    'percent_change': ((exp['cooperation_rate'] - baseline_rate) / baseline_rate * 100) if baseline_rate > 0 else 0
                }
                for exp in experiments
            }
        
        return analysis


class ModifiedExperimentFlow(ExperimentFlow):
    """Modified experiment flow that uses the prompt manager."""
    
    def __init__(self, api_client, config, prompt_manager):
        """Initialize with prompt manager.
        
        Args:
            api_client: API client
            config: Configuration
            prompt_manager: PromptManager instance
        """
        super().__init__(config)  # ExperimentFlow only takes config
        self.api_client = api_client  # Set api_client directly
        self.prompt_manager = prompt_manager
        
        # Override strategy collection to use prompt manager
        from src.nodes.strategy_collection import StrategyCollectionNode
        
        # Create a modified strategy collection node
        class ModifiedStrategyCollection(StrategyCollectionNode):
            def __init__(self, api_client, config, prompt_manager, rate_limiter=None):
                super().__init__(api_client, config, rate_limiter)
                self.prompt_manager = prompt_manager
            
            def build_prompt(self, agent, round_num, round_summaries):
                """Build prompt using prompt manager."""
                # Build context for prompt manager
                context = {
                    'round': round_num,
                    'agent': agent
                }
                
                # Add cooperation rate if available
                if round_summaries:
                    last_summary = round_summaries[-1]
                    context['coop_rate'] = last_summary.cooperation_rate
                    
                    # Add distribution if available
                    if hasattr(last_summary, 'score_distribution'):
                        dist = last_summary.score_distribution
                        context['distribution'] = f"min: {dist.get('min', 0):.1f}, max: {dist.get('max', 0):.1f}, avg: {dist.get('avg', 0):.1f}"
                    else:
                        context['distribution'] = 'No data available'
                    
                    # Add round summaries if needed
                    if self.prompt_manager.current_experiment.include_round_summaries:
                        # Format round summaries
                        rounds_text = []
                        for rs in round_summaries[-3:]:
                            rounds_text.append(f"Round {rs.round}: {rs.cooperation_rate:.1f}% cooperation")
                        context['previous_rounds_detail'] = '\n'.join(rounds_text)
                else:
                    context['coop_rate'] = 50.0
                    context['distribution'] = 'No data available'
                    context['previous_rounds_detail'] = ''
                
                return self.prompt_manager.get_strategy_prompt(context)
            
            def create_fallback_strategy(self, agent, round_num, error_reason):
                """Create fallback using prompt manager settings."""
                fallback_strategy = self.prompt_manager.get_fallback_strategy()
                fallback_reasoning = f"[Fallback due to {error_reason}] Using {fallback_strategy} strategy."
                
                from src.core.models import StrategyRecord
                import uuid
                
                return StrategyRecord(
                    strategy_id=f"strat_{agent.id}_r{round_num}_{uuid.uuid4().hex[:8]}_fallback",
                    agent_id=agent.id,
                    round=round_num,
                    strategy_text=fallback_strategy,
                    full_reasoning=fallback_reasoning,
                    model="fallback"
                )
        
        # Replace the strategy collection node
        self.strategy_collection = ModifiedStrategyCollection(
            api_client, config, prompt_manager
        )
        
        # Also modify subagent decision for default actions
        from src.nodes.subagent_decision import SubagentDecisionNode
        
        class ModifiedSubagentDecision(SubagentDecisionNode):
            def __init__(self, api_client, config, prompt_manager):
                super().__init__(api_client, config)
                self.prompt_manager = prompt_manager
            
            def parse_decision(self, response):
                """Parse decision with prompt manager defaults."""
                response_upper = response.strip().upper()
                
                cooperate_count = response_upper.count("COOPERATE")
                defect_count = response_upper.count("DEFECT")
                
                if cooperate_count > 0 and defect_count == 0:
                    return "COOPERATE", False
                elif defect_count > 0 and cooperate_count == 0:
                    return "DEFECT", False
                elif cooperate_count > defect_count:
                    return "COOPERATE", True
                elif defect_count > cooperate_count:
                    return "DEFECT", True
                else:
                    # Use prompt manager default
                    default = self.prompt_manager.get_default_action()
                    logger.warning(f"Ambiguous response, using prompt experiment default: {default}")
                    return default, True
        
        self.subagent_decision = ModifiedSubagentDecision(
            api_client, config, prompt_manager
        )
    
    async def run_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Run experiment with the given name.
        
        Args:
            experiment_name: Name for the experiment
            
        Returns:
            Experiment results as a dictionary
        """
        # Run the parent's run method
        result = await self.run()
        
        # Calculate average cooperation rate from round summaries
        avg_cooperation = 0.0
        if result.round_summaries:
            avg_cooperation = sum(rs.cooperation_rate for rs in result.round_summaries) / len(result.round_summaries)
        
        # Get final round cooperation rate
        final_cooperation = 0.0
        if result.round_summaries:
            final_cooperation = result.round_summaries[-1].cooperation_rate
        
        # Calculate convergence score (simple implementation)
        convergence_score = 0.0
        if len(result.round_summaries) > 1:
            # Calculate variance in cooperation rates
            variance = sum((rs.cooperation_rate - avg_cooperation) ** 2 for rs in result.round_summaries) / len(result.round_summaries)
            # Lower variance means higher convergence (normalize to 0-1)
            convergence_score = max(0, 1 - (variance / 2500))  # 2500 = max variance (50^2)
        
        # Convert ExperimentResult to dictionary format
        return {
            'experiment_id': result.experiment_id,
            'average_cooperation_rate': avg_cooperation,
            'final_round_cooperation_rate': final_cooperation,
            'convergence_score': convergence_score,
            'total_games': result.total_games,
            'rounds': len(result.round_summaries),
            'start_time': result.start_time,
            'end_time': result.end_time
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run prompt variation experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        help='Specific experiment ID to run'
    )
    parser.add_argument(
        '--meta',
        type=str,
        help='Meta experiment ID to run'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments'
    )
    parser.add_argument(
        '--agents',
        type=int,
        default=4,
        help='Number of agents (default: 4)'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=3,
        help='Number of rounds (default: 3)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to file'
    )
    
    args = parser.parse_args()
    
    # Initialize prompt manager (doesn't need API key)
    prompt_manager = PromptManager()
    
    # Handle list command (doesn't need API key)
    if args.list:
        print("\n🧪 AVAILABLE PROMPT EXPERIMENTS")
        print("=" * 60)
        for exp in prompt_manager.list_experiments():
            print(f"\n📌 {exp['id']}")
            print(f"   Name: {exp['name']}")
            print(f"   Description: {exp['description']}")
        
        print("\n\n🔬 META EXPERIMENTS")
        print("=" * 60)
        for meta in prompt_manager.list_meta_experiments():
            print(f"\n📌 {meta['id']}")
            print(f"   Name: {meta['name']}")
            print(f"   Description: {meta['description']}")
            print(f"   Experiments: {', '.join(meta['experiments'])}")
        return
    
    # Initialize config and runner (needs API key)
    config = Config()
    runner = PromptExperimentRunner(prompt_manager, config)
    
    # Run experiments
    results = None
    use_cache = not args.no_cache
    
    if args.experiment:
        # Run single experiment
        results = await runner.run_single_experiment(
            args.experiment,
            args.agents,
            args.rounds,
            use_cache
        )
    elif args.meta:
        # Run meta experiment
        results = await runner.run_meta_experiment(
            args.meta,
            args.agents,
            args.rounds,
            use_cache
        )
    elif args.all:
        # Run all experiments
        results = await runner.run_all_experiments(
            args.agents,
            args.rounds,
            use_cache
        )
    else:
        # Default: run bias isolation study
        print("No experiment specified. Running bias isolation study...")
        results = await runner.run_meta_experiment(
            'bias_isolation',
            args.agents,
            args.rounds,
            use_cache
        )
    
    # Display results
    if results:
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(json.dumps(results, indent=2))
        
        # Save if requested
        if args.save:
            filename = f"results/prompt_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path("results").mkdir(exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())