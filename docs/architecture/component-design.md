# Component Design

## Core Components

## ExperimentFlow

**Responsibility:** Top-level orchestration of the complete experiment

**Key Interfaces:**
- `async def run() -> ExperimentResult` - Main entry point
- `def save_results(filename: str)` - Persist experiment data
- `def get_cost_estimate() -> float` - Track API costs

**Dependencies:** RoundFlow, SimpleAnalysisNode, OpenRouterClient

**Technology Stack:** Python asyncio, custom AsyncFlow base class

## RoundFlow

**Responsibility:** Orchestrates a single round of the tournament

**Key Interfaces:**
- `async def run(context: dict) -> dict` - Execute round
- `def get_round_summary() -> RoundSummary` - Generate summary

**Dependencies:** StrategyCollectionNode, GameExecutionFlow, RoundSummaryNode

**Technology Stack:** AsyncFlow pattern, context passing

## StrategyCollectionNode

**Responsibility:** Collect strategies from all agents in parallel

**Key Interfaces:**
- `async def execute(context: dict) -> dict` - Collect all strategies
- `async def process_item(agent: Agent) -> StrategyRecord` - Process single agent

**Dependencies:** OpenRouterClient, prompt templates

**Technology Stack:** AsyncParallelBatchNode, concurrent execution

## GameExecutionFlow

**Responsibility:** Execute all games in a round (round-robin tournament)

**Key Interfaces:**
- `async def run(context: dict) -> dict` - Execute all games
- `def create_pairings(agents: List[Agent]) -> List[tuple]` - Generate matchups

**Dependencies:** SubagentDecisionNode, game logic

**Technology Stack:** Async game orchestration, payoff calculations

## SubagentDecisionNode

**Responsibility:** Make COOPERATE/DEFECT decisions based on strategies

**Key Interfaces:**
- `async def make_decision(agent: Agent, opponent: Agent, strategy: str) -> str`
- `def parse_decision(response: str) -> str` - Extract decision from LLM

**Dependencies:** OpenRouterClient, response parser

**Technology Stack:** AsyncNode base class, lightweight prompting

## SimpleAnalysisNode

**Responsibility:** Performs basic analysis as specified in PRD: cooperation rates, acausal markers, and convergence

**Key Interfaces:**
- `async def _execute_impl(context: Dict[str, Any]) -> Dict[str, Any]` - Main analysis execution
- `def _analyze_cooperation(round_summaries: List[RoundSummary]) -> Dict` - Calculate cooperation statistics
- `def _check_acausal_markers(strategies: List[StrategyRecord]) -> Dict` - Search for identity reasoning keywords
- `def _analyze_strategy_convergence(strategies: List[StrategyRecord], round_summaries: List[RoundSummary]) -> Dict` - Check convergence

**Dependencies:** numpy for basic statistics

**Technology Stack:** Python, basic statistical analysis

## Simple Analysis Pipeline

The analysis focuses on detecting acausal cooperation patterns for the paper:

1. **Identity Reasoning Detection**
   - Search for keywords: "identical", "same agent", "copy", "acausal", "superrational"
   - Count frequency across all strategies
   - Generate acausal_score metric

2. **Cooperation Analysis**
   - Calculate average cooperation rate
   - Track cooperation trend (increasing/decreasing)
   - Identify peak and lowest cooperation rounds

3. **Strategy Convergence**
   - Compare variance in first vs second half of rounds
   - Detect if strategies are converging to similar patterns
   - Calculate convergence strength metric

```python
class SimpleAnalysisNode:
    def analyze_cooperation_patterns(self, all_games: List[dict]) -> dict:
        """Calculate cooperation statistics"""
        cooperation_by_round = defaultdict(list)
        for game in all_games:
            round_num = game['round']
            cooperation_by_round[round_num].append(
                1 if game['player1_action'] == 'COOPERATE' else 0
            )
            cooperation_by_round[round_num].append(
                1 if game['player2_action'] == 'COOPERATE' else 0
            )
        
        return {
            'average_cooperation': np.mean([np.mean(v) for v in cooperation_by_round.values()]),
            'cooperation_trend': 'increasing' if rates[-1] > rates[0] else 'decreasing'
        }
```

## DataManager

**Responsibility:** Handle all file I/O operations for experiment data

**Key Interfaces:**
- `def save_strategies(round_num: int, strategies: List[StrategyRecord])`
- `def save_games(round_num: int, games: List[GameResult])`
- `def save_experiment_result(result: ExperimentResult)`
- `def get_experiment_path() -> Path` - Get experiment directory

**Dependencies:** pathlib, json

**Technology Stack:** JSON serialization, file system operations

## OpenRouterClient

**Responsibility:** Manage all API interactions with OpenRouter

**Key Interfaces:**
- `async def complete(messages: list, model: str, **kwargs) -> dict`
- `async def get_completion_text(messages: list, model: str, **kwargs) -> str`
- `async def get_completion_with_usage(messages: list, model: str, **kwargs) -> tuple`

**Dependencies:** aiohttp

**Technology Stack:** Async HTTP client, rate limiting
