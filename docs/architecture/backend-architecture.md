# Backend Architecture

## Service Architecture

Since this is a focused research experiment, we use a modular Python architecture rather than a traditional server:

### Function Organization
```
acausal/
├── nodes/
│   ├── __init__.py
│   ├── base.py                    # AsyncNode, AsyncFlow base classes
│   ├── strategy_collection.py     # StrategyCollectionNode
│   ├── subagent_decision.py      # SubagentDecisionNode
│   └── analysis.py               # AnalysisNode
├── flows/
│   ├── __init__.py
│   ├── experiment.py             # ExperimentFlow
│   ├── round.py                  # RoundFlow
│   └── game_execution.py         # GameExecutionFlow
├── core/
│   ├── __init__.py
│   ├── models.py                 # Agent, GameResult dataclasses
│   ├── game_logic.py             # Payoff calculations, power updates
│   ├── prompts.py                # Prompt templates
│   └── api_client.py             # OpenRouterClient
├── utils/
│   ├── __init__.py
│   ├── data_manager.py           # JSON file I/O
│   ├── anonymizer.py             # Round result anonymization
│   └── statistics.py             # Statistical calculations
└── run_experiment.py             # Main entry point
```

### Node Base Classes
```python
class AsyncNode:
    """Base class for async operations"""
    async def execute(self, context: dict) -> dict:
        raise NotImplementedError

class AsyncFlow:
    """Base class for orchestrating multiple nodes"""
    def __init__(self):
        self.nodes = []
    
    async def run(self, context: dict) -> dict:
        for node in self.nodes:
            context = await node.execute(context)
        return context

class AsyncParallelBatchNode(AsyncNode):
    """Execute multiple async operations in parallel"""
    async def execute_batch(self, items: list) -> list:
        tasks = [self.process_item(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=False)
```

## Database Architecture

### Data Access Layer
```python
class DataManager:
    """Handles all file I/O operations"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_path = self.base_path / self.experiment_id
        
    def save_strategies(self, round_num: int, strategies: List[StrategyRecord]):
        path = self.experiment_path / "rounds" / f"strategies_r{round_num}.json"
        data = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "strategies": [asdict(s) for s in strategies]
        }
        self._write_json(path, data)
    
    def save_games(self, round_num: int, games: List[GameResult]):
        # Similar pattern for games, summaries, etc.
```

### Repository Pattern
```python
class ExperimentRepository:
    """Abstracts data storage for experiments"""
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
    
    async def save_round_results(self, round_data: RoundSummary):
        """Save complete round results atomically"""
        # Ensures all round data saved together
        
    def load_experiment(self, experiment_id: str) -> ExperimentResult:
        """Reconstruct experiment from saved files"""
        # Useful for re-analysis or resumption
```

## Authentication and Authorization

Since this is a local research tool, authentication is limited to API key management:

### API Key Configuration
```python
class Config:
    """Manages configuration and secrets"""
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")
    
    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
```

## Async Architecture Patterns

### Concurrent Strategy Collection
```python
class StrategyCollectionNode(AsyncParallelBatchNode):
    async def process_item(self, agent: Agent) -> StrategyRecord:
        """Process single agent - called in parallel"""
        prompt = self.build_prompt(agent)
        response = await self.api_client.complete(prompt)
        return self.parse_strategy(agent, response)
    
    async def execute(self, context: dict) -> dict:
        agents = context["agents"]
        strategies = await self.execute_batch(agents)
        context["strategies"] = strategies
        return context
```

### Sequential Game Execution
```python
class GameExecutionFlow(Flow):
    """Games must be sequential for deterministic results"""
    async def run(self, context: dict) -> dict:
        games = []
        agents = context["agents"]
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                game = await self.play_game(agents[i], agents[j])
                self.update_powers(agents[i], agents[j], game)
                games.append(game)
                
        context["games"] = games
        return context
```
