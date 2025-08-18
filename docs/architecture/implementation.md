# Implementation

## Directory Structure

```
acausal/
├── nodes/
│   ├── __init__.py
│   ├── base.py                    # AsyncNode, AsyncFlow base classes
│   ├── strategy_collection.py     # StrategyCollectionNode
│   ├── subagent_decision.py      # SubagentDecisionNode
│   └── simple_analysis.py        # SimpleAnalysisNode
├── flows/
│   ├── __init__.py
│   ├── experiment.py             # ExperimentFlow
│   ├── round.py                  # RoundFlow
│   └── game_execution.py         # GameExecutionFlow
├── core/
│   ├── __init__.py
│   ├── models.py                 # Data models (Agent, GameResult, etc.)
│   ├── config.py                 # Configuration
│   └── api_client.py             # OpenRouterClient
├── utils/
│   ├── __init__.py
│   ├── data_manager.py           # JSON file I/O
│   └── anonymizer.py             # Round result anonymization
└── run_experiment.py             # Main entry point
```

### Node Base Classes

```python
class AsyncNode(ABC):
    """Base class for async operations with retry logic"""
    
    @abstractmethod
    async def _execute_impl(self, context: dict) -> dict:
        """Implementation to be provided by subclasses"""
        pass
    
    async def execute(self, context: dict) -> dict:
        """Execute with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return await self._execute_impl(context)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise

class AsyncFlow:
    """Base class for orchestrating multiple nodes"""
    
    def add_node(self, node: AsyncNode) -> 'AsyncFlow':
        """Add node to flow"""
        self.nodes.append(node)
        return self
    
    async def run(self, context: dict) -> dict:
        """Run all nodes in sequence"""
        for node in self.nodes:
            context = await node.execute(context)
        return context
```
