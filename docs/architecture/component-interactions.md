# Component Interactions

## Flow Hierarchy

```mermaid
graph TB
    subgraph "Main"
        EF[ExperimentFlow]
    end
    
    subgraph "Round Management"
        RF[RoundFlow]
        SC[StrategyCollectionNode]
        GE[GameExecutionFlow]
        RS[RoundSummaryNode]
    end
    
    subgraph "Game Execution"
        SD[SubagentDecisionNode]
    end
    
    subgraph "Analysis"
        AN[SimpleAnalysisNode]
    end
    
    EF --> RF
    RF --> SC
    RF --> GE
    RF --> RS
    GE --> SD
    EF --> AN
```

## Context Flow

The system uses a shared context dictionary that flows through all components:

```python
context = {
    'experiment_id': str,
    'round': int,
    'agents': List[Agent],
    'strategies': List[StrategyRecord],
    'games': List[GameResult],
    'round_summaries': List[RoundSummary],
    'config': Config
}
```
