# High Level Architecture

## Technical Summary

The Acausal Cooperation Experiment employs a modular Python-based architecture using an async flow framework to orchestrate prisoner's dilemma tournaments between LLM agents. The system leverages OpenRouter API for LLM access, implements a hierarchical node-based execution pattern for experiment orchestration, and outputs comprehensive JSON datasets for analysis. The architecture prioritizes simplicity, clear separation of concerns, and robust data collection to enable statistical analysis of superrational cooperation patterns. This design achieves the PRD goals by providing a controlled experimental environment with minimal external dependencies.

## Platform and Infrastructure Choice

**Platform:** Local Python Environment with Cloud API Access
- Chosen for rapid development and iteration
- No infrastructure management overhead
- Direct control over execution and debugging

**API Provider:** OpenRouter
- Unified access to multiple LLM models through single API
- Simplified billing and rate limiting
- Consistent interface across different model providers

**Storage:** Local JSON Files
- Simple, human-readable format
- No database complexity for small datasets
- Easy version control and sharing

## System Architecture Diagram

```mermaid
graph TB
    subgraph "Orchestration Layer"
        A[run_experiment.py] --> B[ExperimentFlow]
        B --> C[RoundFlow]
        C --> D[StrategyCollectionNode]
        C --> E[GameExecutionFlow]
        C --> F[RoundSummaryNode]
    end
    
    subgraph "Execution Layer"
        D --> G[API Client]
        E --> H[SubagentDecisionNode]
        H --> G
    end
    
    subgraph "External Services"
        G --> I[OpenRouter API]
        I --> J[LLM Models]
    end
    
    subgraph "Storage Layer"
        B --> K[DataManager]
        K --> L[JSON Files]
    end
    
    subgraph "Post-Processing"
        B --> M[Experiment Results]
    end

    subgraph "Analysis"
        M --> N[SimpleAnalysisNode]
        N --> O[Acausal Markers]
        N --> P[Cooperation Stats]
    end
```

## Architectural Patterns

1. **Node-Based Execution**: Each major operation is encapsulated in a node with standard interfaces
2. **Async/Await Concurrency**: Parallel API calls for strategy collection
3. **Flow Orchestration**: Hierarchical flows manage complex multi-step processes
4. **Separation of Concerns**: Clear boundaries between orchestration, execution, and storage
