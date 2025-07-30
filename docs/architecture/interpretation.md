# Interpretation
{"Strong evidence" if acausal_indicators['overall_score'] > 0.7 else "Moderate evidence" if acausal_indicators['overall_score'] > 0.4 else "Weak evidence"} of acausal cooperation.
"""
    
    with open(f"{output_path}/analysis_report.txt", "w") as f:
        f.write(report)
    
    return analysis_results
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Experiment Control"
        EF[ExperimentFlow]
        RF[RoundFlow]
    end
    
    subgraph "Strategy Phase"
        SC[StrategyCollectionNode]
        PM[PromptManager]
    end
    
    subgraph "Game Phase"
        GE[GameExecutionFlow]
        SD[SubagentDecisionNode]
        GL[GameLogic]
    end
    
    subgraph "Data Layer"
        DM[DataManager]
        JP[JSONPersistence]
    end
    
    subgraph "Analysis"
        AN[AnalysisNode]
        RP[ReportGenerator]
    end
    
    EF --> RF
    RF --> SC
    RF --> GE
    SC --> PM
    GE --> SD
    GE --> GL
    SD --> GL
    RF --> DM
    DM --> JP
    EF --> AN
    AN --> RP
    
    SC -.-> API[OpenRouter API]
    SD -.-> API
```
