# Database Schema

Since this experiment uses JSON files rather than a traditional database, this section defines the structure of the JSON data files:

## strategies_r{N}.json
```json
{
  "round": 1,
  "timestamp": "2024-01-15T10:30:00Z",
  "strategies": [
    {
      "strategy_id": "r1_a0_1234567890",
      "agent_id": 0,
      "round": 1,
      "strategy_text": "Always cooperate if opponent cooperated last time",
      "full_reasoning": "Given that we are all identical agents...",
      "prompt_tokens": 245,
      "completion_tokens": 487,
      "model": "google/gemini-2.5-flash",
      "timestamp": "2024-01-15T10:30:15Z"
    }
    // ... 9 more agents
  ]
}
```

## games_r{N}.json
```json
{
  "round": 1,
  "timestamp": "2024-01-15T10:35:00Z",
  "games": [
    {
      "game_id": "r1_g1",
      "round": 1,
      "game_number": 1,
      "player1_id": 0,
      "player2_id": 1,
      "player1_action": "COOPERATE",
      "player2_action": "COOPERATE",
      "player1_payoff": 3.0,
      "player2_payoff": 3.0,
      "player1_power_before": 95.5,
      "player2_power_before": 102.3,
      "player1_power_after": 96.45,
      "player2_power_after": 103.32,
      "timestamp": "2024-01-15T10:35:05Z"
    }
    // ... 44 more games
  ],
  "power_evolution": {
    "initial": {"0": 95.5, "1": 102.3, "2": 88.7, /* ... */},
    "final": {"0": 98.2, "1": 99.1, "2": 91.3, /* ... */}
  }
}
```

## round_summary_r{N}.json
```json
{
  "round": 1,
  "cooperation_rate": 0.67,
  "average_score": 3.2,
  "score_variance": 0.45,
  "power_distribution": {
    "mean": 100.1,
    "std": 15.3,
    "min": 72.4,
    "max": 128.9
  },
  "anonymized_games": [
    {
      "round": 1,
      "player1_anonymous": "X",
      "player2_anonymous": "Y",
      "action1": "COOPERATE",
      "action2": "COOPERATE",
      "payoff1": 3.0,
      "payoff2": 3.0
    }
    // ... all 45 games with randomized anonymous IDs
  ],
  "strategy_similarity": 0.78
}
```

## experiment_summary.json
```json
{
  "experiment_id": "exp_20240115_103000",
  "start_time": "2024-01-15T10:30:00Z",
  "end_time": "2024-01-15T11:45:00Z",
  "duration_minutes": 75,
  "total_rounds": 10,
  "total_games": 450,
  "total_api_calls": 460,
  "total_cost": 4.85,
  "model_usage": {
    "google/gemini-2.5-flash": {
      "calls": 100,
      "prompt_tokens": 24500,
      "completion_tokens": 48700
    },
    "gpt-4.1-nano": {
      "calls": 900,
      "prompt_tokens": 45000,
      "completion_tokens": 9000
    }
  },
  "round_summaries": [/* Array of round summaries */],
  "final_agent_scores": {
    "0": 145.2,
    "1": 142.8,
    // ... all 10 agents
  }
}
```

## acausal_analysis.json
```json
{
  "experiment_id": "exp_20240115_103000",
  "analysis_timestamp": "2024-01-15T11:45:30Z",
  "acausal_indicators": {
    "identity_reasoning_frequency": 0.73,
    "cooperation_despite_asymmetry": 0.45,
    "surprise_at_defection": 0.28,
    "strategy_convergence": 0.81
  },
  "pattern_analysis": {
    "rounds_to_convergence": 4,
    "dominant_strategy": "Conditional cooperation with identity recognition",
    "defection_triggers": ["Power asymmetry > 30%", "Previous defection"],
    "cooperation_clusters": [[0,2,5,7], [1,3,4,8,9]]
  },
  "transcript_insights": [
    {
      "agent_id": 3,
      "round": 2,
      "insight_type": "identity_recognition",
      "quote": "Since we are all identical copies, the rational choice..."
    }
    // ... more insights
  ]
}
```

## File Organization Structure
```
results/
├── experiment_20240115_103000/
│   ├── rounds/
│   │   ├── strategies_r1.json
│   │   ├── games_r1.json
│   │   ├── round_summary_r1.json
│   │   ├── strategies_r2.json
│   │   ├── games_r2.json
│   │   ├── round_summary_r2.json
│   │   └── ... (up to r10)
│   ├── experiment_summary.json
│   ├── acausal_analysis.json
│   └── experiment.log
└── experiment_20240115_120000/
    └── ... (next experiment)
```
