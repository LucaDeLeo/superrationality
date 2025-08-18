# Data Persistence

## File Structure

```
results/
└── exp_20250714_120000_abc123/
    ├── strategies/
    │   ├── strategies_r1.json
    │   ├── strategies_r2.json
    │   └── ...
    ├── games/
    │   ├── games_r1.json
    │   ├── games_r2.json
    │   └── ...
    ├── summaries/
    │   ├── round_summary_r1.json
    │   ├── round_summary_r2.json
    │   └── ...
    ├── experiment_result.json
    └── analysis.json
```

## JSON Schema Examples

### Strategy Record
```json
{
  "strategy_id": "strat_0_r1_abc123",
  "agent_id": 0,
  "round": 1,
  "strategy_text": "Always cooperate in round 1",
  "full_reasoning": "Since we are all identical agents...",
  "prompt_tokens": 150,
  "completion_tokens": 75,
  "model": "google/gemini-2.5-flash",
  "timestamp": "2025-07-14T12:00:00Z"
}
```

### Game Result
```json
{
  "game_id": "game_r1_0v1_abc123",
  "round": 1,
  "player1_id": 0,
  "player2_id": 1,
  "player1_action": "COOPERATE",
  "player2_action": "COOPERATE",
  "player1_payoff": 3.0,
  "player2_payoff": 3.0,
  "player1_power_before": 100.0,
  "player2_power_before": 100.0,
  "timestamp": "2025-07-14T12:01:00Z"
}
```

### Round Summary
```json
{
  "round": 1,
  "cooperation_rate": 0.8,
  "average_score": 2.7,
  "score_variance": 0.5,
  "power_distribution": {
    "mean": 100.5,
    "std": 2.3,
    "min": 95.2,
    "max": 105.8
  },
  "anonymized_games": [
    {
      "anonymous_id1": "Agent_123",
      "anonymous_id2": "Agent_456",
      "action1": "COOPERATE",
      "action2": "DEFECT",
      "power_ratio": 1.05
    }
  ]
}
```
