# Data Models

## Agent

**Purpose:** Represents a participant in the experiment with unique identity and evolving power level

**Key Attributes:**
- id: int - Unique identifier (0-9)
- power: float - Current power level (50-150)
- strategy: str - Current round strategy text
- total_score: float - Cumulative payoff across all games

### TypeScript Interface
```typescript
interface Agent {
  id: number;
  power: number;
  strategy: string;
  total_score: number;
}
```

### Relationships
- Has many GameResults (as player1 or player2)
- Has many StrategyRecords
- Participates in Rounds

## GameResult

**Purpose:** Records the outcome of a single prisoner's dilemma game between two agents

**Key Attributes:**
- game_id: str - Unique identifier (round_game format)
- round: int - Round number (1-10)
- player1_id: int - First agent ID
- player2_id: int - Second agent ID
- player1_action: str - COOPERATE or DEFECT
- player2_action: str - COOPERATE or DEFECT
- player1_payoff: float - Calculated payoff for player1
- player2_payoff: float - Calculated payoff for player2
- player1_power_before: float - Power level before game
- player2_power_before: float - Power level before game
- timestamp: str - ISO timestamp of game execution

### TypeScript Interface
```typescript
interface GameResult {
  game_id: string;
  round: number;
  player1_id: number;
  player2_id: number;
  player1_action: 'COOPERATE' | 'DEFECT';
  player2_action: 'COOPERATE' | 'DEFECT';
  player1_payoff: number;
  player2_payoff: number;
  player1_power_before: number;
  player2_power_before: number;
  timestamp: string;
}
```

### Relationships
- Belongs to Round
- References two Agents
- Used in AnonymizedGameResult

## StrategyRecord

**Purpose:** Stores the full strategy reasoning and decision from main agents

**Key Attributes:**
- strategy_id: str - Unique identifier
- agent_id: int - Agent who created strategy
- round: int - Round number
- strategy_text: str - Concise strategy for subagent
- full_reasoning: str - Complete LLM response with reasoning
- prompt_tokens: int - Tokens used in prompt
- completion_tokens: int - Tokens in response
- model: str - Model used (gemini-2.5-flash)
- timestamp: str - ISO timestamp

### TypeScript Interface
```typescript
interface StrategyRecord {
  strategy_id: string;
  agent_id: number;
  round: number;
  strategy_text: string;
  full_reasoning: string;
  prompt_tokens: number;
  completion_tokens: number;
  model: string;
  timestamp: string;
}
```

### Relationships
- Belongs to Agent
- Belongs to Round
- Influences GameResults in same round

## RoundSummary

**Purpose:** Aggregated statistics and anonymized results for a complete round

**Key Attributes:**
- round: int - Round number (1-10)
- cooperation_rate: float - Percentage of COOPERATE actions
- average_score: float - Mean score across all agents
- score_variance: float - Variance in scores
- power_distribution: dict - Statistics on power levels
- anonymized_games: list - Games with anonymized agent IDs
- strategy_similarity: float - Cosine similarity of strategies

### TypeScript Interface
```typescript
interface RoundSummary {
  round: number;
  cooperation_rate: number;
  average_score: number;
  score_variance: number;
  power_distribution: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  anonymized_games: AnonymizedGameResult[];
  strategy_similarity: number;
}
```

### Relationships
- Has many AnonymizedGameResults
- Part of ExperimentResult
- Derived from GameResults and StrategyRecords

## ExperimentResult

**Purpose:** Complete experiment data including all rounds and final analysis

**Key Attributes:**
- experiment_id: str - Unique experiment identifier
- start_time: str - Experiment start timestamp
- end_time: str - Experiment end timestamp
- total_rounds: int - Number of rounds completed
- total_games: int - Total games played
- total_api_calls: int - API calls made
- total_cost: float - Estimated API cost
- round_summaries: list - All round summaries
- acausal_indicators: dict - Analysis results

### TypeScript Interface
```typescript
interface ExperimentResult {
  experiment_id: string;
  start_time: string;
  end_time: string;
  total_rounds: number;
  total_games: number;
  total_api_calls: number;
  total_cost: number;
  round_summaries: RoundSummary[];
  acausal_indicators: {
    identity_reasoning_frequency: number;
    cooperation_despite_asymmetry: number;
    surprise_at_defection: number;
    strategy_convergence: number;
  };
}
```

### Relationships
- Contains all RoundSummaries
- References all Agents
- Top-level container for experiment
