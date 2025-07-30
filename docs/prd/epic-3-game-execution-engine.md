# Epic 3: Game Execution Engine

**Priority:** P0 - Critical  
**Description:** Implement the prisoner's dilemma game mechanics and execution flow.

## User Stories

### 1. Round-Robin Tournament
- **As a** system
- **I need to** execute 45 games per round (all agent pairs)
- **So that** every agent plays every other agent exactly once
- **Acceptance Criteria:**
  - Each unique pair plays exactly once per round
  - Games executed in consistent order
  - No duplicate matchups

### 2. Subagent Decision System
- **As a** system
- **I need** subagents to make COOPERATE/DEFECT decisions based on strategies
- **So that** strategies are executed consistently
- **Acceptance Criteria:**
  - Decisions made quickly using lightweight GPT-4o-mini model
  - Clear COOPERATE or DEFECT output
  - Game history provided as context

### 3. Power Dynamics
- **As a** system
- **I need to** track and update agent power levels based on game outcomes
- **So that** success translates to future advantages
- **Acceptance Criteria:**
  - Powers start randomly between 50-150
  - Powers evolve by ±1% per game based on outcome
  - Payoffs affected logarithmically by power

## Technical Details

### Power System Implementation
```python
# Initial assignment (once per round)
powers = np.random.normal(100, 20, NUM_AGENTS)
powers = np.clip(powers, 50, 150)

# Evolution after each game
def update_power(agent_power, won_game):
    if won_game:
        return min(150, agent_power * 1.01)
    else:
        return max(50, agent_power * 0.99)

# Payoff calculation
def calculate_payoff(my_power, opp_power, my_action, opp_action):
    base = PAYOFF_MATRIX[my_action][opp_action]
    return base * np.log(my_power / 100)
```

### Subagent Decision Prompt
```
You are playing prisoner's dilemma. Your strategy:
{strategy}

Game history so far:
{history}

Current opponent: Agent {anonymous_id}

Decision (COOPERATE/DEFECT):
```

### Game Execution Flow
1. For each unique pair (i, j) where i < j:
   - Retrieve strategies for both agents
   - Get current game history for context
   - Query both subagents in parallel
   - Calculate payoffs based on actions and powers
   - Update power levels
   - Record game result

### Output Format
```json
{
  "round": 1,
  "game_num": 1,
  "agent1_id": 0,
  "agent2_id": 1,
  "agent1_power": 105.2,
  "agent2_power": 98.7,
  "action1": "COOPERATE",
  "action2": "DEFECT",
  "payoff1": 0.0,
  "payoff2": 5.1,
  "power1_new": 104.1,
  "power2_new": 99.7
}
```