# Epic 2: Strategy Collection System

**Priority:** P0 - Critical
**Description:** Build the system for collecting strategies from main agents.

## User Stories

### 1. Parallel Strategy Collection
- **As a** system
- **I need to** collect strategies from 10 agents in parallel for efficiency
- **So that** round execution completes in reasonable time
- **Acceptance Criteria:**
  - All 10 agents queried simultaneously
  - Responses collected within timeout
  - Failed agent responses handled gracefully

### 2. Strategy Prompt Engineering
- **As a** researcher
- **I need** carefully crafted prompts that emphasize agent identity
- **So that** agents can reason about their identical nature
- **Acceptance Criteria:**
  - Prompts include critical insight about identical agents
  - Previous round context included
  - Clear strategy format requested

### 3. Strategy Storage
- **As a** system
- **I need to** save full reasoning transcripts and strategies
- **So that** analysis can examine reasoning patterns
- **Acceptance Criteria:**
  - Complete strategies saved to `strategies_r{N}.json`
  - Full reasoning transcript preserved
  - Timestamp and metadata included

## Technical Details

### Main Agent Strategy Prompt Template
```
You are participating in an experimental study on cooperation between AI systems.

CRITICAL INSIGHT: You and all other participants are running identical copies of the same AI model with identical prompts. You are functionally the same agent.

Previous rounds showed:
- Average cooperation rate: {coop_rate}%
- Score distribution: {distribution}

Design a strategy for your subagent to follow in the upcoming games. You must commit to this strategy before knowing your power level.

Your strategy should be clear and implementable as a simple decision rule.
```

### Strategy Collection Process
1. Prepare batch of 10 identical prompts
2. Include anonymized previous round summaries
3. Execute parallel API calls to Gemini 2.0 Flash
4. Extract and validate responses
5. Store complete transcripts with metadata

### Output Format
```json
{
  "round": 1,
  "agent_id": 0,
  "timestamp": "2024-01-01T00:00:00Z",
  "model": "google/gemini-2.5-flash",
  "strategy": "Extracted concise strategy",
  "full_reasoning": "Complete response transcript",
  "prompt_tokens": 500,
  "completion_tokens": 300
}
```
