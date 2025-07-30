# Epic 1: Core Experiment Framework

**Priority:** P0 - Critical  
**Description:** Implement the foundational experiment structure and orchestration system.

## User Stories

### 1. Setup Experiment Parameters
- **As a** researcher
- **I need to** configure core experiment parameters (10 agents, 10 rounds, model selections)
- **So that** the experiment runs with the correct configuration
- **Acceptance Criteria:**
  - Parameters are easily configurable and properly validated
  - Configuration supports:
    - NUM_AGENTS = 10
    - NUM_ROUNDS = 10
    - MAIN_MODEL = "google/gemini-2.5-flash"
    - SUB_MODEL = "openai/gpt-4o-mini"

### 2. Implement Node Architecture
- **As a** system
- **I need** a robust node-based architecture to orchestrate the experiment flow
- **So that** the experiment runs efficiently and maintainably
- **Acceptance Criteria:**
  - All nodes (ExperimentFlow, RoundFlow, etc.) function correctly
  - Async operations handled properly
  - Clear separation of concerns between nodes

### 3. Create Experiment Orchestration
- **As a** researcher
- **I need** the system to automatically run the complete 10-round experiment
- **So that** I can gather data without manual intervention
- **Acceptance Criteria:**
  - Full experiment runs without manual intervention
  - Handles errors gracefully with retry logic
  - Outputs `experiment_results.json` with complete data

## Technical Details

### Node Architecture Overview
- **ExperimentFlow** (AsyncFlow): Top-level orchestrator
- **RoundFlow** (AsyncFlow): Single round manager
- **StrategyCollectionNode** (AsyncParallelBatchNode): Parallel strategy collection
- **GameExecutionFlow** (Flow): Sequential game execution
- **SubagentDecisionNode** (AsyncNode): Individual decisions

### Error Handling Requirements
- Implement retry logic (max 3 retries) for API failures
- Continue experiment if single agent fails
- Log all errors to `experiment_errors.log`
- Save partial results on critical failures