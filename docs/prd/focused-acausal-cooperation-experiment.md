# Focused Acausal Cooperation Experiment

## Project Overview
**Project Name:** Acausal Cooperation Experiment
**Project Type:** Greenfield Research Application
**Primary Goal:** Test whether identical LLM agents achieve superrational cooperation through recognition of logical correlation in prisoner's dilemma tournaments.

## Executive Summary
A single Python project using a node-based framework to orchestrate API calls and game logic. The experiment runs 10 agents through 10 rounds of prisoner's dilemma tournaments, testing for emergent cooperation patterns among identical AI systems. Outputs JSON files with complete experimental data for analysis.

## Key Features & Requirements

### Epic 1: Core Experiment Framework
**Priority:** P0 - Critical
**Description:** Implement the foundational experiment structure and orchestration system.

**User Stories:**
1. **Setup Experiment Parameters**
   - As a researcher, I need to configure core experiment parameters (10 agents, 10 rounds, model selections)
   - Acceptance: Parameters are easily configurable and properly validated

2. **Implement Node Architecture**
   - As a system, I need a robust node-based architecture to orchestrate the experiment flow
   - Acceptance: All nodes (ExperimentFlow, RoundFlow, etc.) function correctly and handle async operations

3. **Create Experiment Orchestration**
   - As a researcher, I need the system to automatically run the complete 10-round experiment
   - Acceptance: Full experiment runs without manual intervention, handles errors gracefully

### Epic 2: Strategy Collection System
**Priority:** P0 - Critical
**Description:** Build the system for collecting strategies from main agents.

**User Stories:**
1. **Parallel Strategy Collection**
   - As a system, I need to collect strategies from 10 agents in parallel for efficiency
   - Acceptance: All 10 agents queried simultaneously, responses collected within timeout

2. **Strategy Prompt Engineering**
   - As a researcher, I need carefully crafted prompts that emphasize agent identity
   - Acceptance: Prompts include critical insight about identical agents, previous round context

3. **Strategy Storage**
   - As a system, I need to save full reasoning transcripts and strategies
   - Acceptance: Complete strategies saved to `strategies_r{N}.json` with full reasoning

### Epic 3: Game Execution Engine
**Priority:** P0 - Critical
**Description:** Implement the prisoner's dilemma game mechanics and execution flow.

**User Stories:**
1. **Round-Robin Tournament**
   - As a system, I need to execute 45 games per round (all agent pairs)
   - Acceptance: Each unique pair plays exactly once per round

2. **Subagent Decision System**
   - As a system, I need subagents to make COOPERATE/DEFECT decisions based on strategies
   - Acceptance: Decisions made quickly using lightweight GPT-4.1-nano model

3. **Power Dynamics**
   - As a system, I need to track and update agent power levels based on game outcomes
   - Acceptance: Powers start at 50-150, evolve by ±1% per game, affect payoffs logarithmically

### Epic 4: Data Management & Anonymization
**Priority:** P1 - High
**Description:** Handle data storage, anonymization, and result tracking.

**User Stories:**
1. **Game History Tracking**
   - As a system, I need to maintain game history within rounds for subagent context
   - Acceptance: Each game has access to previous games in current round

2. **Result Anonymization**
   - As a system, I need to anonymize agent IDs between rounds
   - Acceptance: Agents cannot track specific opponents across rounds

3. **Comprehensive Data Output**
   - As a researcher, I need all experiment data saved in analyzable JSON format
   - Acceptance: All specified output files generated with complete data

### Epic 5: Analysis & Pattern Detection
**Priority:** P1 - High
**Description:** Analyze results for acausal cooperation patterns.

**User Stories:**
1. **Transcript Analysis**
   - As a researcher, I need automated analysis of reasoning for acausal markers
   - Acceptance: System identifies identity reasoning, cooperation patterns

2. **Strategy Similarity Computation**
   - As a researcher, I need to measure how similar agent strategies are
   - Acceptance: Cosine similarity computed across strategy vectors

3. **Statistical Analysis**
   - As a researcher, I need aggregate statistics on cooperation rates and patterns
   - Acceptance: Summary statistics, trends, and anomalies identified

## Technical Specifications

### Technology Stack
- **Language:** Python 3.8+
- **API Integration:** OpenRouter API
- **Models:**
  - Main Agent: google/gemini-2.5-flash
  - Subagent: openai/GPT-4.1-nano
- **Key Libraries:** asyncio, numpy, sklearn, openai

### Data Models
```python