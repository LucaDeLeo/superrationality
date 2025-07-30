# Product Requirements Document
## Focused Acausal Cooperation Experiment

### Project Overview
**Project Name:** Acausal Cooperation Experiment  
**Project Type:** Greenfield Research Application  
**Primary Goal:** Test whether identical LLM agents achieve superrational cooperation through recognition of logical correlation in prisoner's dilemma tournaments.

### Executive Summary
A single Python project using a node-based framework to orchestrate API calls and game logic. The experiment runs 10 agents through 10 rounds of prisoner's dilemma tournaments, testing for emergent cooperation patterns among identical AI systems. Outputs JSON files with complete experimental data for analysis.

### Key Features & Requirements

#### Epic 1: Core Experiment Framework
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

#### Epic 2: Strategy Collection System
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

#### Epic 3: Game Execution Engine
**Priority:** P0 - Critical  
**Description:** Implement the prisoner's dilemma game mechanics and execution flow.

**User Stories:**
1. **Round-Robin Tournament**
   - As a system, I need to execute 45 games per round (all agent pairs)
   - Acceptance: Each unique pair plays exactly once per round

2. **Subagent Decision System**
   - As a system, I need subagents to make COOPERATE/DEFECT decisions based on strategies
   - Acceptance: Decisions made quickly using lightweight GPT-4o-mini model

3. **Power Dynamics**
   - As a system, I need to track and update agent power levels based on game outcomes
   - Acceptance: Powers start at 50-150, evolve by ±1% per game, affect payoffs logarithmically

#### Epic 4: Data Management & Anonymization
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

#### Epic 5: Analysis & Pattern Detection
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

### Technical Specifications

#### Technology Stack
- **Language:** Python 3.8+
- **API Integration:** OpenRouter API
- **Models:** 
  - Main Agent: google/gemini-2.0-flash-exp:free
  - Subagent: openai/gpt-4o-mini
- **Key Libraries:** asyncio, numpy, sklearn, openai

#### Data Models
```python
# Agent State
{
    "id": int,
    "power": float,  # 50-150
    "strategy": str,
    "total_score": float
}

# Game Result
{
    "round": int,
    "game_num": int,
    "agent1_id": int,
    "agent2_id": int,
    "action1": "COOPERATE|DEFECT",
    "action2": "COOPERATE|DEFECT",
    "payoff1": float,
    "payoff2": float
}
```

#### API Specifications
- **OpenRouter Endpoints:** /api/v1/chat/completions
- **Rate Limits:** Handle 429 errors with exponential backoff
- **Cost Controls:** Abort if total cost exceeds $10

### Implementation Timeline

**Week 1: Core Framework**
- Days 1-2: OpenRouter API integration and basic node structure
- Days 3-4: Implement StrategyCollectionNode and SubagentDecisionNode
- Day 5: Build GameExecutionFlow and test single round

**Week 2: Complete System**
- Days 1-2: Add power dynamics and payoff calculations
- Days 3-4: Implement anonymization and multi-round flow
- Day 5: Comprehensive logging and error handling

**Week 3: Analysis & Testing**
- Days 1-2: Build AnalysisNode and pattern detection
- Days 3-4: Run test experiments and refine prompts
- Day 5: Generate final analysis reports

### Success Metrics
1. **Technical Success:**
   - Complete 10-round experiment without failures
   - All data properly collected and stored
   - Analysis runs without errors

2. **Research Success:**
   - Clear patterns in cooperation rates
   - Evidence of identity-based reasoning
   - Measurable strategy convergence

### Constraints & Limitations
- No web interface (CLI only)
- No real-time monitoring (console progress only)
- No database (JSON file storage)
- No checkpoint/resume capability
- Single experiment configuration (no parameter sweeps)
- Fixed model selection

### Risk Mitigation
1. **API Failures:** Implement retry logic with exponential backoff
2. **Cost Overruns:** Monitor costs, abort if exceeding $10
3. **Data Loss:** Save results after each round
4. **Model Inconsistency:** Use deterministic temperature settings

### Future Enhancements (Out of Scope)
- Web-based monitoring dashboard
- Multi-model experiments
- Parameter sweep capabilities
- Database integration
- Checkpoint/resume functionality
- Real-time analysis visualization

### Acceptance Criteria
- [ ] Complete experiment runs end-to-end without manual intervention
- [ ] All specified output files are generated with correct data
- [ ] Analysis correctly identifies cooperation patterns
- [ ] Total cost remains under $10 per experiment
- [ ] Documentation sufficient for replication

### Dependencies
- OpenRouter API access with sufficient credits
- Python environment with required packages
- Sufficient disk space for JSON outputs (~100MB per experiment)

---
*Document Version: 1.0*  
*Last Updated: [Current Date]*  
*Status: Ready for Development*