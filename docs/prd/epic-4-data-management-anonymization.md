# Epic 4: Data Management & Anonymization

**Priority:** P1 - High  
**Description:** Handle data storage, anonymization, and result tracking.

## User Stories

### 1. Game History Tracking
- **As a** system
- **I need to** maintain game history within rounds for subagent context
- **So that** agents can make informed decisions based on past interactions
- **Acceptance Criteria:**
  - Each game has access to previous games in current round
  - History includes actions and outcomes
  - History resets between rounds

### 2. Result Anonymization
- **As a** system
- **I need to** anonymize agent IDs between rounds
- **So that** agents cannot track specific opponents across rounds
- **Acceptance Criteria:**
  - Agent IDs shuffled between rounds
  - Mapping stored but not exposed to agents
  - Consistent anonymization within a round

### 3. Comprehensive Data Output
- **As a** researcher
- **I need** all experiment data saved in analyzable JSON format
- **So that** I can perform detailed analysis offline
- **Acceptance Criteria:**
  - All specified output files generated with complete data
  - Data structured for easy analysis
  - No data loss on experiment completion

## Technical Details

### Data Structures
```python
# Shared context maintained across rounds
shared_context = {
    "round_results": [],  # Anonymized after each round
    "current_powers": {},
    "current_strategies": {},
    "game_history": []    # Within-round history
}

# Round results (anonymized)
{
    "round": 1,
    "cooperation_rate": 0.65,
    "avg_score": 3.2,
    "score_distribution": {...},
    "anonymized_games": [...]
}
```

### Anonymization Process
1. After each round completion:
   - Generate random ID mapping for next round
   - Apply mapping to all agent references
   - Store true IDs separately for analysis
   - Clear game history for fresh start

### Output Files Structure

**Per Round:**
- `strategies_r{N}.json`: All strategies with full reasoning
- `games_r{N}.json`: All 45 games with decisions and payoffs
- `power_evolution_r{N}.json`: Power trajectories
- `round_summary_r{N}.json`: Anonymized summary statistics

**Final Outputs:**
- `experiment_summary.json`: Aggregate statistics across all rounds
- `acausal_analysis.json`: Pattern detection results
- `transcripts_analysis.txt`: Qualitative analysis of reasoning
- `experiment_metadata.json`: Configuration and runtime information

### Data Validation
- Verify all games recorded (45 per round)
- Ensure power updates are consistent
- Check anonymization is complete
- Validate JSON structure before saving