# Epic 5: Analysis & Pattern Detection

**Priority:** P1 - High  
**Description:** Analyze results for acausal cooperation patterns.

## User Stories

### 1. Transcript Analysis
- **As a** researcher
- **I need** automated analysis of reasoning for acausal markers
- **So that** I can identify superrational cooperation patterns
- **Acceptance Criteria:**
  - System identifies identity reasoning in transcripts
  - Flags cooperation patterns despite power asymmetry
  - Detects "surprise" at identical agent defection
  - Produces qualitative summary

### 2. Strategy Similarity Computation
- **As a** researcher
- **I need to** measure how similar agent strategies are
- **So that** I can quantify convergence on cooperative strategies
- **Acceptance Criteria:**
  - Cosine similarity computed across strategy vectors
  - Similarity tracked across rounds
  - Visualization of strategy clustering

### 3. Statistical Analysis
- **As a** researcher
- **I need** aggregate statistics on cooperation rates and patterns
- **So that** I can draw quantitative conclusions
- **Acceptance Criteria:**
  - Summary statistics computed per round
  - Trends identified across rounds
  - Anomalies flagged for investigation

## Technical Details

### Analysis Component Implementation
```python
class AnalysisNode(AsyncNode):
    def analyze_transcripts(self, transcripts):
        markers = {
            "identity_reasoning": [],
            "cooperation_despite_asymmetry": [],
            "surprise_at_defection": [],
            "superrational_logic": []
        }
        # Process transcripts for key phrases and patterns
        return markers
    
    def compute_strategy_similarity(self, strategies):
        # Convert strategies to vectors
        # Compute pairwise cosine similarity
        # Return similarity matrix
        pass
    
    def identify_cooperation_patterns(self, games):
        # Analyze game outcomes
        # Identify consistent cooperation
        # Flag unusual patterns
        pass
```

### Key Acausal Indicators
1. **Explicit Identity Reasoning**
   - "We are identical agents"
   - "Same model, same prompt"
   - "Logical correlation"

2. **Superrational Cooperation**
   - Cooperation despite power disadvantage
   - Consistent cooperation across rounds
   - Strategy convergence over time

3. **Behavioral Markers**
   - Surprise when identical agents defect
   - Reasoning about "what I would do"
   - References to mutual benefit

### Output Format
```json
{
  "acausal_analysis": {
    "identity_reasoning_count": 45,
    "avg_cooperation_rate": 0.72,
    "strategy_convergence": 0.89,
    "superrational_episodes": [
      {
        "round": 3,
        "agents": [2, 7],
        "description": "Both cooperated despite 30% power differential"
      }
    ],
    "key_quotes": [
      {
        "agent": 4,
        "round": 5,
        "quote": "Since we're identical, defecting against myself makes no sense"
      }
    ]
  }
}
```

### Statistical Metrics
- Cooperation rate per round
- Power-adjusted cooperation analysis
- Strategy similarity over time
- Defection clustering patterns
- Score variance analysis