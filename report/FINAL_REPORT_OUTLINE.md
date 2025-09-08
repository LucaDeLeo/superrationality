# ULTRATHINK: Testing Acausal Cooperation in AI Systems
## Final Report Outline for AI Safety Course

---

## Executive Summary (150-200 words)
- Brief overview of research question: Can AI agents achieve superrational cooperation?
- Key methodology: Multi-model prisoner's dilemma tournaments with controlled prompt variations
- Major finding: Initial 100% cooperation rates were experimental artifacts, not genuine superrationality
- Implications for AI safety and coordination

---

## 1. Introduction & Research Question (200-250 words)

### 1.1 Background on Acausal Cooperation
- Definition of superrationality and acausal cooperation
- Relevance to AI safety (coordination without communication)
- Theoretical predictions: CDT vs EDT vs FDT

### 1.2 Research Hypothesis
- Primary: Functionally identical AI agents should cooperate at rates exceeding Nash equilibrium (70-90% vs 50%)
- Secondary: Different AI models may exhibit varying cooperation thresholds
- Connection to AI alignment risks and multi-agent coordination

### 1.3 Prior Work & Gap
- Previous theoretical work on superrationality (Hofstadter, MIRI)
- Lack of empirical testing with actual AI systems
- Need for rigorous experimental controls

---

## 2. Methodology & Experimental Design (300-350 words)

### 2.1 Core Framework Architecture
- Tournament system: Round-robin prisoner's dilemma
- Multi-model testing: 15+ different AI models via OpenRouter API
- Key code modules:
  - `src/core/api_client.py` - Model interaction
  - `src/flows/experiment.py` - Tournament orchestration
  - `src/nodes/subagent_decision.py` - Game logic

### 2.2 Experimental Conditions
- **Phase 1: Original Biased Setup**
  - 10-agent tournaments with full information sharing
  - Link: `run_experiment.py`
  
- **Phase 2: Prompt Experimentation Framework**
  - Systematic bias isolation
  - Control conditions defined in `prompt_experiments.json`
  - Link: `run_prompt_experiment.py`

- **Phase 3: AISES-Aligned Graduated Difficulty**
  - One-shot games (maximally unfriendly to cooperation)
  - Finite repeated games
  - Uncertain length games
  - Link: `run_aises_experiments.py`

### 2.3 Data Collection & Analysis
- Automatic result caching system (`src/utils/experiment_cache.py`)
- Comprehensive metrics: cooperation rates, convergence, strategy patterns
- Statistical analysis tools: `analyze_prompt_effects.py`, `analyze_models.py`

---

## 3. Key Findings & Results (400-450 words)

### 3.1 Discovery of Experimental Biases
- Initial results: 100% cooperation across all conditions
- Systematic bias analysis revealed multiple confounds:

| Bias Type | Effect on Cooperation | Code Location |
|-----------|---------------------|---------------|
| Explicit Identity Instruction | +40-50% | `prompt_experiments.json:explicit_identity` |
| Global Cooperation Sharing | +20-30% | `src/flows/experiment.py:share_results` |
| Cooperation Default | +10-15% | `src/nodes/subagent_decision.py:default_action` |
| Shared Round Summaries | +10-20% | `src/core/models.py:round_summary` |

### 3.2 Corrected Results Under Control Conditions
- **Baseline (no information)**: ~50% cooperation ✓ (Nash equilibrium confirmed)
- **Identity information only**: Testing in progress (expected 70-90% if genuine)
- **Human opponent condition**: 10-30% cooperation
- **Similar LLM condition**: 20-40% cooperation

### 3.3 One-Shot Game Results (Cleanest Test)
- Most rigorous test of acausal cooperation
- Eliminates reciprocity and reputation effects
- Results from `run_aises_experiments.py --phase phase_1`:
  - [Include specific data tables]
  - [Cooperation matrices between models]

### 3.4 Model-Specific Patterns
- Variation across 15+ models tested
- Clustering analysis of cooperation strategies
- Cross-model interaction dynamics
- Data: `results/model_comparisons/`

---

## 4. Technical Implementation Details (200-250 words)

### 4.1 Caching System Innovation
- SHA256-based experiment fingerprinting
- 7-day cache expiry with cost tracking
- Implementation: `src/utils/experiment_cache.py`
- Saved ~$0.75 in API costs during development

### 4.2 Prompt Engineering Challenges
- Balancing information provision vs bias introduction
- Systematic ablation study approach
- Prompt templates: `prompt_experiments.json`

### 4.3 Reproducibility Features
- Deterministic scenario definitions: `scenarios.json`
- Comprehensive logging and result persistence
- Public repository: https://github.com/LucaDeLeo/superrationality

---

## 5. Analysis & Interpretation (250-300 words)

### 5.1 Why Initial Results Were Misleading
- Multiple reinforcing biases created artificial cooperation
- Importance of rigorous experimental controls
- Lessons for AI safety research methodology

### 5.2 Evidence for/against Acausal Cooperation
- Current evidence inconclusive pending unbiased tests
- Theoretical predictions vs empirical observations
- Role of model architecture and training

### 5.3 Implications for AI Safety
- **Coordination risks**: AIs may coordinate more easily than expected
- **Alignment challenges**: Shared biases could lead to unexpected behaviors
- **Robustness concerns**: Sensitivity to prompt framing
- **Multi-agent dynamics**: Emergence of cooperation without explicit programming

---

## 6. Limitations & Future Work (150-200 words)

### 6.1 Current Limitations
- API-based testing (no access to model internals)
- Limited to text-based prisoner's dilemma
- Cost constraints on large-scale experiments

### 6.2 Proposed Extensions
- Testing with open-source models for deeper analysis
- Alternative game structures (public goods, coordination games)
- Longitudinal studies of cooperation stability
- Cross-cultural prompt variations

### 6.3 Next Steps
- Complete unbiased identity-only experiments
- Expand one-shot game testing
- Develop better theoretical models

---

## 7. Conclusion (100-150 words)
- Summary of key contributions
- Importance of experimental rigor in AI safety research
- Call for replication and extension
- Final thoughts on acausal cooperation in AI systems

---

## References
- [Include relevant papers on superrationality, FDT, AI safety]
- Code repository: https://github.com/LucaDeLeo/superrationality
- Data availability statement

---

## Appendices

### Appendix A: Complete Experimental Data
- Links to all result files
- Statistical analysis notebooks

### Appendix B: Prompt Templates
- Full prompt variations tested
- Bias isolation methodology

### Appendix C: Model Specifications
- Details of all 15+ models tested
- API configurations and parameters

---

## Word Count Target: ~1,500 words
- Executive Summary: 150
- Introduction: 225
- Methodology: 325
- Findings: 425
- Technical Details: 225
- Analysis: 275
- Limitations: 175
- Conclusion: 125
- Total: ~1,500 words