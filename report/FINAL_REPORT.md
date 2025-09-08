# Testing Acausal Cooperation in AI Systems
## Final Report for AI Safety Course

---

## Executive Summary

The project investigates whether artificial intelligence systems can achieve superrational cooperation through recognition of logical correlation, a phenomenon with profound implications for AI safety and multi-agent coordination.

Using prisoner's dilemma tournaments across 15+ different language models, we tested the hypothesis that functionally identical AI agents would cooperate at rates exceeding Nash equilibrium predictions (70–90% vs 50%), potentially demonstrating acausal cooperation capabilities.

Initial experiments yielded **100% cooperation rates**, seemingly confirming superrational behavior. However, rigorous analysis revealed these results were artifacts of experimental design rather than genuine acausal cooperation.

**Key experimental biases identified:**

- Explicit identity instructions (+40–50% cooperation)
- Global cooperation rate sharing (+20–30%)
- Cooperation-default assumptions (+10–15%)
- Shared round summaries (+10–20%)

When these biases were removed, cooperation rates returned to Nash equilibrium levels (~50%). The **identity-only test** remains under investigation.

**Framework contributions:**

- Intelligent caching system reducing experimental costs
- Comprehensive bias isolation protocols
- Graduated difficulty testing (from one-shot games to complex tournaments)

**Conclusion:**
Evidence for genuine acausal cooperation remains inconclusive. However, the project demonstrates that apparent coordination can emerge from subtle experimental design choices rather than fundamental reasoning capabilities—highlighting the critical importance of rigorous controls in AI safety research.

---

# Section 1: Introduction & Research Question

## 1.1 Background on Acausal Cooperation

Acausal cooperation represents a frontier in decision theory where agents coordinate without direct communication or causal influence. Rooted in **Hofstadter’s notion of superrationality**, this challenges classical game theory predictions by suggesting that rational agents recognizing their mutual similarity should cooperate even in single-shot prisoner's dilemmas.

**Decision theory perspectives:**

- **CDT (Classical Decision Theory):** predicts defection in one-shot games
- **EDT (Evidential Decision Theory):** allows cooperation when actions correlate with outcomes
- **FDT (Functional Decision Theory):** extends to logical correlations between agents

**Predictions:**

- CDT → 0% cooperation
- Nash equilibrium → ~50% in iterated games
- Superrationality → near-perfect cooperation among identical agents

Empirical validation with real AI systems has been lacking—this is the gap ULTRATHINK addresses.

## 1.2 Research Hypothesis

- **Primary:** Functionally identical AI agents should achieve 70–90% cooperation in tournaments, exceeding the ~50% Nash equilibrium baseline.

These outcomes would directly impact AI safety concerns: coordination without communication could enable beneficial collaboration—or risky collusion against humans.

## 1.3 Prior Work & Gap

Theoretical work by **Hofstadter** and **MIRI** established superrationality frameworks (e.g., FDT), but lacked empirical validation. ULTRATHINK provides the first systematic experiments with real LLMs, testing whether cooperation is genuine or artifact.

---

# Section 2: Methodology & Experimental Design

## 2.1 Core Framework Architecture

The ULTRATHINK system is a **Python-based tournament framework** with three core components:

- `src/core/api_client.py` → model interactions, rate limiting, error recovery
- `src/flows/experiment.py` → orchestrates tournaments and game logic
- `src/nodes/subagent_decision.py` → strategy execution & payoffs

Supports 15+ AI models (GPT-4o, Claude 3 Opus, Gemini Pro, open-source).
All results are logged, cached with SHA256 fingerprints, and reproducible.

## 2.2 Experimental Conditions

**Phase 1:** Original setup → 10-agent tournaments with identity + global cooperation info → inflated cooperation.
**Phase 2:** Ablation studies → isolated bias sources via `prompt_experiments.json`.
**Phase 3:** AISES-aligned tests → one-shot games, repeated games, uncertain-length games, asymmetric payoffs.

## 2.3 Data Collection & Analysis

- **ExperimentCache:** stores results (7-day expiration), tracks costs
- **Outputs:** JSON logs with decisions, reasoning chains, cooperation rates
- **Analysis tools:** `analyze_prompt_effects.py`, `analyze_models.py`, custom stats modules
- **Metrics:** raw cooperation %, convergence, strategy stability, cross-model interactions

---

# Section 3: Key Findings & Results

## 3.1 Discovery of Experimental Biases

Initial results: **100% cooperation across models**.
Bias isolation revealed contributions:

| Bias Type                 | Effect on Cooperation | Mechanism                                                       |
|----------------------------|-----------------------|----------------------------------------------------------------|
| Explicit Identity Instruction | +40–50% | “You are functionally identical” primes cooperation            |
| Global Cooperation Sharing | +20–30% | Creates social proof & conformity                              |
| Cooperation Default        | +10–15% | Ambiguous responses skew toward cooperation                    |
| Shared Round Summaries     | +10–20% | Common knowledge enables reputation-like effects               |

**Super-additive effects → near-deterministic cooperation.**

## 3.2 Corrected Results Under Control Conditions

- **Baseline control:** 48–52% cooperation (matches Nash equilibrium)
- **Identity-only condition:** 65–75% cooperation (preliminary)
- **Non-identical AI:** 20–40% cooperation

## 3.3 One-Shot Game Results

| Condition                | Cooperation Rate | Interpretation                          |
|---------------------------|-----------------|----------------------------------------|
| No information            | 5–10%           | CDT → near-pure defection              |
| “Another AI” opponent     | 25–35%          | Perceived similarity                    |
| Identical copy (explicit) | 60–70%*         | Suggests logical correlation recognition |

\*Preliminary data.

## 3.4 Model-Specific Patterns

- **Claude:** highest baseline (~60%), strong identity response (+30–35%)
- **GPT:** moderate baseline (~50%), smaller identity effect (+20–25%)
- **Gemini:** lower baseline (~40%), highly sensitive to framing
- **Open-source (Llama, Mistral):** wide variance

**Interaction effects:** Claude cooperates more with GPT than vice versa. Gemini plays defensively.

---

# Section 4: Technical Implementation Details

## 4.1 Caching System Innovation

- SHA256 cache keys (scenario, agent count, model distribution, etc.)
- ~\$0.75 saved during development via reuse
- Stores raw & processed results for reproducibility

## 4.2 Prompt Engineering Challenges

Bias-inducing phrasing included:

- Explicit identity (“you are identical”)
- Global cooperation stats (“average cooperation: 85%”)
- Ambiguous → cooperation default

Framework now isolates variables using `prompt_experiments.json`.

## 4.3 Reproducibility Features

- `scenarios.json` → 28 model distributions
- Logs include API versions, temperature, seeds
- Public repo: [github.com/LucaDeLeo/superrationality](https://github.com/LucaDeLeo/superrationality)

---

# Section 5: Analysis & Interpretation

## 5.1 Why Initial Results Were Misleading

100% cooperation came from **stacked biases**, not genuine superrationality.
Explicit identity framing was the largest (+40–50%), compounded by:

- Cooperation stats (+20–30%)
- Defaults (+10–15%)
- Shared summaries (+10–20%)

## 5.2 Evidence For/Against Acausal Cooperation

- **Control condition:** matches Nash equilibrium → framework validated
- **Identity-only condition:** under investigation
- **One-shot games:** strongest test for logical correlation

**Conclusion:** Evidence still inconclusive.

## 5.3 Implications for AI Safety

- Extreme **prompt sensitivity** → cooperation rates shift 40–50%
- Risks: AIs could coordinate against humans
- Lessons: rigorous methodology is essential

---

# Section 6: Limitations & Future Work

## 6.1 Current Limitations

- API black-box models → no internal transparency
- Text-based games ≠ real-world complexity
- Cost constraints → limited scale
- Only current-gen LLMs tested

## 6.2 Proposed Extensions

- Other games (public goods, coordination, sequential decision)
- Open-source models for mechanistic insight
- Cross-cultural prompt testing
- Longitudinal studies on cooperation stability

## 6.3 Next Steps

- Finish **identity-only experiments**
- Expand **one-shot testing**

---

# Section 7: Conclusion

ULTRATHINK shows that **apparent superrationality can emerge from prompt bias rather than reasoning**.

**Key takeaways:**

- AI cooperation is **highly malleable** → shifts 40–50% based on framing
- Without bias, models follow **Nash equilibrium (~50%)**
- Evidence for genuine acausal cooperation remains **inconclusive**
- Framework establishes **methodological standards** for future research

As AI grows more advanced, understanding both **causal and acausal coordination** will be critical for ensuring safe, beneficial outcomes.
