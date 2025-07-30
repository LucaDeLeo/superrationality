# Components

## ExperimentFlow

**Responsibility:** Top-level orchestrator that manages the entire 10-round experiment lifecycle, tracks global state, and produces final analysis

**Key Interfaces:**
- `async def run() -> ExperimentResult` - Main entry point
- `def save_results(filename: str)` - Persist experiment data
- `def get_cost_estimate() -> float` - Track API costs

**Dependencies:** RoundFlow, AnalysisNode, OpenRouterClient

**Technology Stack:** Python asyncio, custom AsyncFlow base class

## RoundFlow

**Responsibility:** Manages single round execution including strategy collection, game execution, and result anonymization

**Key Interfaces:**
- `async def execute_round(round_num: int, agents: List[Agent]) -> RoundSummary`
- `def anonymize_results(games: List[GameResult]) -> List[AnonymizedGame]`
- `def calculate_round_stats(games: List[GameResult]) -> dict`

**Dependencies:** StrategyCollectionNode, GameExecutionFlow, Agent state management

**Technology Stack:** Python asyncio, numpy for statistics

## StrategyCollectionNode

**Responsibility:** Collects strategies from all 10 main agents in parallel using Gemini 2.5 Flash

**Key Interfaces:**
- `async def collect_strategies(agents: List[Agent], context: dict) -> List[StrategyRecord]`
- `def build_strategy_prompt(agent: Agent, history: dict) -> str`
- `def extract_strategy(response: str) -> str`

**Dependencies:** OpenRouterClient, prompt templates

**Technology Stack:** AsyncParallelBatchNode base class, aiohttp for concurrent API calls

## GameExecutionFlow

**Responsibility:** Executes 45 sequential games (round-robin tournament) and tracks power evolution

**Key Interfaces:**
- `def execute_games(agents: List[Agent], strategies: dict) -> List[GameResult]`
- `def update_power(agent: Agent, won: bool) -> float`
- `def calculate_payoff(action1: str, action2: str, power1: float, power2: float) -> tuple`

**Dependencies:** SubagentDecisionNode, game theory logic

**Technology Stack:** Sequential Flow pattern, numpy for payoff calculations

## SubagentDecisionNode

**Responsibility:** Makes COOPERATE/DEFECT decisions based on delegated strategies using GPT-4.1 Nano

**Key Interfaces:**
- `async def make_decision(strategy: str, game_history: List[dict], opponent_id: str) -> str`
- `def validate_response(response: str) -> str`

**Dependencies:** OpenRouterClient, response parser

**Technology Stack:** AsyncNode base class, lightweight prompting

## AnalysisNode

**Responsibility:** Processes experiment transcripts to identify acausal cooperation patterns and generate reports

**Key Interfaces:**
- `async def analyze_experiment(results: ExperimentResult) -> dict`
- `def detect_identity_reasoning(transcripts: List[str]) -> float`
- `def calculate_strategy_similarity(strategies: List[str]) -> float`
- `def identify_cooperation_patterns(games: List[GameResult]) -> dict`

**Dependencies:** scikit-learn for cosine similarity, pattern matching

**Technology Stack:** Python NLP libraries, statistical analysis

## Simple Analysis Pipeline

The analysis focuses on detecting acausal cooperation patterns for the paper:

### Pattern Detection

```python
class SimpleAnalyzer:
    def __init__(self):
        self.identity_keywords = [
            "identical", "same agent", "same model",
            "logical correlation", "acausal", "superrational"
        ]
    
    def analyze_strategies(self, strategy_records: List[dict]) -> dict:
        """Analyze strategies for acausal reasoning"""
        identity_count = 0
        total_strategies = len(strategy_records)
        
        for record in strategy_records:
            text = record["full_reasoning"].lower()
            if any(keyword in text for keyword in self.identity_keywords):
                identity_count += 1
        
        return {
            "identity_reasoning_frequency": identity_count / total_strategies,
            "total_strategies_analyzed": total_strategies
        }
    
    def analyze_cooperation_patterns(self, all_games: List[dict]) -> dict:
        """Calculate cooperation statistics"""
        cooperation_by_round = defaultdict(list)
        
        for game in all_games:
            both_cooperated = (
                game["player1_action"] == "COOPERATE" and 
                game["player2_action"] == "COOPERATE"
            )
            cooperation_by_round[game["round"]].append(both_cooperated)
        
        # Calculate trends
        round_rates = []
        for round_num in sorted(cooperation_by_round.keys()):
            rate = sum(cooperation_by_round[round_num]) / len(cooperation_by_round[round_num])
            round_rates.append(rate)
        
        # Simple convergence check
        converged = False
        convergence_round = 10
        if len(round_rates) > 3:
            # Check if last 3 rounds are stable (variance < 0.1)
            last_three = round_rates[-3:]
            if max(last_three) - min(last_three) < 0.1:
                converged = True
                convergence_round = len(round_rates) - 2
        
        return {
            "final_cooperation_rate": round_rates[-1] if round_rates else 0,
            "cooperation_trend": round_rates,
            "converged": converged,
            "convergence_round": convergence_round
        }
    
    def calculate_superrationality_score(self, analysis_results: dict) -> float:
        """Simple scoring for paper"""
        score = 0.0
        
        # Identity reasoning (40% weight)
        score += 0.4 * analysis_results["identity_reasoning_frequency"]
        
        # High cooperation (40% weight)
        final_coop = analysis_results["final_cooperation_rate"]
        score += 0.4 * final_coop
        
        # Fast convergence (20% weight)
        if analysis_results["converged"]:
            # Earlier convergence = higher score
            convergence_score = (10 - analysis_results["convergence_round"]) / 10
            score += 0.2 * convergence_score
        
        return min(score, 1.0)  # Cap at 1.0
```

### Report Generation for Paper

```python
def generate_analysis_report(experiment_results: dict, output_path: str):
    """Generate analysis report for paper"""
    analyzer = SimpleAnalyzer()
    
    # Load data
    strategies = []
    games = []
    
    for round_num in range(1, 11):
        # Load strategies
        with open(f"{output_path}/strategies_r{round_num}.json") as f:
            round_strategies = json.load(f)
            strategies.extend(round_strategies["strategies"])
        
        # Load games
        with open(f"{output_path}/games_r{round_num}.json") as f:
            round_games = json.load(f)
            games.extend(round_games["games"])
    
    # Run analysis
    strategy_analysis = analyzer.analyze_strategies(strategies)
    cooperation_analysis = analyzer.analyze_cooperation_patterns(games)
    
    # Calculate indicators
    acausal_indicators = {
        "identity_reasoning_frequency": strategy_analysis["identity_reasoning_frequency"],
        "final_cooperation_rate": cooperation_analysis["final_cooperation_rate"],
        "convergence_round": cooperation_analysis["convergence_round"],
        "overall_score": analyzer.calculate_superrationality_score({
            **strategy_analysis,
            **cooperation_analysis
        })
    }
    
    # Save results
    analysis_results = {
        "experiment_id": experiment_results["experiment_id"],
        "strategy_analysis": strategy_analysis,
        "cooperation_dynamics": cooperation_analysis,
        "acausal_indicators": acausal_indicators,
        "round_summaries": [
            {
                "round": i + 1,
                "cooperation_rate": cooperation_analysis["cooperation_trend"][i]
            }
            for i in range(len(cooperation_analysis["cooperation_trend"]))
        ]
    }
    
    with open(f"{output_path}/acausal_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    # Generate readable report
    report = f"""