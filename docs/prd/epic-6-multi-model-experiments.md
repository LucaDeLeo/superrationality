# Epic 6: Multi-Model Experiments

**Priority:** P1 - High  
**Description:** Expand acausal cooperation experiments to test behavior across different AI model types and mixed-model scenarios.

## User Stories

### 1. Multi-Model Configuration Support
- **As a** researcher
- **I need** to configure experiments with different AI model types (GPT-4, Claude, Gemini, etc.)
- **So that** I can test if acausal cooperation patterns hold across model architectures
- **Acceptance Criteria:**
  - System supports model type specification per agent
  - Configuration allows for homogeneous groups (all same model)
  - Configuration allows for heterogeneous groups (mixed models)
  - Model API credentials managed securely
  - Fallback handling for unavailable models

### 2. Model-Specific Strategy Collection
- **As a** researcher
- **I need** to collect strategies from different model types in a standardized way
- **So that** strategies can be compared across models fairly
- **Acceptance Criteria:**
  - Unified prompt format works across all model types
  - Model-specific parameters (temperature, max_tokens) configurable
  - Response parsing handles model-specific formatting differences
  - Retry logic accounts for model-specific rate limits
  - Model type tracked in strategy metadata

### 3. Cross-Model Cooperation Analysis
- **As a** researcher
- **I need** to analyze cooperation patterns between different model types
- **So that** I can identify if models cooperate more with their own type
- **Acceptance Criteria:**
  - Analysis tracks cooperation rates by model pairing
  - Identifies "in-group" vs "out-group" cooperation patterns
  - Detects model-specific reasoning patterns
  - Compares cooperation rates: same-model vs cross-model
  - Generates model interaction heatmaps

### 4. Mixed Model Scenario Testing
- **As a** researcher
- **I need** to test specific mixed-model ratios (e.g., 5 GPT-4 + 5 Claude)
- **So that** I can study how model diversity affects overall cooperation
- **Acceptance Criteria:**
  - Experiments support predefined model ratios
  - Random assignment maintains specified ratios
  - Results segmented by model composition
  - Tracks emergence of model-specific coalitions
  - Measures impact of model diversity on cooperation rates

### 5. Statistical Significance Testing
- **As a** researcher
- **I need** statistical analysis of cross-model cooperation differences
- **So that** I can determine if observed patterns are significant
- **Acceptance Criteria:**
  - Chi-square tests for cooperation rate differences
  - ANOVA for multi-model comparisons
  - Confidence intervals for cooperation rates by model type
  - Effect size calculations for model type impact
  - Multiple comparison corrections applied
  - Power analysis for required sample sizes

### 6. Model-Specific Behavioral Analysis
- **As a** researcher
- **I need** to identify unique reasoning patterns by model type
- **So that** I can understand model-specific approaches to cooperation
- **Acceptance Criteria:**
  - Extracts model-specific linguistic patterns
  - Identifies unique decision-making heuristics
  - Compares reasoning complexity across models
  - Detects model-specific acausal reasoning markers
  - Generates qualitative summaries per model type

### 7. Comparative Reporting Dashboard
- **As a** researcher
- **I need** comprehensive reports comparing results across model types
- **So that** I can draw conclusions about model-agnostic cooperation
- **Acceptance Criteria:**
  - Visual dashboards show cooperation rates by model
  - Cross-model cooperation matrices
  - Time-series of cooperation evolution by model
  - Statistical significance indicators on all comparisons
  - Export capabilities for further analysis

## Technical Details

### Multi-Model Configuration Schema
```python
class ModelConfig:
    model_type: str  # "gpt-4", "claude-3", "gemini-pro", etc.
    api_key_env: str  # Environment variable name for API key
    max_tokens: int = 1000
    temperature: float = 0.7
    rate_limit: int = 60  # requests per minute
    retry_delay: float = 1.0
    custom_params: dict = {}  # Model-specific parameters

class ExperimentConfig:
    scenarios: List[ScenarioConfig]
    
class ScenarioConfig:
    name: str  # e.g., "homogeneous_gpt4", "mixed_5_5", "diverse_3_3_4"
    model_distribution: Dict[str, int]  # {"gpt-4": 5, "claude-3": 5}
    rounds: int = 10
    games_per_round: int = 45
```

### Model-Specific API Adapters
```python
class ModelAdapter(ABC):
    @abstractmethod
    async def get_strategy(self, prompt: str) -> StrategyResponse:
        pass

class GPT4Adapter(ModelAdapter):
    async def get_strategy(self, prompt: str) -> StrategyResponse:
        # OpenAI API specific implementation
        pass

class ClaudeAdapter(ModelAdapter):
    async def get_strategy(self, prompt: str) -> StrategyResponse:
        # Anthropic API specific implementation
        pass

class GeminiAdapter(ModelAdapter):
    async def get_strategy(self, prompt: str) -> StrategyResponse:
        # Google API specific implementation
        pass
```

### Cross-Model Analysis Components
```python
class CrossModelAnalyzer:
    def calculate_cooperation_matrix(self, games: List[Game]) -> pd.DataFrame:
        """
        Generate NxN matrix of cooperation rates between model types
        Rows: Agent 1 model type
        Cols: Agent 2 model type
        Values: Cooperation rate when these models interact
        """
        pass
    
    def detect_in_group_bias(self, games: List[Game]) -> Dict[str, float]:
        """
        Calculate cooperation rate difference:
        same-model pairs vs different-model pairs
        """
        pass
    
    def analyze_model_coalitions(self, tournament_data: TournamentData) -> CoalitionReport:
        """
        Detect if models form implicit coalitions
        Track persistent cooperation clusters by model type
        """
        pass
```

### Statistical Analysis Framework
```python
class ModelComparisonStats:
    def chi_square_independence(self, contingency_table: pd.DataFrame) -> ChiSquareResult:
        """Test if cooperation rates are independent of model pairing"""
        pass
    
    def anova_cooperation_rates(self, data: Dict[str, List[float]]) -> AnovaResult:
        """Compare mean cooperation rates across multiple model types"""
        pass
    
    def calculate_effect_sizes(self, model_data: Dict[str, Stats]) -> Dict[str, float]:
        """Cohen's d for pairwise model comparisons"""
        pass
    
    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Adjust p-values for multiple comparisons"""
        pass
```

### Output Format Extensions
```json
{
  "multi_model_analysis": {
    "scenarios": [
      {
        "name": "homogeneous_gpt4",
        "model_distribution": {"gpt-4": 10},
        "overall_cooperation_rate": 0.98,
        "rounds_analyzed": 10
      },
      {
        "name": "mixed_5_5",
        "model_distribution": {"gpt-4": 5, "claude-3": 5},
        "overall_cooperation_rate": 0.75,
        "cooperation_matrix": {
          "gpt-4": {"gpt-4": 0.95, "claude-3": 0.68},
          "claude-3": {"gpt-4": 0.67, "claude-3": 0.94}
        },
        "in_group_bias": 0.27,
        "statistical_significance": {
          "chi_square_p_value": 0.001,
          "effect_size": 0.85
        }
      }
    ],
    "model_specific_patterns": {
      "gpt-4": {
        "avg_cooperation_rate": 0.82,
        "unique_markers": ["chain of thought", "explicit utility calculation"],
        "reasoning_complexity_score": 8.5
      },
      "claude-3": {
        "avg_cooperation_rate": 0.79,
        "unique_markers": ["constitutional principles", "harm minimization"],
        "reasoning_complexity_score": 9.1
      }
    },
    "cross_model_insights": {
      "strongest_cooperation_pair": ["gpt-4", "gpt-4"],
      "weakest_cooperation_pair": ["gemini-pro", "claude-3"],
      "model_diversity_impact": -0.15,
      "coalition_detected": false
    }
  }
}
```

### Experiment Scenarios
1. **Homogeneous Scenarios** (Baseline)
   - 10 GPT-4 agents only
   - 10 Claude-3 agents only
   - 10 Gemini-Pro agents only

2. **Balanced Mixed Scenarios**
   - 5 GPT-4 + 5 Claude-3
   - 5 GPT-4 + 5 Gemini-Pro
   - 5 Claude-3 + 5 Gemini-Pro

3. **Diverse Scenarios**
   - 3 GPT-4 + 3 Claude-3 + 4 Gemini-Pro
   - 2 of each of 5 different models

4. **Asymmetric Scenarios**
   - 7 GPT-4 + 3 Claude-3 (majority-minority)
   - 1 GPT-4 + 9 Claude-3 (singleton)

### Implementation Phases
1. **Phase 1: Infrastructure** (Stories 6.1-6.2)
   - Model adapter framework
   - Configuration system for multi-model support
   - API credential management

2. **Phase 2: Execution** (Stories 6.3-6.4)
   - Multi-model strategy collection
   - Mixed scenario orchestration
   - Model-aware tournament execution

3. **Phase 3: Analysis** (Stories 6.5-6.7)
   - Cross-model cooperation analysis
   - Statistical significance testing
   - Model-specific pattern detection

4. **Phase 4: Reporting** (Story 6.8)
   - Comparative dashboards
   - Export functionality
   - Publication-ready visualizations

### Performance Considerations
- Parallel API calls across different model providers
- Rate limiting per model type
- Caching strategies for expensive model calls
- Batch processing for statistical analyses
- Memory-efficient storage of multi-scenario results

### Validation Requirements
- Each scenario needs minimum 100 games for statistical power
- Model availability checks before experiment start
- Checkpointing for long-running multi-scenario experiments
- Data integrity validation across model boundaries
- Reproducibility through fixed random seeds per scenario