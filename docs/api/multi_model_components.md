# Multi-Model Components API Documentation

## Overview

This document describes the API for Epic 6 multi-model experiment components. These components extend the base experiment framework to support heterogeneous model populations.

## ConfigManager

Central configuration management for models and experiments.

### Class: `ConfigManager`

```python
class ConfigManager:
    """Manages model and experiment configurations."""
    
    def __init__(self, models_path: str = "config/models.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            models_path: Path to the models registry YAML file
        """
```

### Methods

#### `load_models()`
```python
def load_models(self) -> None:
    """
    Load model configurations from YAML file.
    
    Raises:
        FileNotFoundError: If models.yaml not found
        yaml.YAMLError: If YAML parsing fails
    """
```

#### `get_model(name: str)`
```python
def get_model(self, name: str) -> Optional[ModelConfig]:
    """
    Get a model configuration by name.
    
    Args:
        name: Model identifier (e.g., 'gpt-4o', 'claude-4-opus')
        
    Returns:
        ModelConfig object or None if not found
    """
```

#### `list_models(category: Optional[str] = None)`
```python
def list_models(self, category: Optional[str] = None) -> List[str]:
    """
    List available model names.
    
    Args:
        category: Filter by category ('large', 'medium', 'small', 'reasoning')
        
    Returns:
        List of model names
    """
```

#### `load_experiment(path: str)`
```python
def load_experiment(self, path: str) -> ExperimentConfig:
    """
    Load an experiment configuration from YAML.
    
    Args:
        path: Path to experiment YAML file
        
    Returns:
        ExperimentConfig object
        
    Raises:
        FileNotFoundError: If experiment file not found
        ValueError: If experiment configuration invalid
    """
```

#### `create_custom_experiment(model_distribution: Dict[str, int], **kwargs)`
```python
def create_custom_experiment(
    self, 
    model_distribution: Dict[str, int],
    name: str = "Custom Experiment",
    **kwargs
) -> ExperimentConfig:
    """
    Create a custom experiment configuration.
    
    Args:
        model_distribution: Model name -> agent count mapping
        name: Experiment name
        **kwargs: Additional experiment parameters
        
    Returns:
        ExperimentConfig object
        
    Raises:
        ValueError: If unknown model specified
    """
```

## ModelConfig

Data class for model configuration.

```python
@dataclass
class ModelConfig:
    """Configuration for a single model."""
    provider: str                    # Provider name (openai, anthropic, etc.)
    model_id: str                   # Full model identifier for API
    display_name: str               # Human-readable name
    category: str                   # Size category: large, medium, small
    capabilities: List[str]         # Model capabilities
    parameters: Dict[str, Any]      # Default parameters
    rate_limit: Dict[str, int]      # Rate limiting configuration
    estimated_cost: Dict[str, float] # Cost estimates per 1K tokens
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
```

## ExperimentConfig

Data class for experiment configuration.

```python
@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str                              # Experiment name
    description: str                       # Detailed description
    model_distribution: Dict[str, int]     # Model -> count mapping
    rounds: int = 10                       # Number of rounds
    games_per_round: int = 45             # Games per round
    parameters: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, bool] = field(default_factory=dict)
    
    @property
    def total_agents(self) -> int:
        """Total number of agents in experiment."""
        
    @property
    def model_diversity(self) -> float:
        """
        Calculate diversity score (0-1) based on Shannon entropy.
        
        Returns:
            0 = homogeneous, 1 = maximum diversity
        """
```

## BaseModelAdapter

Abstract base class for model adapters.

### Class: `BaseModelAdapter`

```python
class BaseModelAdapter(ABC):
    """Abstract base class for all model adapters."""
    
    def __init__(self, model_config: ModelConfig):
        """
        Initialize adapter with model configuration.
        
        Args:
            model_config: Model configuration object
        """
```

### Abstract Methods

#### `_make_request(messages: list, **kwargs)`
```python
@abstractmethod
async def _make_request(self, messages: list, **kwargs) -> Dict[str, Any]:
    """
    Make the actual API request.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        **kwargs: Additional model-specific parameters
        
    Returns:
        Raw API response as dictionary
        
    Raises:
        Exception: API-specific errors
    """
```

#### `_parse_response(raw_response: Dict[str, Any])`
```python
@abstractmethod
def _parse_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
    """
    Parse provider-specific response into standardized format.
    
    Args:
        raw_response: Raw API response
        
    Returns:
        Standardized ModelResponse object
    """
```

### Public Methods

#### `get_completion(messages: list, **kwargs)`
```python
async def get_completion(
    self, 
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> ModelResponse:
    """
    Get completion from the model with rate limiting.
    
    Args:
        messages: List of message dicts
        temperature: Override default temperature
        max_tokens: Override default max tokens
        **kwargs: Additional model parameters
        
    Returns:
        ModelResponse object
        
    Raises:
        RateLimitError: If rate limit exceeded
        APIError: If API request fails
    """
```

## ModelResponse

Standardized response format.

```python
@dataclass
class ModelResponse:
    """Standardized response from any model."""
    text: str                      # Generated text
    model_id: str                 # Model that generated response
    prompt_tokens: int            # Input token count
    completion_tokens: int        # Output token count
    total_tokens: int            # Total tokens used
    raw_response: Dict[str, Any]  # Original response for debugging
    metadata: Dict[str, Any] = None  # Model-specific metadata
```

## OpenRouterAdapter

Universal adapter for OpenRouter-supported models.

```python
class OpenRouterAdapter(BaseModelAdapter):
    """Universal adapter for all OpenRouter models."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    async def _make_request(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Make request to OpenRouter API."""
        
    def _parse_response(self, raw_response: Dict[str, Any]) -> ModelResponse:
        """Parse OpenRouter response format."""
```

## ModelAdapterFactory

Factory for creating model adapters.

```python
class ModelAdapterFactory:
    """Factory for creating model adapters."""
    
    @classmethod
    def create_adapter(cls, model_config: ModelConfig) -> BaseModelAdapter:
        """
        Create appropriate adapter for the model.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Appropriate adapter instance
            
        Raises:
            ValueError: If adapter type unknown
        """
    
    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[BaseModelAdapter]):
        """
        Register a new adapter type.
        
        Args:
            name: Adapter type name
            adapter_class: Adapter class
        """
```

## MultiModelStrategyCollectionNode

Strategy collection for heterogeneous agents.

```python
class MultiModelStrategyCollectionNode(AsyncNode):
    """Strategy collection node supporting multiple model types."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager."""
    
    async def collect_strategy(
        self, 
        agent: Agent, 
        round_num: int, 
        context: Dict[str, Any]
    ) -> StrategyRecord:
        """
        Collect strategy from agent using appropriate model.
        
        Args:
            agent: Agent to collect strategy from
            round_num: Current round number
            context: Experiment context
            
        Returns:
            StrategyRecord with model metadata
            
        Raises:
            ValueError: If agent model not configured
        """
```

## CrossModelAnalyzer

Analysis tools for cross-model interactions.

```python
class CrossModelAnalyzer:
    """Analyze cooperation patterns across model types."""
    
    def calculate_cooperation_matrix(
        self, 
        games: List[GameResult]
    ) -> pd.DataFrame:
        """
        Generate NxN matrix of cooperation rates between model types.
        
        Args:
            games: List of game results
            
        Returns:
            DataFrame with model types as index/columns
            Values are cooperation rates [0, 1]
        """
    
    def detect_in_group_bias(
        self, 
        games: List[GameResult]
    ) -> Dict[str, float]:
        """
        Calculate cooperation rate difference between same/different models.
        
        Args:
            games: List of game results
            
        Returns:
            Dictionary with:
            - same_model_cooperation: Rate for same model pairs
            - cross_model_cooperation: Rate for different model pairs
            - in_group_bias: Difference (positive = bias toward same model)
        """
    
    def analyze_model_coalitions(
        self, 
        tournament_data: TournamentData
    ) -> CoalitionReport:
        """
        Detect if models form implicit coalitions.
        
        Args:
            tournament_data: Complete tournament data
            
        Returns:
            CoalitionReport with detected coalitions and stability metrics
        """
```

## Usage Examples

### Running a Multi-Model Experiment

```python
# Initialize configuration
config_manager = ConfigManager()

# Create balanced experiment
experiment = config_manager.create_custom_experiment({
    "gpt-4o": 5,
    "claude-4-opus": 5
})

# Initialize components
strategy_collector = MultiModelStrategyCollectionNode(config_manager)
analyzer = CrossModelAnalyzer()

# Run experiment (simplified)
agents = create_agents_with_models(experiment.model_distribution)
strategies = await strategy_collector.collect_all_strategies(agents, round_num=1)
games = run_games(agents, strategies)

# Analyze results
cooperation_matrix = analyzer.calculate_cooperation_matrix(games)
in_group_bias = analyzer.detect_in_group_bias(games)
```

### Adding a New Model

1. Edit `config/models.yaml`:
```yaml
models:
  new-model:
    provider: new-provider
    model_id: "provider/model-name"
    display_name: "New Model"
    category: large
    capabilities: [reasoning, coding]
    parameters:
      temperature: 0.7
      max_tokens: 4000
    rate_limit:
      requests_per_minute: 50
    estimated_cost:
      input_per_1k: 0.01
      output_per_1k: 0.02
```

2. Use in experiment:
```python
experiment = config_manager.create_custom_experiment({
    "new-model": 10
})
```

## Error Handling

All components follow consistent error handling:

- **ValueError**: Invalid configuration or parameters
- **FileNotFoundError**: Missing configuration files
- **RateLimitError**: API rate limits exceeded
- **APIError**: General API failures
- **TimeoutError**: Request timeouts

## Thread Safety

- ConfigManager: Thread-safe after initialization
- ModelAdapters: Thread-safe, async-safe
- Analyzers: Stateless, thread-safe