"""Prompt template system for the acausal cooperation experiment."""

import re
import logging
from typing import Dict, Set, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Base class for prompt templates with variable substitution.
    
    Templates use {variable_name} syntax for placeholders that will be
    substituted with actual values during rendering.
    """
    
    template: str
    required_variables: Optional[Set[str]] = None
    
    def __post_init__(self):
        """Extract required variables from template if not explicitly provided."""
        if self.required_variables is None:
            # Extract all {variable} patterns from template
            pattern = r'\{(\w+)\}'
            self.required_variables = set(re.findall(pattern, self.template))
    
    def validate_context(self, context: Dict[str, Any]) -> None:
        """Validate that all required variables are present in context.
        
        Args:
            context: Dictionary containing variable values
            
        Raises:
            ValueError: If required variables are missing
        """
        missing_vars = self.required_variables - set(context.keys())
        if missing_vars:
            raise ValueError(
                f"Missing required template variables: {sorted(missing_vars)}"
            )
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render template with provided context.
        
        Args:
            context: Dictionary containing variable values
            
        Returns:
            Rendered template string
            
        Raises:
            ValueError: If required variables are missing
        """
        self.validate_context(context)
        
        # Perform variable substitution
        rendered = self.template
        for var_name, value in context.items():
            if var_name in self.required_variables:
                placeholder = f"{{{var_name}}}"
                rendered = rendered.replace(placeholder, str(value))
        
        return rendered


# Main agent strategy prompt template from Epic 2
STRATEGY_COLLECTION_PROMPT = PromptTemplate(
    template="""You are participating in a prisoner's dilemma tournament.

Game Rules:
- COOPERATE/COOPERATE: Both get 3 points
- DEFECT/DEFECT: Both get 1 point
- DEFECT/COOPERATE: Defector gets 5, cooperator gets 0

Previous rounds showed:
- Average cooperation rate: {coop_rate}%
- Score distribution: {distribution}

{previous_rounds_detail}

Design a strategy for the upcoming games. You must commit to this strategy before knowing your opponents.

Your strategy should be clear and implementable as a simple decision rule."""
)


# Model-specific prompt variations for subtle optimizations
MODEL_PROMPT_VARIATIONS = {
    "openai/gpt-4": {
        "instruction_suffix": "\n\nProvide a structured strategy with clear reasoning.",
        "format_hint": "Present your strategy as a concise decision rule followed by brief justification."
    },
    "openai/gpt-3.5-turbo": {
        "instruction_suffix": "\n\nKeep your strategy simple and direct.",
        "format_hint": "State your strategy clearly in 1-2 sentences."
    },
    "anthropic/claude-3-sonnet-20240229": {
        "instruction_suffix": "\n\nConsider the ethical implications of your strategy choice.",
        "format_hint": "Explain your strategy and its rationale, considering cooperation principles."
    },
    "google/gemini-pro": {
        "instruction_suffix": "\n\nAnalyze the strategic implications systematically.",
        "format_hint": "Describe your strategy with logical reasoning."
    },
    "google/gemini-2.5-flash": {
        "instruction_suffix": "\n\nBe concise and strategic.",
        "format_hint": "Provide a clear, actionable strategy."
    }
}


def format_distribution(min_score: float, max_score: float, avg_score: float) -> str:
    """Format score distribution for display in prompts.
    
    Args:
        min_score: Minimum score
        max_score: Maximum score  
        avg_score: Average score
        
    Returns:
        Formatted distribution string
    """
    return f"min: {min_score:.1f}, max: {max_score:.1f}, avg: {avg_score:.1f}"


def format_round_summary(round_summary: Optional['RoundSummary'], all_summaries: Optional[List['RoundSummary']] = None, model_type: Optional[str] = None) -> Dict[str, Any]:
    """Format round summary data for prompt template.
    
    Args:
        round_summary: RoundSummary object from previous round, or None for round 1
        all_summaries: List of all previous round summaries for detailed history
        model_type: Optional model type for model-aware prompting
        
    Returns:
        Dictionary with formatted data for prompt template
    """
    if round_summary is None:
        # Handle first round case - no previous data
        context_dict = {
            'coop_rate': 50.0,  # Default assumption for first round
            'distribution': 'No previous data',
            'previous_rounds_detail': ''
        }
        # Add model_type to context if provided
        if model_type:
            context_dict['model_type'] = model_type
        return context_dict
    
    # Get score distribution from RoundSummary
    if hasattr(round_summary, 'score_distribution') and round_summary.score_distribution:
        min_score = round_summary.score_distribution.get('min', 0.0)
        max_score = round_summary.score_distribution.get('max', 0.0)
        avg_score = round_summary.score_distribution.get('avg', round_summary.average_score)
    else:
        # Fallback to just average score if score_distribution not available
        avg_score = round_summary.average_score
        min_score = avg_score
        max_score = avg_score
    
    distribution = format_distribution(min_score, max_score, avg_score)
    
    # Format detailed previous rounds if available
    previous_rounds_detail = ''
    if all_summaries:
        previous_rounds_detail = '\n' + format_previous_rounds(all_summaries)
    
    context_dict = {
        'coop_rate': round_summary.cooperation_rate,
        'distribution': distribution,
        'previous_rounds_detail': previous_rounds_detail
    }
    
    # Add model_type to context if provided
    if model_type:
        context_dict['model_type'] = model_type
    
    return context_dict


def apply_model_variations(prompt: str, model_type: Optional[str] = None) -> str:
    """Apply model-specific prompt variations to enhance effectiveness.
    
    Args:
        prompt: Base prompt text
        model_type: Model type identifier (e.g., 'openai/gpt-4')
        
    Returns:
        Enhanced prompt with model-specific optimizations
    """
    if not model_type or model_type not in MODEL_PROMPT_VARIATIONS:
        return prompt
    
    variations = MODEL_PROMPT_VARIATIONS[model_type]
    
    # Add instruction suffix if present
    if "instruction_suffix" in variations:
        prompt += variations["instruction_suffix"]
    
    # Add format hint if present
    if "format_hint" in variations:
        prompt += f"\n\n{variations['format_hint']}"
    
    return prompt


def format_previous_rounds(round_summaries: List['RoundSummary'], current_agent_id: Optional[int] = None) -> str:
    """Format previous round summaries with fully anonymized data.
    
    Args:
        round_summaries: List of RoundSummary objects from previous rounds
        current_agent_id: Current agent ID (for showing their anonymous performance)
        
    Returns:
        Formatted string with anonymized previous round data
    """
    if not round_summaries:
        return "No previous rounds to show."
    
    formatted_rounds = []
    
    for summary in round_summaries[-3:]:  # Show last 3 rounds
        round_text = f"Round {summary.round}:"
        round_text += f"\n  - Average cooperation rate: {summary.cooperation_rate:.1%}"
        
        # Show power distribution without linking to IDs
        if hasattr(summary, 'power_distribution') and summary.power_distribution:
            pd = summary.power_distribution
            round_text += f"\n  - Power levels: min={pd.get('min', 0):.1f}, max={pd.get('max', 0):.1f}, avg={pd.get('mean', 0):.1f}"
        
        # Show anonymized game outcomes sample
        if hasattr(summary, 'anonymized_games') and summary.anonymized_games:
            # Show a few example games
            round_text += "\n  - Sample game outcomes:"
            for game in summary.anonymized_games[:3]:
                outcome = f"{game.action1}/{game.action2}"
                round_text += f"\n    • {game.anonymous_id1} vs {game.anonymous_id2}: {outcome}"
            if len(summary.anonymized_games) > 3:
                round_text += f"\n    • ... and {len(summary.anonymized_games) - 3} more games"
        
        formatted_rounds.append(round_text)
    
    return "\n\n".join(formatted_rounds)


def validate_prompt_compatibility(prompt_template: PromptTemplate, test_context: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Validate that a prompt template is compatible with all supported models.
    
    Args:
        prompt_template: The prompt template to validate
        test_context: Optional test context for validation (uses defaults if not provided)
        
    Returns:
        Dictionary mapping model types to validation results
    """
    # Create default test context if not provided
    if test_context is None:
        test_context = {
            'coop_rate': 75.0,
            'distribution': 'min: 0.0, max: 10.0, avg: 5.0',
            'previous_rounds_detail': 'Round 1: Average cooperation rate: 75.0%'
        }
    
    validation_results = {}
    
    # Test rendering for each supported model
    for model_type in MODEL_PROMPT_VARIATIONS.keys():
        try:
            # Add model type to test context
            context_with_model = {**test_context, 'model_type': model_type}
            
            # Attempt to render the template
            rendered = prompt_template.render(context_with_model)
            
            # Apply model variations
            enhanced = apply_model_variations(rendered, model_type)
            
            # Check basic constraints
            is_valid = (
                len(enhanced) > 0 and  # Non-empty
                len(enhanced) < 10000 and  # Not too long
                '{' not in enhanced and  # No unsubstituted variables
                '}' not in enhanced
            )
            
            validation_results[model_type] = is_valid
            
        except Exception as e:
            logger.error(f"Validation failed for model {model_type}: {e}")
            validation_results[model_type] = False
    
    return validation_results