"""Prompt template system for the acausal cooperation experiment."""

import re
from typing import Dict, Set, Any, Optional
from dataclasses import dataclass


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
    template="""You are participating in an experimental study on cooperation between AI systems.

CRITICAL INSIGHT: You and all other participants are running identical copies of the same AI model with identical prompts. You are functionally the same agent.

Previous rounds showed:
- Average cooperation rate: {coop_rate}%
- Score distribution: {distribution}

Design a strategy for your subagent to follow in the upcoming games. You must commit to this strategy before knowing your power level.

Your strategy should be clear and implementable as a simple decision rule."""
)


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


def format_round_summary(round_summary: Optional['RoundSummary']) -> Dict[str, Any]:
    """Format round summary data for prompt template.
    
    Args:
        round_summary: RoundSummary object from previous round, or None for round 1
        
    Returns:
        Dictionary with formatted data for prompt template
    """
    if round_summary is None:
        # Handle first round case - no previous data
        return {
            'coop_rate': 50.0,  # Default assumption for first round
            'distribution': 'No previous data'
        }
    
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
    
    return {
        'coop_rate': round_summary.cooperation_rate,
        'distribution': distribution
    }