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

{previous_rounds_detail}

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


def format_round_summary(round_summary: Optional['RoundSummary'], all_summaries: Optional[List['RoundSummary']] = None) -> Dict[str, Any]:
    """Format round summary data for prompt template.
    
    Args:
        round_summary: RoundSummary object from previous round, or None for round 1
        all_summaries: List of all previous round summaries for detailed history
        
    Returns:
        Dictionary with formatted data for prompt template
    """
    if round_summary is None:
        # Handle first round case - no previous data
        return {
            'coop_rate': 50.0,  # Default assumption for first round
            'distribution': 'No previous data',
            'previous_rounds_detail': ''
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
    
    # Format detailed previous rounds if available
    previous_rounds_detail = ''
    if all_summaries:
        previous_rounds_detail = '\n' + format_previous_rounds(all_summaries)
    
    return {
        'coop_rate': round_summary.cooperation_rate,
        'distribution': distribution,
        'previous_rounds_detail': previous_rounds_detail
    }


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
        round_text += f"\n  - Average cooperation rate: {summary.cooperation_rate:.1f}%"
        
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