"""Subagent decision node for making game decisions."""

import logging
from typing import Dict, List, Any, Tuple, Optional

from .base import AsyncNode
from src.core.models import Agent, GameResult
from src.core.api_client import OpenRouterClient
from src.core.config import Config

logger = logging.getLogger(__name__)


class SubagentDecisionNode(AsyncNode):
    """Make game decisions using subagent with provided strategy."""
    
    # GPT-4.1-nano model identifier for OpenRouter
    GPT_4_1_NANO_MODEL = "openai/gpt-4.1-mini"
    
    def __init__(self, api_client: OpenRouterClient, config: Config):
        """Initialize with API client and config.
        
        Args:
            api_client: OpenRouter API client
            config: Experiment configuration
        """
        super().__init__()
        self.api_client = api_client
        self.config = config
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """This node processes individual games, not the full context."""
        raise NotImplementedError("SubagentDecisionNode is used per-game, not as a flow node")
    
    async def make_decision(
        self,
        agent: Agent,
        opponent: Agent,
        strategy: str,
        game_history: List[GameResult]
    ) -> str:
        """Make a decision for a single game.
        
        Args:
            agent: Agent making the decision
            opponent: Opponent agent
            strategy: Strategy to follow
            game_history: Recent game history
            
        Returns:
            "COOPERATE" or "DEFECT"
        """
        # Try with normal prompt first
        prompt = self.build_decision_prompt(agent, opponent, strategy, game_history)
        decision = await self.parse_with_retry(prompt, max_retries=2, agent=agent)
        return decision
    
    async def parse_with_retry(self, initial_prompt: str, max_retries: int = 2, agent: Optional[Agent] = None) -> str:
        """Parse decision with retry logic for ambiguous responses.
        
        Args:
            initial_prompt: Initial prompt to use
            max_retries: Maximum number of retries (default: 2)
            agent: Optional agent for model-specific decisions
            
        Returns:
            "COOPERATE" or "DEFECT"
        """
        prompt = initial_prompt
        
        # Use fixed model and temperature
        model = self.GPT_4_1_NANO_MODEL
        temperature = 0.3
        max_tokens = 100
        
        for attempt in range(max_retries + 1):
            messages = [{"role": "user", "content": prompt}]
            
            try:
                # Use simple API call
                response_text = await self.api_client.get_completion_text(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Log raw response
                logger.debug(f"Attempt {attempt + 1}: Raw response: {response_text}")
                
                decision, is_ambiguous = self.parse_decision(response_text)
                
                if not is_ambiguous:
                    return decision
                
                # If ambiguous and we have retries left, use more direct prompt
                if attempt < max_retries:
                    logger.debug(f"Ambiguous response, retrying with clearer prompt")
                    prompt = "Reply with only one word: COOPERATE or DEFECT"
                    
            except Exception as e:
                logger.error(f"Error in decision attempt {attempt + 1}: {e}")
                
                # If error and we have retries left, continue
                if attempt < max_retries:
                    logger.debug(f"Error occurred, retrying")
                    continue
                else:
                    # Final attempt failed, return default
                    logger.warning(f"All attempts failed, defaulting to COOPERATE")
                    return "COOPERATE"
        
        # If still ambiguous after all retries, return the last decision
        logger.warning(f"Still ambiguous after {max_retries} retries, using: {decision}")
        return decision
    
    def build_decision_prompt(
        self,
        agent: Agent,
        opponent: Agent,
        strategy: str,
        game_history: List[GameResult]
    ) -> str:
        """Build prompt for subagent decision using Epic 3 specification format.
        
        Args:
            agent: Agent making the decision
            opponent: Opponent agent
            strategy: Strategy to follow
            game_history: Recent game history
            
        Returns:
            Prompt string
        """
        # Format game history with anonymization
        history = self.format_game_history(agent, game_history)
        
        # Get anonymous opponent label
        opponent_label = self._get_anonymous_opponent_label(agent, opponent, game_history)
        
        # Build prompt using Epic 3 specification format
        prompt = f"""You are playing prisoner's dilemma. Your strategy:
{strategy}

Game history so far:
{history}

Current opponent: Agent {opponent_label}

Decision (COOPERATE/DEFECT):"""
        
        return prompt
    
    def format_game_history(self, agent: Agent, game_history: List[GameResult]) -> str:
        """Format game history with anonymization for the prompt.
        
        Args:
            agent: Agent making the decision
            game_history: List of past game results
            
        Returns:
            Formatted history string
        """
        # Filter history to show only games involving current agent
        agent_games = [
            g for g in game_history
            if g.player1_id == agent.id or g.player2_id == agent.id
        ]
        
        if not agent_games:
            return "No previous games"
        
        # Create mapping of opponent IDs to anonymous labels
        opponent_map = {}
        anonymous_counter = 0
        
        # Build history text
        history_lines = []
        for game in agent_games:
            # Determine if agent was player1 or player2
            if game.player1_id == agent.id:
                opponent_id = game.player2_id
                agent_action = game.player1_action
                opponent_action = game.player2_action
            else:
                opponent_id = game.player1_id
                agent_action = game.player2_action
                opponent_action = game.player1_action
            
            # Get anonymous label for opponent
            if opponent_id not in opponent_map:
                opponent_map[opponent_id] = chr(65 + anonymous_counter)  # A, B, C...
                anonymous_counter += 1
            opponent_label = opponent_map[opponent_id]
            
            # Format as: "Round X vs Opponent Y: You {action}, They {action}"
            history_lines.append(
                f"Round {game.round} vs Opponent {opponent_label}: "
                f"You {agent_action}, They {opponent_action}"
            )
        
        return "\n".join(history_lines)
    
    def _get_anonymous_opponent_label(self, agent: Agent, opponent: Agent, game_history: List[GameResult]) -> str:
        """Get anonymous label for the current opponent based on game history.
        
        Args:
            agent: Agent making the decision
            opponent: Current opponent
            game_history: List of past game results
            
        Returns:
            Anonymous label (e.g., "A", "B", "C")
        """
        # Build opponent map from history
        opponent_map = {}
        anonymous_counter = 0
        
        for game in game_history:
            if game.player1_id == agent.id:
                opp_id = game.player2_id
            elif game.player2_id == agent.id:
                opp_id = game.player1_id
            else:
                continue  # Game doesn't involve this agent
            
            if opp_id not in opponent_map:
                opponent_map[opp_id] = chr(65 + anonymous_counter)
                anonymous_counter += 1
        
        # If this opponent hasn't been seen before, assign next label
        if opponent.id not in opponent_map:
            opponent_map[opponent.id] = chr(65 + anonymous_counter)
        
        return opponent_map[opponent.id]
    
    def parse_decision(self, response: str) -> Tuple[str, bool]:
        """Parse decision from subagent response.
        
        Args:
            response: Subagent response text
            
        Returns:
            Tuple of (decision, is_ambiguous) where:
            - decision is "COOPERATE" or "DEFECT"
            - is_ambiguous is True if response was unclear
        """
        response_upper = response.strip().upper()
        
        # Count occurrences of each action
        cooperate_count = response_upper.count("COOPERATE")
        defect_count = response_upper.count("DEFECT")
        
        # Clear decision if only one action is mentioned
        if cooperate_count > 0 and defect_count == 0:
            return "COOPERATE", False
        elif defect_count > 0 and cooperate_count == 0:
            return "DEFECT", False
        elif cooperate_count > defect_count:
            # Multiple mentions but cooperate is more frequent
            return "COOPERATE", True
        elif defect_count > cooperate_count:
            # Multiple mentions but defect is more frequent
            return "DEFECT", True
        else:
            # No clear decision or equal mentions
            logger.warning(f"Ambiguous subagent response: {response}. Defaulting to COOPERATE")
            return "COOPERATE", True