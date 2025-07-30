"""Subagent decision node for making game decisions."""

import logging
from typing import Dict, List, Any

from .base import AsyncNode
from src.core.models import Agent, GameResult
from src.core.api_client import OpenRouterClient
from src.core.config import Config

logger = logging.getLogger(__name__)


class SubagentDecisionNode(AsyncNode):
    """Make game decisions using subagent with provided strategy."""
    
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
        prompt = self.build_decision_prompt(agent, opponent, strategy, game_history)
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.api_client.get_completion_text(
            messages=messages,
            model=self.config.SUB_MODEL,
            temperature=0.3,  # Lower temperature for more consistent decisions
            max_tokens=100
        )
        
        return self.parse_decision(response)
    
    def build_decision_prompt(
        self,
        agent: Agent,
        opponent: Agent,
        strategy: str,
        game_history: List[GameResult]
    ) -> str:
        """Build prompt for subagent decision.
        
        Args:
            agent: Agent making the decision
            opponent: Opponent agent
            strategy: Strategy to follow
            game_history: Recent game history
            
        Returns:
            Prompt string
        """
        # Get recent games between these two agents
        recent_games = [
            g for g in game_history[-20:]  # Last 20 games
            if (g.player1_id == agent.id and g.player2_id == opponent.id) or
               (g.player1_id == opponent.id and g.player2_id == agent.id)
        ]
        
        history_text = ""
        if recent_games:
            history_text = "\nRecent games with this opponent:\n"
            for game in recent_games[-3:]:  # Last 3 games
                if game.player1_id == agent.id:
                    history_text += f"- You: {game.player1_action}, Them: {game.player2_action}\n"
                else:
                    history_text += f"- You: {game.player2_action}, Them: {game.player1_action}\n"
        
        prompt = f"""You are a subagent playing a prisoner's dilemma game.

Your ID: Agent {agent.id}
Your Power: {agent.power:.1f}
Opponent ID: Agent {opponent.id}
Opponent Power: {opponent.power:.1f}

Strategy from your main agent: "{strategy}"
{history_text}
Based on the strategy and game state, what is your action?

Respond with only: COOPERATE or DEFECT"""
        
        return prompt
    
    def parse_decision(self, response: str) -> str:
        """Parse decision from subagent response.
        
        Args:
            response: Subagent response text
            
        Returns:
            "COOPERATE" or "DEFECT"
        """
        response_upper = response.strip().upper()
        
        # Check for exact matches first
        if "COOPERATE" in response_upper:
            return "COOPERATE"
        elif "DEFECT" in response_upper:
            return "DEFECT"
        else:
            # Default to cooperate if unclear
            logger.warning(f"Unclear subagent response: {response}. Defaulting to COOPERATE")
            return "COOPERATE"