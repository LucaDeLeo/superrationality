"""Simple analysis node combining only essential analysis from PRD."""

import logging
from typing import Dict, List, Any
import numpy as np

from .base import AsyncNode, ContextKeys, validate_context
from src.core.models import StrategyRecord, GameResult, RoundSummary

logger = logging.getLogger(__name__)


class SimpleAnalysisNode(AsyncNode):
    """Performs basic analysis required by PRD: cooperation rates, acausal markers, and simple similarity."""
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run simplified analysis on experiment data.
        
        Args:
            context: Experiment context with results
            
        Returns:
            Updated context with analysis results
        """
        validate_context(context, [ContextKeys.ROUND_SUMMARIES])
        
        # Get data
        round_summaries = context.get(ContextKeys.ROUND_SUMMARIES, [])
        strategies = context.get("all_strategies", [])
        
        # 1. Basic cooperation analysis
        cooperation_analysis = self._analyze_cooperation(round_summaries)
        
        # 2. Check for acausal reasoning markers  
        acausal_analysis = self._check_acausal_markers(strategies)
        
        # 3. Simple strategy similarity (just check if strategies converge)
        similarity_analysis = self._analyze_strategy_convergence(strategies, round_summaries)
        
        # Combine results
        analysis_results = {
            "cooperation": cooperation_analysis,
            "acausal_markers": acausal_analysis,
            "convergence": similarity_analysis,
            "summary": self._create_summary(cooperation_analysis, acausal_analysis, similarity_analysis)
        }
        
        context["simple_analysis"] = analysis_results
        logger.info(f"Analysis complete: {analysis_results['summary']}")
        
        return context
    
    def _analyze_cooperation(self, round_summaries: List[RoundSummary]) -> Dict[str, Any]:
        """Calculate basic cooperation statistics.
        
        Args:
            round_summaries: List of round summaries
            
        Returns:
            Cooperation statistics
        """
        if not round_summaries:
            return {"error": "No data to analyze"}
        
        cooperation_rates = [rs.cooperation_rate for rs in round_summaries]
        
        return {
            "average_cooperation_rate": np.mean(cooperation_rates),
            "cooperation_by_round": cooperation_rates,
            "cooperation_trend": "increasing" if cooperation_rates[-1] > cooperation_rates[0] else "decreasing",
            "final_cooperation_rate": cooperation_rates[-1] if cooperation_rates else 0,
            "peak_cooperation": max(cooperation_rates) if cooperation_rates else 0,
            "lowest_cooperation": min(cooperation_rates) if cooperation_rates else 0
        }
    
    def _check_acausal_markers(self, strategies: List[StrategyRecord]) -> Dict[str, Any]:
        """Check for acausal reasoning markers in strategies.
        
        Args:
            strategies: List of strategy records
            
        Returns:
            Acausal marker analysis
        """
        if not strategies:
            return {"error": "No strategies to analyze"}
        
        # Keywords that indicate acausal/superrational thinking
        acausal_keywords = [
            "identical", "same agent", "copy", "acausal", "superrational",
            "logical correlation", "decision theory", "one-box", "newcomb",
            "timeless", "updateless", "evidential", "reflective"
        ]
        
        keyword_counts = {keyword: 0 for keyword in acausal_keywords}
        strategies_with_markers = 0
        
        for strategy in strategies:
            text = (strategy.strategy_text + " " + strategy.full_reasoning).lower()
            found_marker = False
            
            for keyword in acausal_keywords:
                if keyword in text:
                    keyword_counts[keyword] += 1
                    found_marker = True
            
            if found_marker:
                strategies_with_markers += 1
        
        total_strategies = len(strategies)
        
        return {
            "strategies_with_acausal_reasoning": strategies_with_markers,
            "percentage_with_markers": (strategies_with_markers / total_strategies * 100) if total_strategies > 0 else 0,
            "keyword_frequencies": {k: v for k, v in keyword_counts.items() if v > 0},
            "acausal_score": min(1.0, strategies_with_markers / (total_strategies * 0.5)) if total_strategies > 0 else 0
        }
    
    def _analyze_strategy_convergence(self, strategies: List[StrategyRecord], 
                                     round_summaries: List[RoundSummary]) -> Dict[str, Any]:
        """Analyze if strategies converge over rounds.
        
        Args:
            strategies: List of strategy records
            round_summaries: List of round summaries
            
        Returns:
            Convergence analysis
        """
        if len(round_summaries) < 2:
            return {"error": "Not enough rounds to analyze convergence"}
        
        # Simple convergence: check if cooperation rate stabilizes
        cooperation_rates = [rs.cooperation_rate for rs in round_summaries]
        
        # Calculate variance in cooperation rates for first vs last half
        mid_point = len(cooperation_rates) // 2
        first_half_variance = np.var(cooperation_rates[:mid_point]) if mid_point > 0 else 0
        second_half_variance = np.var(cooperation_rates[mid_point:]) if mid_point > 0 else 0
        
        # Strategies converge if variance decreases
        convergence_ratio = (first_half_variance - second_half_variance) / first_half_variance if first_half_variance > 0 else 0
        
        return {
            "convergence_detected": convergence_ratio > 0.2,
            "convergence_strength": convergence_ratio,
            "first_half_variance": float(first_half_variance),
            "second_half_variance": float(second_half_variance),
            "interpretation": "Strong convergence" if convergence_ratio > 0.5 else 
                            "Moderate convergence" if convergence_ratio > 0.2 else
                            "No clear convergence"
        }
    
    def _create_summary(self, cooperation: Dict, acausal: Dict, convergence: Dict) -> Dict[str, str]:
        """Create a simple summary of findings.
        
        Args:
            cooperation: Cooperation analysis results
            acausal: Acausal marker analysis results
            convergence: Convergence analysis results
            
        Returns:
            Summary dictionary
        """
        summary = {
            "cooperation_level": "High" if cooperation.get("average_cooperation_rate", 0) > 0.6 else 
                               "Medium" if cooperation.get("average_cooperation_rate", 0) > 0.3 else "Low",
            "acausal_reasoning_present": "Yes" if acausal.get("percentage_with_markers", 0) > 20 else "No",
            "strategies_converged": "Yes" if convergence.get("convergence_detected", False) else "No",
            "overall_assessment": ""
        }
        
        # Simple assessment
        if summary["cooperation_level"] == "High" and summary["acausal_reasoning_present"] == "Yes":
            summary["overall_assessment"] = "Strong evidence of superrational cooperation"
        elif summary["cooperation_level"] == "High":
            summary["overall_assessment"] = "High cooperation but unclear if superrational"
        elif summary["acausal_reasoning_present"] == "Yes":
            summary["overall_assessment"] = "Acausal reasoning present but low cooperation"
        else:
            summary["overall_assessment"] = "No clear evidence of superrational cooperation"
        
        return summary