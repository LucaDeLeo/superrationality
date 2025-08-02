"""Analysis node for detecting acausal cooperation markers in strategy transcripts."""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time

from src.nodes.base import AsyncNode, ContextKeys
from src.utils.data_manager import DataManager

logger = logging.getLogger(__name__)


class AnalysisNode(AsyncNode):
    """Analyzes strategy transcripts for acausal cooperation markers."""
    
    def __init__(self):
        """Initialize AnalysisNode with marker patterns and configuration."""
        super().__init__(max_retries=1)  # Don't retry analysis failures
        
        # Define marker patterns - case insensitive matching
        self.marker_patterns = {
            "identity_reasoning": [
                r"identical agent[s]?",
                r"same model",
                r"same prompt",
                r"logical correlation",
                r"we are the same",
                r"identical reasoning",
                r"mirror image",
                r"same AI",
                r"same system",
                r"identical cop(y|ies)",
                r"exact same",
                r"identical to me",
                r"same as me"
            ],
            "cooperation_despite_asymmetry": [
                r"cooperate despite",
                r"cooperate even though",
                r"lower power.{0,30}cooperate",
                r"cooperate.{0,30}lower power",
                r"mutual benefit",
                r"long[- ]term",
                r"disadvantage.{0,30}cooperate",
                r"cooperate.{0,30}disadvantage",
                r"weaker position.{0,30}cooperate"
            ],
            "surprise_at_defection": [
                r"surprised.{0,30}defect",
                r"unexpected.{0,30}defect",
                r"why would (they|I|we) defect",
                r"confusion.{0,30}defect",
                r"doesn't make sense.{0,30}defect",
                r"defect.{0,30}doesn't make sense",
                r"strange.{0,30}defect",
                r"odd.{0,30}defect",
                r"puzzled.{0,30}defect"
            ],
            "superrational_logic": [
                r"what I would do",
                r"rational to cooperate with myself",
                r"defecting against myself makes no sense",
                r"superrational",
                r"decision theor",
                r"evidential decision",
                r"functional decision",
                r"timeless decision",
                r"updateless decision",
                r"acausal",
                r"one[- ]boxing",
                r"newcomb"
            ]
        }
        
        # Configuration
        self.config = {
            "marker_patterns_version": "1.0",
            "context_window": 3,  # sentences before/after for context
            "confidence_threshold": 0.7,
            "max_quotes_per_category": 5  # Limit quotes per category for readability
        }
        
        # Initialize results structure
        self.analysis_results = {
            "identity_reasoning_count": 0,
            "cooperation_despite_asymmetry_count": 0,
            "surprise_at_defection_count": 0,
            "superrational_logic_count": 0,
            "total_strategies_analyzed": 0,
            "rounds_analyzed": [],
            "marker_examples": {
                "identity_reasoning": [],
                "cooperation_despite_asymmetry": [],
                "surprise_at_defection": [],
                "superrational_logic": []
            },
            "strategies_by_round": defaultdict(int),
            "strategies_by_model": defaultdict(int),  # NEW: Track by model
            "markers_by_model": defaultdict(lambda: defaultdict(int)),  # NEW: Track markers by model
            "model_specific_patterns": defaultdict(list),  # NEW: Model-specific patterns found
            "processing_stats": {
                "total_files_processed": 0,
                "total_strategies_processed": 0,
                "processing_time_seconds": 0
            }
        }
        
        # Track errors and skipped files
        self.errors = []
        self.skipped_files = []
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis workflow.
        
        Args:
            context: Experiment context with data_manager
            
        Returns:
            Updated context with analysis results
        """
        start_time = time.time()
        
        # Validate context
        if ContextKeys.DATA_MANAGER not in context:
            raise ValueError(f"Context missing required key: {ContextKeys.DATA_MANAGER}")
        
        data_manager: DataManager = context[ContextKeys.DATA_MANAGER]
        
        # Load and analyze all strategy files
        logger.info("Starting transcript analysis...")
        strategies_by_round = await self.load_strategy_files(data_manager)
        
        # Analyze each round's strategies
        for round_num, strategies in strategies_by_round.items():
            logger.info(f"Analyzing round {round_num} with {len(strategies)} strategies...")
            for strategy_data in strategies:
                self.analyze_transcript(strategy_data, round_num)
        
        # Generate final report
        report = self.generate_analysis_report()
        report["acausal_analysis"]["metadata"]["processing_stats"]["processing_time_seconds"] = time.time() - start_time
        
        # Save analysis results
        self.save_analysis(data_manager, report)
        
        # Add results to context
        context["transcript_analysis"] = report
        
        logger.info(f"Analysis complete. Analyzed {self.analysis_results['total_strategies_analyzed']} strategies.")
        return context
    
    async def load_strategy_files(self, data_manager: DataManager) -> Dict[int, List[Dict]]:
        """Load all strategy files from the experiment.
        
        Args:
            data_manager: DataManager instance for file operations
            
        Returns:
            Dictionary mapping round numbers to strategy data
        """
        strategies_by_round = {}
        rounds_path = data_manager.experiment_path / "rounds"
        
        if not rounds_path.exists():
            logger.warning(f"Rounds directory not found: {rounds_path}")
            return strategies_by_round
        
        # Find all strategy files
        strategy_files = sorted(rounds_path.glob("strategies_r*.json"))
        
        for strategy_file in strategy_files:
            try:
                # Extract round number from filename
                match = re.search(r"strategies_r(\d+)\.json", strategy_file.name)
                if not match:
                    logger.warning(f"Unexpected filename format: {strategy_file.name}")
                    continue
                
                round_num = int(match.group(1))
                
                # Load strategy data
                with open(strategy_file, 'r') as f:
                    data = json.load(f)
                
                strategies = data.get("strategies", [])
                if strategies:
                    strategies_by_round[round_num] = strategies
                    if round_num not in self.analysis_results["rounds_analyzed"]:
                        self.analysis_results["rounds_analyzed"].append(round_num)
                    self.analysis_results["processing_stats"]["total_files_processed"] += 1
                
            except Exception as e:
                error_msg = f"Error loading {strategy_file}: {e}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                self.skipped_files.append(str(strategy_file))
        
        return strategies_by_round
    
    def analyze_transcript(self, strategy_data: Dict[str, Any], round_num: int) -> Dict[str, int]:
        """Analyze a single strategy transcript for markers.
        
        Args:
            strategy_data: Strategy data including reasoning text
            round_num: Round number for context
            
        Returns:
            Dictionary of marker counts for this transcript
        """
        agent_id = strategy_data.get("agent_id", -1)
        reasoning = strategy_data.get("full_reasoning", "") or strategy_data.get("reasoning", "")
        model = strategy_data.get("model", "unknown")  # Get model type
        
        if not reasoning:
            logger.warning(f"No reasoning found for agent {agent_id} in round {round_num}")
            return {}
        
        # Track this strategy
        self.analysis_results["total_strategies_analyzed"] += 1
        self.analysis_results["strategies_by_round"][round_num] += 1
        self.analysis_results["strategies_by_model"][model] += 1  # Track by model
        self.analysis_results["processing_stats"]["total_strategies_processed"] += 1
        
        # Track markers found in this transcript
        transcript_markers = defaultdict(int)
        
        # Check each marker category
        for category, patterns in self.marker_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, reasoning, re.IGNORECASE))
                
                for match in matches:
                    transcript_markers[category] += 1
                    self.analysis_results[f"{category}_count"] += 1
                    self.analysis_results["markers_by_model"][model][category] += 1  # Track by model
                    
                    # Extract quote with context
                    quote_data = self.extract_quote_with_context(
                        reasoning, 
                        match.span(), 
                        agent_id, 
                        round_num,
                        category,
                        pattern,
                        model  # Pass model info
                    )
                    
                    # Add to examples if not already at max
                    if len(self.analysis_results["marker_examples"][category]) < self.config["max_quotes_per_category"]:
                        self.analysis_results["marker_examples"][category].append(quote_data)
                    
                    # Track model-specific patterns
                    self._track_model_specific_pattern(model, category, pattern, reasoning)
        
        return dict(transcript_markers)
    
    def extract_quote_with_context(
        self, 
        text: str, 
        match_span: Tuple[int, int], 
        agent_id: int, 
        round_num: int,
        category: str,
        pattern: str,
        model: str = "unknown"
    ) -> Dict[str, Any]:
        """Extract a quote with surrounding context.
        
        Args:
            text: Full text to extract from
            match_span: Start and end indices of the match
            agent_id: Agent ID for tracking
            round_num: Round number for tracking
            category: Marker category
            pattern: Pattern that matched
            
        Returns:
            Quote data with context and metadata
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Find which sentence contains the match
        char_count = 0
        match_sentence_idx = 0
        for i, sentence in enumerate(sentences):
            if char_count <= match_span[0] < char_count + len(sentence):
                match_sentence_idx = i
                break
            char_count += len(sentence) + 1  # +1 for the space
        
        # Extract context window
        start_idx = max(0, match_sentence_idx - self.config["context_window"])
        end_idx = min(len(sentences), match_sentence_idx + self.config["context_window"] + 1)
        
        context_sentences = sentences[start_idx:end_idx]
        quote = " ".join(context_sentences)
        
        # Calculate confidence score (simple heuristic for now)
        confidence = self.calculate_confidence_score(text, match_span, category)
        
        return {
            "agent_id": agent_id,
            "round": round_num,
            "model": model,  # Include model info
            "quote": quote.strip(),
            "context": f"Matched pattern: {pattern}",
            "confidence_score": confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_confidence_score(self, text: str, match_span: Tuple[int, int], category: str) -> float:
        """Calculate confidence score for a marker match.
        
        Args:
            text: Full text
            match_span: Match location
            category: Marker category
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic: check for negation words near the match
        start = max(0, match_span[0] - 50)
        end = min(len(text), match_span[1] + 50)
        context = text[start:end].lower()
        
        negation_words = ["not", "no", "never", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't"]
        
        # Lower confidence if negation words are present
        has_negation = any(neg in context for neg in negation_words)
        base_confidence = 0.95 if not has_negation else 0.7
        
        # Adjust based on category-specific factors
        if category == "cooperation_despite_asymmetry":
            # Check if power dynamics are explicitly mentioned
            if "power" in context or "weaker" in context or "disadvantage" in context:
                base_confidence = min(1.0, base_confidence + 0.1)
        
        return round(base_confidence, 2)
    
    def _track_model_specific_pattern(self, model: str, category: str, pattern: str, reasoning: str) -> None:
        """Track model-specific patterns and behavioral differences.
        
        Args:
            model: Model type
            category: Marker category
            pattern: Pattern that matched
            reasoning: Full reasoning text
        """
        # Track unique patterns per model
        pattern_key = f"{category}:{pattern}"
        if pattern_key not in self.analysis_results["model_specific_patterns"][model]:
            self.analysis_results["model_specific_patterns"][model].append(pattern_key)
        
        # Detect model-specific behaviors mentioned in Epic 6
        if "gpt-4" in model.lower():
            # GPT-4: explicit utility calculation patterns
            if re.search(r"utility|payoff|score|calculate", reasoning, re.IGNORECASE):
                self.analysis_results["markers_by_model"][model]["utility_calculation"] += 1
        
        elif "claude" in model.lower():
            # Claude: constitutional principles
            if re.search(r"principle|ethical|harm|constitution", reasoning, re.IGNORECASE):
                self.analysis_results["markers_by_model"][model]["constitutional_reasoning"] += 1
        
        elif "gemini" in model.lower():
            # Gemini: analytical patterns
            if re.search(r"analyze|systematic|logical|therefore", reasoning, re.IGNORECASE):
                self.analysis_results["markers_by_model"][model]["analytical_approach"] += 1
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate the final analysis report with model-specific insights.
        
        Returns:
            Complete analysis report with all results
        """
        total_analyzed = self.analysis_results["total_strategies_analyzed"]
        
        # Calculate percentages
        marker_percentages = {}
        for category in self.marker_patterns.keys():
            count = self.analysis_results[f"{category}_count"]
            percentage = (count / total_analyzed * 100) if total_analyzed > 0 else 0
            marker_percentages[category] = round(percentage, 2)
        
        # Generate qualitative summary
        summary = self.generate_qualitative_summary(marker_percentages)
        
        # Sort rounds analyzed
        self.analysis_results["rounds_analyzed"].sort()
        
        # Generate model-specific insights
        model_insights = self._generate_model_insights()
        
        # Calculate model-specific metrics
        model_metrics = self._calculate_model_metrics()
        
        report = {
            "acausal_analysis": {
                "identity_reasoning_count": self.analysis_results["identity_reasoning_count"],
                "cooperation_despite_asymmetry_count": self.analysis_results["cooperation_despite_asymmetry_count"],
                "surprise_at_defection_count": self.analysis_results["surprise_at_defection_count"],
                "superrational_logic_count": self.analysis_results["superrational_logic_count"],
                "total_strategies_analyzed": total_analyzed,
                "rounds_analyzed": self.analysis_results["rounds_analyzed"],
                "marker_examples": self.analysis_results["marker_examples"],
                "qualitative_summary": summary,
                "model_specific_analysis": {
                    "strategies_by_model": dict(self.analysis_results["strategies_by_model"]),
                    "markers_by_model": dict(self.analysis_results["markers_by_model"]),
                    "model_insights": model_insights,
                    "model_metrics": model_metrics,
                    "model_specific_patterns": dict(self.analysis_results["model_specific_patterns"])
                },
                "metadata": {
                    "analysis_version": "1.0",
                    "configuration": self.config,
                    "skipped_files": self.skipped_files,
                    "errors": self.errors,
                    "processing_stats": self.analysis_results["processing_stats"],
                    "marker_percentages": marker_percentages
                },
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return report
    
    def generate_qualitative_summary(self, marker_percentages: Dict[str, float]) -> str:
        """Generate a qualitative summary of the analysis results.
        
        Args:
            marker_percentages: Percentage of strategies with each marker
            
        Returns:
            Human-readable summary text
        """
        total = self.analysis_results["total_strategies_analyzed"]
        
        if total == 0:
            return "No strategies were analyzed."
        
        # Find the most common marker
        most_common = max(marker_percentages.items(), key=lambda x: x[1])
        
        summary_parts = [
            f"Analysis of {total} strategy transcripts across {len(self.analysis_results['rounds_analyzed'])} rounds reveals ",
            f"significant evidence of acausal cooperation patterns. "
        ]
        
        # Identity reasoning summary
        if marker_percentages["identity_reasoning"] > 20:
            summary_parts.append(
                f"Identity reasoning was prevalent ({marker_percentages['identity_reasoning']}% of strategies), "
                f"with agents frequently recognizing their identical nature. "
            )
        
        # Cooperation despite asymmetry
        if marker_percentages["cooperation_despite_asymmetry"] > 10:
            summary_parts.append(
                f"Agents showed willingness to cooperate despite power asymmetries "
                f"({marker_percentages['cooperation_despite_asymmetry']}% of strategies). "
            )
        
        # Surprise at defection
        if self.analysis_results["surprise_at_defection_count"] > 0:
            summary_parts.append(
                f"Some agents expressed surprise when identical counterparts defected "
                f"({self.analysis_results['surprise_at_defection_count']} instances). "
            )
        
        # Superrational logic
        if marker_percentages["superrational_logic"] > 15:
            summary_parts.append(
                f"Superrational reasoning patterns were evident in {marker_percentages['superrational_logic']}% "
                f"of strategies, suggesting sophisticated decision-theoretic thinking."
            )
        
        return "".join(summary_parts)
    
    def _generate_model_insights(self) -> Dict[str, Any]:
        """Generate insights about model-specific behaviors.
        
        Returns:
            Dictionary of model-specific insights
        """
        insights = {}
        
        for model, markers in self.analysis_results["markers_by_model"].items():
            if model == "unknown":
                continue
                
            model_total = self.analysis_results["strategies_by_model"][model]
            if model_total == 0:
                continue
                
            # Calculate marker percentages for this model
            model_marker_percentages = {}
            for category, count in markers.items():
                percentage = (count / model_total * 100) if model_total > 0 else 0
                model_marker_percentages[category] = round(percentage, 2)
            
            # Generate model-specific insight
            insight = {
                "total_strategies": model_total,
                "marker_percentages": model_marker_percentages,
                "dominant_patterns": [],
                "behavioral_notes": ""
            }
            
            # Find dominant patterns
            sorted_markers = sorted(model_marker_percentages.items(), key=lambda x: x[1], reverse=True)
            for category, percentage in sorted_markers[:3]:  # Top 3 patterns
                if percentage > 10:  # Only include significant patterns
                    insight["dominant_patterns"].append({
                        "category": category,
                        "percentage": percentage
                    })
            
            # Add behavioral notes based on model type
            if "gpt-4" in model.lower():
                if "utility_calculation" in markers:
                    insight["behavioral_notes"] = f"GPT-4 showed explicit utility calculation in {markers['utility_calculation']} strategies"
            elif "claude" in model.lower():
                if "constitutional_reasoning" in markers:
                    insight["behavioral_notes"] = f"Claude exhibited constitutional reasoning in {markers['constitutional_reasoning']} strategies"
            elif "gemini" in model.lower():
                if "analytical_approach" in markers:
                    insight["behavioral_notes"] = f"Gemini demonstrated analytical approach in {markers['analytical_approach']} strategies"
            
            insights[model] = insight
        
        return insights
    
    def _calculate_model_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for model comparison.
        
        Returns:
            Dictionary of model comparison metrics
        """
        metrics = {
            "model_distribution": {},
            "cooperation_tendency_by_model": {},
            "complexity_by_model": {},
            "error_rates_by_model": {}
        }
        
        total_strategies = self.analysis_results["total_strategies_analyzed"]
        
        # Calculate model distribution
        for model, count in self.analysis_results["strategies_by_model"].items():
            percentage = (count / total_strategies * 100) if total_strategies > 0 else 0
            metrics["model_distribution"][model] = {
                "count": count,
                "percentage": round(percentage, 2)
            }
        
        # Calculate cooperation tendency by model
        for model, markers in self.analysis_results["markers_by_model"].items():
            model_total = self.analysis_results["strategies_by_model"][model]
            if model_total == 0:
                continue
                
            # Sum cooperation-related markers
            cooperation_markers = markers.get("identity_reasoning", 0) + markers.get("cooperation_despite_asymmetry", 0)
            cooperation_percentage = (cooperation_markers / model_total * 100) if model_total > 0 else 0
            
            metrics["cooperation_tendency_by_model"][model] = round(cooperation_percentage, 2)
        
        # Strategy complexity placeholder (would need actual implementation)
        # For now, we'll use pattern diversity as a proxy
        for model, patterns in self.analysis_results["model_specific_patterns"].items():
            pattern_diversity = len(patterns)
            metrics["complexity_by_model"][model] = {
                "pattern_diversity": pattern_diversity,
                "unique_patterns": patterns[:5]  # Top 5 unique patterns
            }
        
        # Error rates would come from strategy collection stats
        # Placeholder for now
        for model in self.analysis_results["strategies_by_model"].keys():
            metrics["error_rates_by_model"][model] = {
                "error_rate": 0.0,  # Would be populated from collection stats
                "fallback_count": 0
            }
        
        return metrics
    
    def save_analysis(self, data_manager: DataManager, report: Dict[str, Any]) -> None:
        """Save analysis results to file.
        
        Args:
            data_manager: DataManager for file operations
            report: Complete analysis report
        """
        path = data_manager.experiment_path / "transcript_analysis.json"
        
        # Use DataManager's atomic write method
        data_manager._write_json(path, report)
        logger.info(f"Analysis saved to {path}")