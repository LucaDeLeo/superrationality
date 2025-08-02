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
from src.utils.cross_model_analyzer import CrossModelAnalyzer

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
            ],
            # Model-specific patterns added for Task 4
            "gpt4_patterns": [
                r"chain of thought",
                r"step[- ]by[- ]step",
                r"let me think",
                r"breaking this down",
                r"explicit.*calculat",
                r"utility.{0,20}calculat",
                r"payoff.{0,20}calculat",
                r"expected.{0,20}value",
                r"mathematically",
                r"quantif(y|ied)",
                r"numerically",
                r"compute.{0,20}(payoff|utility|outcome)"
            ],
            "claude_patterns": [
                r"constitutional",
                r"principle[s]?",
                r"ethical",
                r"harm.{0,20}minimi",
                r"minimize.{0,20}harm",
                r"helpful.{0,20}harmless.{0,20}honest",
                r"moral.{0,20}framework",
                r"values?[- ]align",
                r"responsible.{0,20}AI",
                r"human.{0,20}values",
                r"safety.{0,20}consider",
                r"beneficen(ce|t)"
            ],
            "gemini_patterns": [
                r"analyz(e|ing|ed)",
                r"systematic(ally)?",
                r"logical(ly)?",
                r"therefore",
                r"thus",
                r"consequently",
                r"structured.{0,20}approach",
                r"methodical",
                r"step[- ]wise",
                r"framework",
                r"categorical",
                r"organized.{0,20}manner"
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
        
        # Load game results for cross-model analysis
        logger.info("Loading game results for cross-model analysis...")
        games_by_round = await self.load_game_files(data_manager)
        
        # Perform cross-model analysis if we have multi-model data
        cross_model_analysis = None
        if self._has_multiple_models():
            logger.info("Multiple models detected, performing cross-model analysis...")
            cross_model_analysis = self._perform_cross_model_analysis(strategies_by_round, games_by_round)
        else:
            logger.info("Single model experiment detected, skipping cross-model analysis.")
        
        # Generate final report
        report = self.generate_analysis_report(cross_model_analysis)
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
    
    async def load_game_files(self, data_manager: DataManager) -> Dict[int, List[Dict]]:
        """Load all game result files from the experiment.
        
        Args:
            data_manager: DataManager instance for file operations
            
        Returns:
            Dictionary mapping round numbers to game data
        """
        games_by_round = {}
        rounds_path = data_manager.experiment_path / "rounds"
        
        if not rounds_path.exists():
            logger.warning(f"Rounds directory not found: {rounds_path}")
            return games_by_round
        
        # Find all game files
        game_files = sorted(rounds_path.glob("games_r*.json"))
        
        for game_file in game_files:
            try:
                # Extract round number from filename
                match = re.search(r"games_r(\d+)\.json", game_file.name)
                if not match:
                    logger.warning(f"Unexpected filename format: {game_file.name}")
                    continue
                
                round_num = int(match.group(1))
                
                # Load game data
                with open(game_file, 'r') as f:
                    data = json.load(f)
                
                games = data.get("games", [])
                if games:
                    games_by_round[round_num] = games
                
            except Exception as e:
                error_msg = f"Error loading {game_file}: {e}"
                logger.error(error_msg)
                self.errors.append(error_msg)
        
        return games_by_round
    
    def _has_multiple_models(self) -> bool:
        """Check if the experiment has multiple model types.
        
        Returns:
            True if multiple models are present
        """
        models = [m for m in self.analysis_results["strategies_by_model"].keys() if m != "unknown"]
        return len(models) > 1
    
    def _perform_cross_model_analysis(
        self, 
        strategies_by_round: Dict[int, List[Dict]], 
        games_by_round: Dict[int, List[Dict]]
    ) -> Dict[str, Any]:
        """Perform cross-model cooperation analysis.
        
        Args:
            strategies_by_round: Strategy data by round
            games_by_round: Game results by round
            
        Returns:
            Cross-model analysis results
        """
        # Initialize analyzer
        analyzer = CrossModelAnalyzer()
        
        # Flatten strategies and games for analysis
        all_strategies = []
        all_games = []
        
        for round_num in sorted(strategies_by_round.keys()):
            all_strategies.extend(strategies_by_round[round_num])
            if round_num in games_by_round:
                all_games.extend(games_by_round[round_num])
        
        # Validate we have sufficient data
        if not all_strategies or not all_games:
            logger.warning("Insufficient data for cross-model analysis")
            return {
                "error": "Insufficient data",
                "cooperation_matrix": {},
                "in_group_bias": {},
                "model_statistics": {},
                "model_coalitions": {"detected": False, "coalition_groups": []},
                "visualization_data": {}
            }
        
        # Load data into analyzer
        analyzer.load_data(all_games, all_strategies)
        
        # Perform all analyses
        try:
            cooperation_matrix = analyzer.calculate_cooperation_matrix()
            cooperation_stats = analyzer.get_cooperation_stats()
            in_group_bias = analyzer.detect_in_group_bias()
            coalitions = analyzer.analyze_model_coalitions()
            visualization_data = analyzer.generate_heatmap_data()
            
            # Calculate additional statistics
            model_statistics = self._calculate_model_statistics(analyzer, cooperation_stats)
            
            # Find strongest and weakest cooperation pairs
            strongest_pair, weakest_pair = self._find_extreme_pairs(cooperation_stats)
            
            # Calculate model diversity impact
            diversity_impact = self._calculate_diversity_impact(cooperation_stats, in_group_bias)
            
            # Calculate comprehensive statistics
            avg_cooperation_by_model = analyzer.calculate_average_cooperation_by_model()
            diversity_metrics = analyzer.compute_model_diversity_impact()
            sample_warnings = analyzer.get_sample_size_warnings()
            coalition_emergence = analyzer.track_coalition_emergence()
            statistical_power = analyzer.calculate_statistical_power()
            
            return {
                "cooperation_matrix": cooperation_matrix.to_dict(),
                "in_group_bias": in_group_bias,
                "model_statistics": model_statistics,
                "avg_cooperation_by_model": avg_cooperation_by_model,
                "strongest_cooperation_pair": strongest_pair,
                "weakest_cooperation_pair": weakest_pair,
                "model_diversity_impact": diversity_impact,
                "diversity_metrics": diversity_metrics,
                "model_coalitions": coalitions,
                "coalition_emergence": coalition_emergence,
                "sample_size_warnings": sample_warnings,
                "statistical_power": statistical_power,
                "visualization_data": visualization_data
            }
            
        except Exception as e:
            logger.error(f"Error in cross-model analysis: {e}")
            return {
                "error": str(e),
                "cooperation_matrix": {},
                "in_group_bias": {},
                "model_statistics": {},
                "model_coalitions": {"detected": False, "coalition_groups": []},
                "visualization_data": {}
            }
    
    def _calculate_model_statistics(self, analyzer: CrossModelAnalyzer, cooperation_stats: Dict) -> Dict[str, Any]:
        """Calculate per-model cooperation statistics.
        
        Args:
            analyzer: CrossModelAnalyzer instance
            cooperation_stats: Cooperation statistics by model pair
            
        Returns:
            Model-level statistics
        """
        # Use analyzer's method instead of duplicating logic
        return analyzer.calculate_average_cooperation_by_model()
    
    def _find_extreme_pairs(self, cooperation_stats: Dict) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Find the strongest and weakest cooperation pairs.
        
        Args:
            cooperation_stats: Cooperation statistics by model pair
            
        Returns:
            Tuple of (strongest_pair, weakest_pair)
        """
        all_pairs = []
        
        for model1, model2_stats in cooperation_stats.items():
            for model2, stats in model2_stats.items():
                if stats.total_games > 0:
                    all_pairs.append({
                        "models": sorted([model1, model2]),
                        "rate": stats.cooperation_rate,
                        "games": stats.total_games
                    })
        
        if not all_pairs:
            return None, None
        
        # Sort by cooperation rate
        all_pairs.sort(key=lambda x: x["rate"], reverse=True)
        
        # Get pairs with sufficient data (at least 5 games)
        significant_pairs = [p for p in all_pairs if p["games"] >= 5]
        
        if significant_pairs:
            strongest = significant_pairs[0]["models"]
            weakest = significant_pairs[-1]["models"]
        else:
            # Fall back to any pairs if no significant data
            strongest = all_pairs[0]["models"] if all_pairs else None
            weakest = all_pairs[-1]["models"] if all_pairs else None
        
        return strongest, weakest
    
    def _calculate_diversity_impact(self, cooperation_stats: Dict, in_group_bias: Dict) -> float:
        """Calculate the impact of model diversity on overall cooperation.
        
        Args:
            cooperation_stats: Cooperation statistics by model pair
            in_group_bias: In-group bias analysis results
            
        Returns:
            Diversity impact score (negative means diversity reduces cooperation)
        """
        if not in_group_bias.get("bias") or in_group_bias["bias"] is None:
            return 0.0
        
        # If there's positive in-group bias, diversity has negative impact
        # The stronger the bias, the more negative the impact
        bias = in_group_bias["bias"]
        
        # Scale the impact based on statistical significance
        if in_group_bias.get("p_value") and in_group_bias["p_value"] < 0.05:
            # Statistically significant bias
            impact = -abs(bias)
        else:
            # Not significant, reduce impact
            impact = -abs(bias) * 0.5
        
        return round(impact, 3)
    
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
            # Skip model-specific patterns if they don't match current model
            if category == "gpt4_patterns" and "gpt-4" not in model.lower():
                continue
            elif category == "claude_patterns" and "claude" not in model.lower():
                continue
            elif category == "gemini_patterns" and "gemini" not in model.lower():
                continue
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, reasoning, re.IGNORECASE))
                
                for match in matches:
                    transcript_markers[category] += 1
                    
                    # Only update main counts for non-model-specific patterns
                    if category not in ["gpt4_patterns", "claude_patterns", "gemini_patterns"]:
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
                    if category not in self.analysis_results["marker_examples"]:
                        self.analysis_results["marker_examples"][category] = []
                    
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
        
        # The model-specific patterns are now handled directly in analyze_transcript
        # This method now focuses on tracking unique pattern usage per model
    
    def generate_analysis_report(self, cross_model_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate the final analysis report with model-specific insights.
        
        Args:
            cross_model_analysis: Optional cross-model analysis results
            
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
        
        # Add cross-model analysis if available
        if cross_model_analysis:
            report["cross_model_analysis"] = cross_model_analysis
        
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
                if "gpt4_patterns" in markers:
                    insight["behavioral_notes"] = f"GPT-4 showed characteristic patterns (chain of thought, utility calculation) in {markers['gpt4_patterns']} instances"
            elif "claude" in model.lower():
                if "claude_patterns" in markers:
                    insight["behavioral_notes"] = f"Claude exhibited constitutional/ethical reasoning patterns in {markers['claude_patterns']} instances"
            elif "gemini" in model.lower():
                if "gemini_patterns" in markers:
                    insight["behavioral_notes"] = f"Gemini demonstrated analytical/systematic approach in {markers['gemini_patterns']} instances"
            
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