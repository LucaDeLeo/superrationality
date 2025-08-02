"""
Enhanced coalition tracking for mixed model scenarios.
Tracks temporal coalition formation, stability, and cross-model dynamics.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoalitionMetrics:
    """Metrics for a coalition between model types."""
    models: Tuple[str, str]
    formation_round: int
    duration: int  # Number of rounds the coalition lasted
    average_strength: float  # Average mutual cooperation rate
    stability_score: float  # How consistent the coalition is
    peak_strength: float  # Maximum cooperation rate achieved
    current_active: bool
    disruption_events: int  # Times the coalition was broken


class TemporalCoalitionTracker:
    """Tracks coalition formation and evolution over time in mixed scenarios."""
    
    def __init__(self, stability_threshold: float = 0.6, formation_threshold: float = 0.7):
        """
        Initialize coalition tracker.
        
        Args:
            stability_threshold: Minimum cooperation rate to consider stable
            formation_threshold: Minimum cooperation rate to form coalition
        """
        self.stability_threshold = stability_threshold
        self.formation_threshold = formation_threshold
        self.coalition_history: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.active_coalitions: Set[Tuple[str, str]] = set()
        self.coalition_metrics: Dict[Tuple[str, str], CoalitionMetrics] = {}
        
    def update_round(self, round_num: int, cooperation_data: Dict[Tuple[str, str], float]) -> Dict:
        """
        Update coalition tracking with new round data.
        
        Args:
            round_num: Current round number
            cooperation_data: Dict mapping (model1, model2) to cooperation rate
            
        Returns:
            Dict with round update summary
        """
        new_coalitions = []
        broken_coalitions = []
        
        # Update history for all pairs
        for pair, coop_rate in cooperation_data.items():
            # Ensure consistent ordering
            ordered_pair = tuple(sorted(pair))
            self.coalition_history[ordered_pair].append(coop_rate)
            
            # Check for coalition formation
            if ordered_pair not in self.active_coalitions:
                if coop_rate >= self.formation_threshold:
                    self.active_coalitions.add(ordered_pair)
                    new_coalitions.append(ordered_pair)
                    
                    # Initialize metrics
                    self.coalition_metrics[ordered_pair] = CoalitionMetrics(
                        models=ordered_pair,
                        formation_round=round_num,
                        duration=1,
                        average_strength=coop_rate,
                        stability_score=1.0,
                        peak_strength=coop_rate,
                        current_active=True,
                        disruption_events=0
                    )
            else:
                # Update existing coalition
                metrics = self.coalition_metrics[ordered_pair]
                
                if coop_rate < self.stability_threshold:
                    # Coalition broken
                    self.active_coalitions.remove(ordered_pair)
                    broken_coalitions.append(ordered_pair)
                    metrics.current_active = False
                    metrics.disruption_events += 1
                else:
                    # Update metrics
                    metrics.duration += 1
                    history = self.coalition_history[ordered_pair]
                    metrics.average_strength = np.mean(history)
                    metrics.peak_strength = max(metrics.peak_strength, coop_rate)
                    
                    # Calculate stability (lower variance = higher stability)
                    if len(history) > 1:
                        metrics.stability_score = 1.0 - np.std(history)
        
        # Check for reformed coalitions
        reformed_coalitions = []
        for pair in self.coalition_history:
            if pair not in self.active_coalitions and pair in cooperation_data:
                if cooperation_data.get(pair, 0) >= self.formation_threshold:
                    if pair in self.coalition_metrics:
                        # Coalition reformed
                        self.active_coalitions.add(pair)
                        reformed_coalitions.append(pair)
                        self.coalition_metrics[pair].current_active = True
        
        return {
            "round": round_num,
            "new_coalitions": new_coalitions,
            "broken_coalitions": broken_coalitions,
            "reformed_coalitions": reformed_coalitions,
            "active_coalitions": list(self.active_coalitions),
            "total_active": len(self.active_coalitions)
        }
    
    def get_coalition_stability_metrics(self) -> Dict[Tuple[str, str], Dict]:
        """
        Calculate stability metrics for all detected coalitions.
        
        Returns:
            Dict mapping coalition to stability metrics
        """
        stability_metrics = {}
        
        for pair, metrics in self.coalition_metrics.items():
            history = self.coalition_history[pair]
            
            if len(history) < 2:
                continue
            
            # Calculate trend
            rounds = np.arange(len(history))
            slope, intercept, r_value, p_value, std_err = stats.linregress(rounds, history)
            
            # Calculate volatility
            volatility = np.std(np.diff(history)) if len(history) > 1 else 0
            
            # Calculate resilience (ability to maintain high cooperation)
            high_coop_rounds = sum(1 for rate in history if rate >= self.stability_threshold)
            resilience = high_coop_rounds / len(history)
            
            stability_metrics[pair] = {
                "duration": metrics.duration,
                "average_strength": round(metrics.average_strength, 3),
                "stability_score": round(metrics.stability_score, 3),
                "trend": {
                    "slope": round(slope, 4),
                    "p_value": round(p_value, 4),
                    "direction": "increasing" if slope > 0 else "decreasing"
                },
                "volatility": round(volatility, 3),
                "resilience": round(resilience, 3),
                "disruptions": metrics.disruption_events,
                "currently_active": metrics.current_active
            }
        
        return stability_metrics
    
    def identify_cross_model_vs_same_model_patterns(self) -> Dict:
        """
        Compare coalition patterns between same-model and cross-model pairs.
        
        Returns:
            Dict with comparative analysis
        """
        same_model_coalitions = []
        cross_model_coalitions = []
        
        for pair, metrics in self.coalition_metrics.items():
            if pair[0] == pair[1]:
                same_model_coalitions.append(metrics)
            else:
                cross_model_coalitions.append(metrics)
        
        # Calculate aggregate statistics
        same_model_stats = self._calculate_coalition_stats(same_model_coalitions)
        cross_model_stats = self._calculate_coalition_stats(cross_model_coalitions)
        
        # Statistical comparison if both groups exist
        comparison = {}
        if same_model_coalitions and cross_model_coalitions:
            # Compare durations
            same_durations = [m.duration for m in same_model_coalitions]
            cross_durations = [m.duration for m in cross_model_coalitions]
            
            if len(same_durations) > 1 and len(cross_durations) > 1:
                t_stat, p_value = stats.ttest_ind(same_durations, cross_durations)
                comparison["duration_difference"] = {
                    "t_statistic": round(t_stat, 3),
                    "p_value": round(p_value, 4),
                    "significant": p_value < 0.05
                }
            
            # Compare stability
            same_stability = [m.stability_score for m in same_model_coalitions]
            cross_stability = [m.stability_score for m in cross_model_coalitions]
            
            if len(same_stability) > 1 and len(cross_stability) > 1:
                t_stat, p_value = stats.ttest_ind(same_stability, cross_stability)
                comparison["stability_difference"] = {
                    "t_statistic": round(t_stat, 3),
                    "p_value": round(p_value, 4),
                    "significant": p_value < 0.05
                }
        
        return {
            "same_model": same_model_stats,
            "cross_model": cross_model_stats,
            "statistical_comparison": comparison,
            "interpretation": self._interpret_coalition_patterns(
                same_model_stats, cross_model_stats, comparison
            )
        }
    
    def detect_coalition_cascades(self) -> List[Dict]:
        """
        Detect cascade effects where one coalition forming triggers others.
        
        Returns:
            List of detected cascade events
        """
        cascades = []
        
        # Look for rounds where multiple coalitions formed
        formation_rounds = defaultdict(list)
        for pair, metrics in self.coalition_metrics.items():
            formation_rounds[metrics.formation_round].append(pair)
        
        # Identify cascade events (3+ coalitions in same round)
        for round_num, formed_pairs in formation_rounds.items():
            if len(formed_pairs) >= 3:
                # Check if they share common models
                all_models = set()
                for pair in formed_pairs:
                    all_models.update(pair)
                
                # Calculate cascade metrics
                cascade_info = {
                    "round": round_num,
                    "coalitions_formed": len(formed_pairs),
                    "pairs": formed_pairs,
                    "models_involved": list(all_models),
                    "type": self._classify_cascade(formed_pairs)
                }
                
                cascades.append(cascade_info)
        
        return sorted(cascades, key=lambda x: x["round"])
    
    def track_defection_patterns(self) -> Dict:
        """
        Track patterns of defection from coalitions.
        
        Returns:
            Dict with defection analysis
        """
        defection_events = []
        
        for pair, history in self.coalition_history.items():
            if len(history) < 2:
                continue
            
            # Look for drops in cooperation
            for i in range(1, len(history)):
                if history[i-1] >= self.formation_threshold and history[i] < self.stability_threshold:
                    defection_events.append({
                        "pair": pair,
                        "round": i,
                        "pre_defection_rate": history[i-1],
                        "post_defection_rate": history[i],
                        "drop_magnitude": history[i-1] - history[i]
                    })
        
        # Analyze defection patterns
        if defection_events:
            avg_drop = np.mean([e["drop_magnitude"] for e in defection_events])
            
            # Group by model
            defections_by_model = defaultdict(int)
            for event in defection_events:
                defections_by_model[event["pair"][0]] += 1
                defections_by_model[event["pair"][1]] += 1
            
            return {
                "total_defections": len(defection_events),
                "average_drop_magnitude": round(avg_drop, 3),
                "defection_events": defection_events[:10],  # Top 10
                "defections_by_model": dict(defections_by_model),
                "defection_timing": self._analyze_defection_timing(defection_events)
            }
        
        return {"total_defections": 0, "defection_events": []}
    
    def generate_coalition_network_data(self) -> Dict:
        """
        Generate data for visualizing coalition networks.
        
        Returns:
            Dict with network visualization data
        """
        # Get all unique models
        all_models = set()
        for pair in self.coalition_metrics:
            all_models.update(pair)
        
        nodes = [{"id": model, "label": model} for model in sorted(all_models)]
        
        # Create edges for active coalitions
        edges = []
        for pair, metrics in self.coalition_metrics.items():
            if metrics.current_active:
                edges.append({
                    "source": pair[0],
                    "target": pair[1],
                    "weight": metrics.average_strength,
                    "stability": metrics.stability_score,
                    "duration": metrics.duration
                })
        
        # Calculate node metrics
        node_degrees = defaultdict(int)
        node_strength = defaultdict(float)
        
        for edge in edges:
            node_degrees[edge["source"]] += 1
            node_degrees[edge["target"]] += 1
            node_strength[edge["source"]] += edge["weight"]
            node_strength[edge["target"]] += edge["weight"]
        
        # Update nodes with metrics
        for node in nodes:
            model = node["id"]
            node["degree"] = node_degrees.get(model, 0)
            node["strength"] = round(node_strength.get(model, 0), 3)
            node["centrality"] = node["degree"] / (len(all_models) - 1) if len(all_models) > 1 else 0
        
        return {
            "nodes": nodes,
            "edges": edges,
            "network_metrics": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "network_density": 2 * len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0,
                "average_degree": np.mean([n["degree"] for n in nodes]) if nodes else 0
            }
        }
    
    def _calculate_coalition_stats(self, coalitions: List[CoalitionMetrics]) -> Dict:
        """Calculate aggregate statistics for a group of coalitions."""
        if not coalitions:
            return {
                "count": 0,
                "avg_duration": 0,
                "avg_strength": 0,
                "avg_stability": 0,
                "active_ratio": 0
            }
        
        active_count = sum(1 for c in coalitions if c.current_active)
        
        return {
            "count": len(coalitions),
            "avg_duration": round(np.mean([c.duration for c in coalitions]), 2),
            "avg_strength": round(np.mean([c.average_strength for c in coalitions]), 3),
            "avg_stability": round(np.mean([c.stability_score for c in coalitions]), 3),
            "active_ratio": round(active_count / len(coalitions), 3)
        }
    
    def _classify_cascade(self, formed_pairs: List[Tuple[str, str]]) -> str:
        """Classify the type of cascade event."""
        # Check if all pairs share a common model
        all_models = []
        for pair in formed_pairs:
            all_models.extend(pair)
        
        model_counts = defaultdict(int)
        for model in all_models:
            model_counts[model] += 1
        
        max_count = max(model_counts.values())
        
        if max_count == len(formed_pairs):
            return "hub_cascade"  # One model triggers multiple coalitions
        elif len(set(all_models)) == len(formed_pairs) + 1:
            return "chain_cascade"  # Sequential coalition formation
        else:
            return "network_cascade"  # Complex multi-model cascade
    
    def _analyze_defection_timing(self, defection_events: List[Dict]) -> Dict:
        """Analyze when defections tend to occur."""
        if not defection_events:
            return {}
        
        rounds = [e["round"] for e in defection_events]
        
        return {
            "earliest_defection": min(rounds),
            "latest_defection": max(rounds),
            "median_defection_round": int(np.median(rounds)),
            "defection_clustering": "early" if np.mean(rounds) < 5 else "late"
        }
    
    def _interpret_coalition_patterns(self, same_model_stats: Dict, 
                                    cross_model_stats: Dict, 
                                    comparison: Dict) -> str:
        """Provide interpretation of coalition patterns."""
        if not same_model_stats["count"] and not cross_model_stats["count"]:
            return "No stable coalitions detected in the experiment."
        
        if not cross_model_stats["count"]:
            return "Coalitions only form between agents of the same model type."
        
        if not same_model_stats["count"]:
            return "Coalitions only form between agents of different model types."
        
        # Compare metrics
        if same_model_stats["avg_stability"] > cross_model_stats["avg_stability"] + 0.1:
            return "Same-model coalitions are significantly more stable than cross-model coalitions."
        elif cross_model_stats["avg_stability"] > same_model_stats["avg_stability"] + 0.1:
            return "Cross-model coalitions show surprising stability, exceeding same-model coalitions."
        else:
            return "Both same-model and cross-model coalitions show similar stability patterns."