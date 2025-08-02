"""Visualization utilities for mixed model coalition analysis."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CoalitionVisualizer:
    """Generate visualization data for coalition analysis in mixed scenarios."""
    
    @staticmethod
    def create_coalition_timeline(temporal_data: Dict) -> Dict:
        """
        Create timeline visualization of coalition formation and dissolution.
        
        Args:
            temporal_data: Temporal coalition analysis data
            
        Returns:
            Dictionary with timeline visualization data
        """
        stability_metrics = temporal_data.get("stability_metrics", {})
        
        # Extract timeline events
        events = []
        
        for pair, metrics in stability_metrics.items():
            # Formation event
            if "formation_round" in metrics:
                events.append({
                    "round": metrics.get("formation_round", 1),
                    "type": "formation",
                    "coalition": list(pair),
                    "strength": metrics.get("average_strength", 0)
                })
            
            # Disruption events
            if metrics.get("disruptions", 0) > 0:
                # Estimate disruption rounds (simplified)
                events.append({
                    "round": metrics.get("duration", 1) // 2,
                    "type": "disruption",
                    "coalition": list(pair),
                    "disruptions": metrics["disruptions"]
                })
        
        # Sort by round
        events.sort(key=lambda x: x["round"])
        
        # Create timeline data
        timeline_data = {
            "type": "timeline",
            "title": "Coalition Formation and Evolution",
            "events": events,
            "x_axis": {
                "label": "Round",
                "min": 1,
                "max": max([e["round"] for e in events]) if events else 10
            },
            "categories": ["formation", "disruption", "dissolution"],
            "color_scheme": {
                "formation": "#2ca02c",  # Green
                "disruption": "#ff7f0e",  # Orange
                "dissolution": "#d62728"  # Red
            }
        }
        
        return timeline_data
    
    @staticmethod
    def create_stability_comparison_chart(temporal_data: Dict) -> Dict:
        """
        Create chart comparing stability between coalition types.
        
        Args:
            temporal_data: Temporal coalition analysis data
            
        Returns:
            Dictionary with comparison chart data
        """
        cross_vs_same = temporal_data.get("cross_vs_same_model", {})
        
        chart_data = {
            "type": "grouped_bar",
            "title": "Coalition Stability: Same-Model vs Cross-Model",
            "x_axis": {
                "label": "Coalition Type",
                "categories": ["Same Model", "Cross Model"]
            },
            "y_axis": {
                "label": "Metric Value",
                "min": 0,
                "max": 1
            },
            "series": [
                {
                    "name": "Average Strength",
                    "data": [
                        cross_vs_same.get("same_model", {}).get("avg_strength", 0),
                        cross_vs_same.get("cross_model", {}).get("avg_strength", 0)
                    ]
                },
                {
                    "name": "Average Stability",
                    "data": [
                        cross_vs_same.get("same_model", {}).get("avg_stability", 0),
                        cross_vs_same.get("cross_model", {}).get("avg_stability", 0)
                    ]
                },
                {
                    "name": "Active Ratio",
                    "data": [
                        cross_vs_same.get("same_model", {}).get("active_ratio", 0),
                        cross_vs_same.get("cross_model", {}).get("active_ratio", 0)
                    ]
                }
            ]
        }
        
        return chart_data
    
    @staticmethod
    def create_coalition_network_graph(network_data: Dict) -> Dict:
        """
        Create network graph visualization data.
        
        Args:
            network_data: Coalition network data
            
        Returns:
            Dictionary with network visualization data
        """
        graph_data = {
            "type": "network_graph",
            "title": "Active Coalition Network",
            "layout": "force-directed",
            "nodes": [],
            "edges": [],
            "node_styling": {
                "size_by": "degree",
                "color_by": "centrality",
                "min_size": 20,
                "max_size": 50
            },
            "edge_styling": {
                "width_by": "weight",
                "color_by": "stability",
                "min_width": 1,
                "max_width": 5
            }
        }
        
        # Process nodes
        for node in network_data.get("nodes", []):
            graph_data["nodes"].append({
                "id": node["id"],
                "label": node["label"],
                "size": 20 + 30 * node.get("centrality", 0),  # Scale by centrality
                "color": CoalitionVisualizer._get_node_color(node["centrality"]),
                "metrics": {
                    "degree": node["degree"],
                    "strength": node["strength"],
                    "centrality": node["centrality"]
                }
            })
        
        # Process edges
        for edge in network_data.get("edges", []):
            graph_data["edges"].append({
                "source": edge["source"],
                "target": edge["target"],
                "weight": edge["weight"],
                "width": 1 + 4 * edge["weight"],  # Scale by weight
                "color": CoalitionVisualizer._get_edge_color(edge["stability"]),
                "label": f"{edge['weight']:.2f}",
                "metrics": {
                    "stability": edge["stability"],
                    "duration": edge["duration"]
                }
            })
        
        return graph_data
    
    @staticmethod
    def create_defection_heatmap(defection_data: Dict) -> Dict:
        """
        Create heatmap showing defection patterns.
        
        Args:
            defection_data: Defection pattern analysis
            
        Returns:
            Dictionary with heatmap data
        """
        defections_by_model = defection_data.get("defections_by_model", {})
        
        if not defections_by_model:
            return {"type": "heatmap", "error": "No defection data available"}
        
        models = sorted(defections_by_model.keys())
        
        # Create defection matrix (simplified - diagonal shows self-defections)
        matrix = []
        for model1 in models:
            row = []
            for model2 in models:
                if model1 == model2:
                    # Self-defection count
                    value = defections_by_model.get(model1, 0)
                else:
                    # For cross-model, we'd need pair-specific data
                    value = 0  # Placeholder
                row.append(value)
            matrix.append(row)
        
        heatmap_data = {
            "type": "heatmap",
            "title": "Defection Patterns by Model",
            "x_axis": {
                "label": "Model",
                "categories": models
            },
            "y_axis": {
                "label": "Model",
                "categories": models
            },
            "matrix": matrix,
            "color_scale": {
                "min": 0,
                "max": max(defections_by_model.values()) if defections_by_model else 1,
                "scheme": "Reds",  # Red scale for defections
                "zero_color": "#ffffff"
            },
            "annotations": []
        }
        
        # Add annotations for high defection cells
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if matrix[i][j] > 0:
                    heatmap_data["annotations"].append({
                        "x": j,
                        "y": i,
                        "text": str(matrix[i][j])
                    })
        
        return heatmap_data
    
    @staticmethod
    def create_cascade_visualization(cascade_data: List[Dict]) -> Dict:
        """
        Create visualization for coalition cascade events.
        
        Args:
            cascade_data: List of cascade events
            
        Returns:
            Dictionary with cascade visualization data
        """
        if not cascade_data:
            return {"type": "cascade", "error": "No cascade events detected"}
        
        viz_data = {
            "type": "cascade_flow",
            "title": "Coalition Cascade Events",
            "cascades": []
        }
        
        for cascade in cascade_data:
            cascade_viz = {
                "round": cascade["round"],
                "type": cascade["type"],
                "size": cascade["coalitions_formed"],
                "flow": []
            }
            
            # Create flow diagram for the cascade
            models = cascade["models_involved"]
            for i, pair in enumerate(cascade["pairs"]):
                cascade_viz["flow"].append({
                    "from": pair[0],
                    "to": pair[1],
                    "order": i + 1,
                    "label": f"Coalition {i+1}"
                })
            
            viz_data["cascades"].append(cascade_viz)
        
        return viz_data
    
    @staticmethod
    def create_mixed_scenario_dashboard(coalition_analysis: Dict, 
                                      diversity_score: float) -> Dict:
        """
        Create comprehensive dashboard for mixed scenario analysis.
        
        Args:
            coalition_analysis: Full coalition analysis results
            diversity_score: Model diversity score for the scenario
            
        Returns:
            Dictionary with complete dashboard data
        """
        temporal = coalition_analysis.get("temporal_analysis", {})
        
        dashboard = {
            "title": f"Mixed Model Coalition Analysis (Diversity: {diversity_score:.3f})",
            "summary_metrics": {
                "total_coalitions": len(coalition_analysis.get("all_pairings", [])),
                "active_coalitions": len([c for c in coalition_analysis.get("all_pairings", []) 
                                        if c.get("strength", 0) > 0.7]),
                "strongest_coalition": coalition_analysis.get("strongest_pair"),
                "model_diversity": diversity_score
            },
            "visualizations": {}
        }
        
        # Add visualizations if temporal data available
        if temporal:
            dashboard["visualizations"]["timeline"] = CoalitionVisualizer.create_coalition_timeline(temporal)
            dashboard["visualizations"]["stability_comparison"] = CoalitionVisualizer.create_stability_comparison_chart(temporal)
            
            if "network_data" in temporal:
                dashboard["visualizations"]["network"] = CoalitionVisualizer.create_coalition_network_graph(
                    temporal["network_data"]
                )
            
            if "defection_patterns" in temporal:
                dashboard["visualizations"]["defections"] = CoalitionVisualizer.create_defection_heatmap(
                    temporal["defection_patterns"]
                )
            
            if "coalition_cascades" in temporal:
                dashboard["visualizations"]["cascades"] = CoalitionVisualizer.create_cascade_visualization(
                    temporal["coalition_cascades"]
                )
        
        # Add interpretation
        dashboard["interpretation"] = CoalitionVisualizer._interpret_mixed_scenario(
            coalition_analysis, diversity_score, temporal
        )
        
        return dashboard
    
    @staticmethod
    def _get_node_color(centrality: float) -> str:
        """Get color based on centrality score."""
        # Green to red gradient
        if centrality < 0.33:
            return "#d62728"  # Red - peripheral
        elif centrality < 0.67:
            return "#ff7f0e"  # Orange - moderate
        else:
            return "#2ca02c"  # Green - central
    
    @staticmethod
    def _get_edge_color(stability: float) -> str:
        """Get color based on stability score."""
        # Blue gradient for stability
        if stability < 0.5:
            return "#c7c7c7"  # Light gray - unstable
        elif stability < 0.7:
            return "#9ecae1"  # Light blue - moderate
        else:
            return "#3182bd"  # Dark blue - stable
    
    @staticmethod
    def _interpret_mixed_scenario(coalition_analysis: Dict, 
                                diversity_score: float,
                                temporal_data: Dict) -> str:
        """Generate interpretation of mixed scenario results."""
        interpretations = []
        
        # Diversity impact
        if diversity_score == 0:
            interpretations.append("This is a homogeneous scenario with no model diversity.")
        elif diversity_score < 0.7:
            interpretations.append(f"Moderate model diversity (H={diversity_score:.3f}) creates balanced coalition opportunities.")
        else:
            interpretations.append(f"High model diversity (H={diversity_score:.3f}) leads to complex coalition dynamics.")
        
        # Coalition patterns
        if coalition_analysis.get("detected"):
            interpretations.append("Strong coalition patterns detected in the experiment.")
        
        # Temporal patterns if available
        if temporal_data:
            cross_vs_same = temporal_data.get("cross_vs_same_model", {})
            if cross_vs_same:
                same_stats = cross_vs_same.get("same_model", {})
                cross_stats = cross_vs_same.get("cross_model", {})
                
                if same_stats.get("avg_stability", 0) > cross_stats.get("avg_stability", 0):
                    interpretations.append("Same-model coalitions show higher stability than cross-model coalitions.")
                else:
                    interpretations.append("Cross-model coalitions demonstrate surprising stability.")
            
            # Cascade events
            cascades = temporal_data.get("coalition_cascades", [])
            if cascades:
                interpretations.append(f"Detected {len(cascades)} coalition cascade events, suggesting social contagion effects.")
        
        return " ".join(interpretations)