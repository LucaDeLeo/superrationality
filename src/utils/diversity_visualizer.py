"""Visualization utilities for model diversity analysis."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DiversityVisualizer:
    """Generate visualization data for diversity impact analysis."""
    
    @staticmethod
    def create_diversity_cooperation_scatter(results: List[Dict]) -> Dict:
        """
        Create scatter plot data for diversity vs cooperation.
        
        Args:
            results: List of experiment results with diversity scores
            
        Returns:
            Dictionary with plot data
        """
        scatter_data = {
            "type": "scatter",
            "title": "Model Diversity vs Cooperation Rate",
            "x_axis": {
                "label": "Model Diversity (Shannon Entropy)",
                "data": []
            },
            "y_axis": {
                "label": "Overall Cooperation Rate",
                "data": []
            },
            "points": [],
            "annotations": []
        }
        
        for result in results:
            diversity = result.get('diversity_score', 0)
            cooperation = result.get('overall_cooperation_rate', 0)
            scenario = result.get('scenario_name', 'unknown')
            
            scatter_data["x_axis"]["data"].append(diversity)
            scatter_data["y_axis"]["data"].append(cooperation)
            scatter_data["points"].append({
                "x": diversity,
                "y": cooperation,
                "label": scenario,
                "size": 10
            })
        
        # Add diversity level annotations
        scatter_data["annotations"] = [
            {"x": 0, "y": 0.95, "text": "Homogeneous", "anchor": "left"},
            {"x": 0.693, "y": 0.95, "text": "Balanced", "anchor": "center"},
            {"x": 1.099, "y": 0.95, "text": "Diverse", "anchor": "center"}
        ]
        
        return scatter_data
    
    @staticmethod
    def create_cooperation_evolution_lines(results: List[Dict]) -> Dict:
        """
        Create line plot data for cooperation evolution by diversity level.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary with line plot data
        """
        # Group by diversity level
        grouped_data = {
            "homogeneous": [],
            "balanced": [],
            "diverse": []
        }
        
        for result in results:
            diversity = result.get('diversity_score', 0)
            cooperation_by_round = result.get('cooperation_by_round', [])
            scenario = result.get('scenario_name', 'unknown')
            
            if diversity == 0:
                group = "homogeneous"
            elif diversity < 0.8:
                group = "balanced"
            else:
                group = "diverse"
            
            grouped_data[group].append({
                "scenario": scenario,
                "diversity": diversity,
                "data": cooperation_by_round
            })
        
        line_data = {
            "type": "multi_line",
            "title": "Cooperation Evolution by Model Diversity",
            "x_axis": {
                "label": "Round",
                "data": list(range(1, 11))  # Assuming 10 rounds
            },
            "y_axis": {
                "label": "Cooperation Rate",
                "min": 0,
                "max": 1
            },
            "series": []
        }
        
        # Add average lines for each group
        for group, scenarios in grouped_data.items():
            if scenarios:
                # Calculate average cooperation by round
                max_rounds = max(len(s['data']) for s in scenarios)
                avg_cooperation = []
                
                for round_idx in range(max_rounds):
                    round_values = [s['data'][round_idx] 
                                  for s in scenarios 
                                  if round_idx < len(s['data'])]
                    if round_values:
                        avg_cooperation.append(sum(round_values) / len(round_values))
                
                line_data["series"].append({
                    "name": f"{group.capitalize()} (n={len(scenarios)})",
                    "data": avg_cooperation,
                    "style": {
                        "homogeneous": {"color": "#1f77b4", "width": 2},
                        "balanced": {"color": "#ff7f0e", "width": 2},
                        "diverse": {"color": "#2ca02c", "width": 2}
                    }[group]
                })
        
        return line_data
    
    @staticmethod
    def create_diversity_distribution_bar(results: List[Dict]) -> Dict:
        """
        Create bar chart data for cooperation by diversity level.
        
        Args:
            results: List of experiment results
            
        Returns:
            Dictionary with bar chart data
        """
        # Group and average by diversity level
        groups = {
            "Homogeneous\n(H=0)": [],
            "Balanced\n(H≈0.69)": [],
            "Diverse\n(H>1.0)": []
        }
        
        for result in results:
            diversity = result.get('diversity_score', 0)
            cooperation = result.get('overall_cooperation_rate', 0)
            
            if diversity == 0:
                groups["Homogeneous\n(H=0)"].append(cooperation)
            elif diversity < 0.8:
                groups["Balanced\n(H≈0.69)"].append(cooperation)
            else:
                groups["Diverse\n(H>1.0)"].append(cooperation)
        
        bar_data = {
            "type": "bar",
            "title": "Average Cooperation Rate by Diversity Level",
            "x_axis": {
                "label": "Diversity Level",
                "categories": []
            },
            "y_axis": {
                "label": "Average Cooperation Rate",
                "min": 0,
                "max": 1
            },
            "bars": [],
            "error_bars": []
        }
        
        for category, values in groups.items():
            if values:
                import numpy as np
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                bar_data["x_axis"]["categories"].append(category)
                bar_data["bars"].append({
                    "category": category,
                    "value": mean_val,
                    "n": len(values)
                })
                bar_data["error_bars"].append({
                    "category": category,
                    "lower": max(0, mean_val - std_val),
                    "upper": min(1, mean_val + std_val)
                })
        
        return bar_data
    
    @staticmethod
    def create_model_coalition_heatmap(coalition_data: Dict[str, Dict[str, float]]) -> Dict:
        """
        Create heatmap data for inter-model cooperation.
        
        Args:
            coalition_data: Cooperation rates between model pairs
            
        Returns:
            Dictionary with heatmap data
        """
        models = sorted(set(
            list(coalition_data.keys()) + 
            [model for rates in coalition_data.values() for model in rates.keys()]
        ))
        
        heatmap_data = {
            "type": "heatmap",
            "title": "Inter-Model Cooperation Rates",
            "x_axis": {
                "label": "Model 2",
                "categories": models
            },
            "y_axis": {
                "label": "Model 1",
                "categories": models
            },
            "cells": [],
            "color_scale": {
                "min": 0,
                "max": 1,
                "scheme": "RdYlGn"  # Red-Yellow-Green
            }
        }
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if model1 in coalition_data and model2 in coalition_data[model1]:
                    value = coalition_data[model1][model2]
                elif model2 in coalition_data and model1 in coalition_data[model2]:
                    value = coalition_data[model2][model1]
                elif model1 == model2:
                    # Intra-model cooperation
                    value = coalition_data.get(model1, {}).get(model1, None)
                else:
                    value = None
                
                if value is not None:
                    heatmap_data["cells"].append({
                        "x": j,
                        "y": i,
                        "value": value,
                        "label": f"{value:.2f}"
                    })
        
        return heatmap_data
    
    @staticmethod
    def save_visualization_data(viz_data: Dict, output_path: Path) -> None:
        """
        Save visualization data to JSON file.
        
        Args:
            viz_data: Dictionary with all visualization data
            output_path: Path to save the data
        """
        with open(output_path, 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        logger.info(f"Saved visualization data to {output_path}")
    
    @staticmethod
    def create_diversity_impact_dashboard(results: List[Dict], 
                                        coalition_data: Optional[Dict] = None) -> Dict:
        """
        Create complete dashboard data for diversity impact.
        
        Args:
            results: List of experiment results
            coalition_data: Optional coalition cooperation data
            
        Returns:
            Dictionary with complete dashboard data
        """
        dashboard = {
            "title": "Model Diversity Impact Analysis",
            "summary_stats": DiversityVisualizer._calculate_summary_stats(results),
            "visualizations": {
                "scatter": DiversityVisualizer.create_diversity_cooperation_scatter(results),
                "evolution": DiversityVisualizer.create_cooperation_evolution_lines(results),
                "bars": DiversityVisualizer.create_diversity_distribution_bar(results)
            }
        }
        
        if coalition_data:
            dashboard["visualizations"]["heatmap"] = DiversityVisualizer.create_model_coalition_heatmap(coalition_data)
        
        return dashboard
    
    @staticmethod
    def _calculate_summary_stats(results: List[Dict]) -> Dict:
        """Calculate summary statistics for dashboard."""
        if not results:
            return {}
        
        diversity_scores = [r.get('diversity_score', 0) for r in results]
        cooperation_rates = [r.get('overall_cooperation_rate', 0) for r in results]
        
        import numpy as np
        
        return {
            "total_experiments": len(results),
            "diversity_range": {
                "min": min(diversity_scores),
                "max": max(diversity_scores),
                "mean": np.mean(diversity_scores)
            },
            "cooperation_range": {
                "min": min(cooperation_rates),
                "max": max(cooperation_rates),
                "mean": np.mean(cooperation_rates)
            },
            "best_scenario": max(results, key=lambda x: x.get('overall_cooperation_rate', 0)).get('scenario_name', 'unknown'),
            "most_diverse": max(results, key=lambda x: x.get('diversity_score', 0)).get('scenario_name', 'unknown')
        }