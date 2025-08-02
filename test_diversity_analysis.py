"""Tests for diversity analysis components."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from src.utils.diversity_analyzer import DiversityAnalyzer, DiversityImpactResult
from src.utils.diversity_statistics import DiversityStatistics
from src.utils.diversity_visualizer import DiversityVisualizer
from src.core.models import ExperimentResult, RoundSummary, Agent
from src.core.scenario_manager import ScenarioManager, ScenarioConfig


class TestDiversityAnalyzer:
    """Test suite for DiversityAnalyzer."""
    
    def create_mock_experiment_result(self, cooperation_rates: list) -> ExperimentResult:
        """Create mock experiment result with given cooperation rates."""
        result = ExperimentResult(
            experiment_id="test_exp",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T01:00:00"
        )
        
        result.round_summaries = [
            RoundSummary(
                round=i+1,
                cooperation_rate=rate,
                average_score=50.0,
                score_variance=10.0,
                power_distribution={"mean": 100, "std": 5, "min": 90, "max": 110}
            )
            for i, rate in enumerate(cooperation_rates)
        ]
        
        return result
    
    def test_analyze_experiment(self):
        """Test analyzing a single experiment."""
        analyzer = DiversityAnalyzer()
        
        # Create mock data
        experiment = self.create_mock_experiment_result([0.6, 0.65, 0.7, 0.72, 0.75])
        
        # Create scenario manager
        manager = ScenarioManager(num_agents=10)
        scenario = ScenarioConfig(
            name="test_balanced",
            model_distribution={"gpt-4": 5, "claude-3": 5}
        )
        agents = [Agent(id=i) for i in range(10)]
        manager.assign_models_to_agents(agents, scenario)
        
        # Analyze
        result = analyzer.analyze_experiment(experiment, manager)
        
        assert result.scenario_name == "test_balanced"
        assert 0.69 < result.diversity_score < 0.70  # Shannon entropy for 50-50
        assert result.overall_cooperation_rate == pytest.approx(0.684, rel=0.01)
        assert len(result.cooperation_by_round) == 5
        assert result.cooperation_trend > 0  # Positive trend
    
    def test_compare_scenarios(self):
        """Test comparing multiple scenarios."""
        analyzer = DiversityAnalyzer()
        
        # Create mock results
        results = [
            DiversityImpactResult(
                scenario_name="homogeneous",
                diversity_score=0.0,
                overall_cooperation_rate=0.5,
                cooperation_by_round=[0.5] * 10,
                cooperation_trend=0.0,
                correlation_coefficient=0.0,
                p_value=1.0,
                model_proportions={"gpt-4": 1.0}
            ),
            DiversityImpactResult(
                scenario_name="balanced",
                diversity_score=0.693,
                overall_cooperation_rate=0.7,
                cooperation_by_round=[0.65, 0.70, 0.72, 0.73, 0.70],
                cooperation_trend=0.015,
                correlation_coefficient=0.8,
                p_value=0.05,
                model_proportions={"gpt-4": 0.5, "claude-3": 0.5}
            ),
            DiversityImpactResult(
                scenario_name="diverse",
                diversity_score=1.099,
                overall_cooperation_rate=0.8,
                cooperation_by_round=[0.75, 0.78, 0.82, 0.85, 0.80],
                cooperation_trend=0.02,
                correlation_coefficient=0.9,
                p_value=0.01,
                model_proportions={"gpt-4": 0.33, "claude-3": 0.33, "gemini": 0.34}
            )
        ]
        
        comparison = analyzer.compare_scenarios(results)
        
        assert comparison["total_scenarios"] == 3
        assert comparison["best_performing_scenario"] == "diverse"
        assert comparison["highest_diversity_scenario"] == "diverse"
        assert comparison["cooperation_by_diversity_level"]["homogeneous"]["avg_cooperation"] == 0.5
        assert comparison["cooperation_by_diversity_level"]["balanced"]["avg_cooperation"] == 0.7
        assert comparison["cooperation_by_diversity_level"]["diverse"]["avg_cooperation"] == 0.8
    
    def test_track_diversity_evolution(self):
        """Test tracking diversity evolution across rounds."""
        analyzer = DiversityAnalyzer()
        
        # Create round summaries with varying cooperation
        summaries = [
            RoundSummary(round=i+1, cooperation_rate=rate, average_score=50, 
                        score_variance=5, power_distribution={})
            for i, rate in enumerate([0.4, 0.45, 0.5, 0.6, 0.65, 0.7, 0.72, 0.73, 0.74, 0.75])
        ]
        
        manager = ScenarioManager(num_agents=10)
        scenario = ScenarioConfig(name="test", model_distribution={"gpt-4": 5, "claude-3": 5})
        agents = [Agent(id=i) for i in range(10)]
        manager.assign_models_to_agents(agents, scenario)
        
        evolution = analyzer.track_diversity_evolution(summaries, manager)
        
        assert evolution["diversity_score"] > 0
        assert evolution["cooperation_evolution"]["early_phase"]["rounds"] == 3
        assert evolution["cooperation_evolution"]["late_phase"]["avg_cooperation"] > \
               evolution["cooperation_evolution"]["early_phase"]["avg_cooperation"]
        assert evolution["stability_metrics"]["convergence_achieved"] is True
    
    def test_save_analysis(self):
        """Test saving analysis results."""
        analyzer = DiversityAnalyzer()
        
        # Add some results
        analyzer.results = [
            DiversityImpactResult(
                scenario_name="test1",
                diversity_score=0.5,
                overall_cooperation_rate=0.6,
                cooperation_by_round=[0.6],
                cooperation_trend=0.0,
                correlation_coefficient=0.5,
                p_value=0.1,
                model_proportions={"gpt-4": 0.5, "claude-3": 0.5}
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            analyzer.save_analysis(temp_path)
            
            # Verify saved data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "individual_results" in data
            assert len(data["individual_results"]) == 1
            assert data["individual_results"][0]["scenario_name"] == "test1"
            assert "comparative_analysis" in data
            assert "visualization_data" in data
        finally:
            temp_path.unlink()


class TestDiversityStatistics:
    """Test suite for DiversityStatistics."""
    
    def test_diversity_cooperation_correlation(self):
        """Test correlation analysis between diversity and cooperation."""
        # Create synthetic data with positive correlation
        diversity_scores = [0.0, 0.3, 0.5, 0.693, 0.8, 1.0, 1.099]
        cooperation_rates = [0.4, 0.5, 0.55, 0.65, 0.7, 0.75, 0.8]
        
        result = DiversityStatistics.test_diversity_cooperation_correlation(
            diversity_scores, cooperation_rates
        )
        
        assert result["n_samples"] == 7
        assert result["pearson_correlation"]["coefficient"] > 0.9
        assert result["pearson_correlation"]["significant"] is True
        assert "strong positive" in result["pearson_correlation"]["interpretation"]
        assert result["linear_regression"]["slope"] > 0
        assert result["effect_size"]["cohens_d"] is not None
    
    def test_diversity_groups_anova(self):
        """Test ANOVA for diversity groups."""
        scenario_results = [
            {"diversity_score": 0.0, "overall_cooperation_rate": 0.4},
            {"diversity_score": 0.0, "overall_cooperation_rate": 0.45},
            {"diversity_score": 0.0, "overall_cooperation_rate": 0.42},
            {"diversity_score": 0.693, "overall_cooperation_rate": 0.6},
            {"diversity_score": 0.693, "overall_cooperation_rate": 0.65},
            {"diversity_score": 0.7, "overall_cooperation_rate": 0.62},
            {"diversity_score": 1.099, "overall_cooperation_rate": 0.75},
            {"diversity_score": 1.099, "overall_cooperation_rate": 0.8},
            {"diversity_score": 1.1, "overall_cooperation_rate": 0.78}
        ]
        
        result = DiversityStatistics.test_diversity_groups_anova(scenario_results)
        
        assert result["test"] == "One-way ANOVA"
        assert result["groups"]["homogeneous"]["n"] == 3
        assert result["groups"]["balanced"]["n"] == 3
        assert result["groups"]["diverse"]["n"] == 3
        assert result["f_statistic"] > 0
        # With clear differences, should be significant
        assert result["significant"] is True
    
    def test_diversity_stability(self):
        """Test diversity stability analysis."""
        cooperation_data = {
            0.0: [0.5, 0.5, 0.51, 0.49, 0.5, 0.5, 0.5],  # Stable
            0.693: [0.4, 0.5, 0.6, 0.55, 0.65, 0.7, 0.72],  # Increasing
            1.099: [0.3, 0.6, 0.4, 0.7, 0.5, 0.8, 0.6]  # Volatile
        }
        
        result = DiversityStatistics.test_diversity_stability(cooperation_data)
        
        assert "individual_stability" in result
        assert "diversity_0.000" in result["individual_stability"]
        assert "diversity_0.693" in result["individual_stability"]
        assert "diversity_1.099" in result["individual_stability"]
        
        # Check that homogeneous has lowest variance
        homo_var = result["individual_stability"]["diversity_0.000"]["variance"]
        diverse_var = result["individual_stability"]["diversity_1.099"]["variance"]
        assert homo_var < diverse_var
        
        # Check trend detection
        balanced_trend = result["individual_stability"]["diversity_0.693"]["trend"]
        assert balanced_trend["slope"] > 0
        assert balanced_trend["has_significant_trend"] is True
    
    def test_edge_cases(self):
        """Test edge cases for statistical functions."""
        # Too few data points
        result = DiversityStatistics.test_diversity_cooperation_correlation([0.5], [0.6])
        assert "error" in result
        
        # Perfect correlation
        x = [0, 1, 2, 3, 4]
        y = [0, 2, 4, 6, 8]
        result = DiversityStatistics.test_diversity_cooperation_correlation(x, y)
        assert result["pearson_correlation"]["coefficient"] == pytest.approx(1.0)
        
        # No variance in groups
        scenario_results = [
            {"diversity_score": 0.0, "overall_cooperation_rate": 0.5},
            {"diversity_score": 0.0, "overall_cooperation_rate": 0.5}
        ]
        result = DiversityStatistics.test_diversity_groups_anova(scenario_results)
        assert result["n_groups"] < 2


class TestDiversityVisualizer:
    """Test suite for DiversityVisualizer."""
    
    def test_create_scatter_plot(self):
        """Test scatter plot data generation."""
        results = [
            {"diversity_score": 0.0, "overall_cooperation_rate": 0.5, "scenario_name": "homo"},
            {"diversity_score": 0.693, "overall_cooperation_rate": 0.65, "scenario_name": "balanced"},
            {"diversity_score": 1.099, "overall_cooperation_rate": 0.8, "scenario_name": "diverse"}
        ]
        
        scatter = DiversityVisualizer.create_diversity_cooperation_scatter(results)
        
        assert scatter["type"] == "scatter"
        assert len(scatter["points"]) == 3
        assert scatter["x_axis"]["label"] == "Model Diversity (Shannon Entropy)"
        assert scatter["y_axis"]["label"] == "Overall Cooperation Rate"
        assert len(scatter["annotations"]) > 0
    
    def test_create_evolution_lines(self):
        """Test cooperation evolution line plot."""
        results = [
            {
                "diversity_score": 0.0,
                "cooperation_by_round": [0.5, 0.5, 0.5, 0.5, 0.5],
                "scenario_name": "homo1"
            },
            {
                "diversity_score": 0.693,
                "cooperation_by_round": [0.6, 0.65, 0.7, 0.72, 0.75],
                "scenario_name": "balanced1"
            }
        ]
        
        lines = DiversityVisualizer.create_cooperation_evolution_lines(results)
        
        assert lines["type"] == "multi_line"
        assert len(lines["series"]) >= 1
        assert lines["x_axis"]["label"] == "Round"
        assert lines["y_axis"]["label"] == "Cooperation Rate"
    
    def test_create_bar_chart(self):
        """Test bar chart generation."""
        results = [
            {"diversity_score": 0.0, "overall_cooperation_rate": 0.5},
            {"diversity_score": 0.0, "overall_cooperation_rate": 0.45},
            {"diversity_score": 0.693, "overall_cooperation_rate": 0.7},
            {"diversity_score": 1.099, "overall_cooperation_rate": 0.8}
        ]
        
        bars = DiversityVisualizer.create_diversity_distribution_bar(results)
        
        assert bars["type"] == "bar"
        assert len(bars["bars"]) > 0
        assert bars["bars"][0]["n"] > 0
        assert "error_bars" in bars
    
    def test_create_heatmap(self):
        """Test coalition heatmap generation."""
        coalition_data = {
            "gpt-4": {"gpt-4": 0.8, "claude-3": 0.6},
            "claude-3": {"claude-3": 0.75, "gpt-4": 0.6}
        }
        
        heatmap = DiversityVisualizer.create_model_coalition_heatmap(coalition_data)
        
        assert heatmap["type"] == "heatmap"
        assert len(heatmap["cells"]) > 0
        assert heatmap["color_scale"]["scheme"] == "RdYlGn"
    
    def test_create_dashboard(self):
        """Test complete dashboard generation."""
        results = [
            {
                "diversity_score": 0.693,
                "overall_cooperation_rate": 0.7,
                "cooperation_by_round": [0.65, 0.7, 0.72],
                "scenario_name": "test"
            }
        ]
        
        dashboard = DiversityVisualizer.create_diversity_impact_dashboard(results)
        
        assert "title" in dashboard
        assert "summary_stats" in dashboard
        assert "visualizations" in dashboard
        assert "scatter" in dashboard["visualizations"]
        assert "evolution" in dashboard["visualizations"]
        assert "bars" in dashboard["visualizations"]