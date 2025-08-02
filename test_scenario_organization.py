"""Tests for scenario-based data organization and comparison."""
import pytest
import tempfile
from pathlib import Path
import json
from unittest.mock import Mock, patch

from src.utils.data_manager import DataManager
from src.utils.scenario_comparator import ScenarioComparator, ScenarioResult
from src.core.models import StrategyRecord, GameResult, RoundSummary, ExperimentResult


class TestScenarioDataOrganization:
    """Test suite for scenario-based data organization."""
    
    def test_data_manager_scenario_paths(self):
        """Test that DataManager creates correct scenario paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Without scenario
            dm1 = DataManager(base_path=tmpdir)
            assert "scenarios" not in str(dm1.experiment_path)
            
            # With scenario
            dm2 = DataManager(base_path=tmpdir, scenario_name="test_scenario")
            assert "scenarios/test_scenario" in str(dm2.experiment_path)
            assert dm2.experiment_path.exists()
    
    def test_scenario_specific_filenames(self):
        """Test that files include scenario names when appropriate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(base_path=tmpdir, scenario_name="mixed_5_5")
            
            # Test strategy filename
            strategy = Mock(spec=StrategyRecord)
            strategy.round = 1
            strategy.agent_id = 0
            strategy.timestamp = "2024-01-01T00:00:00"
            strategy.model = "gpt-4"
            strategy.strategy_text = "cooperate"
            strategy.full_reasoning = "reasoning"
            strategy.prompt_tokens = 100
            strategy.completion_tokens = 50
            strategies = [strategy]
            dm.save_strategies(1, strategies)
            
            strategy_files = list((dm.experiment_path / "rounds").glob("strategies*.json"))
            assert len(strategy_files) == 1
            assert "mixed_5_5" in strategy_files[0].name
    
    def test_save_experiment_with_scenario_metadata(self):
        """Test that experiment results include scenario metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(base_path=tmpdir, scenario_name="test_scenario")
            
            result = ExperimentResult(
                experiment_id="test_exp",
                start_time="2024-01-01T00:00:00",
                end_time="2024-01-01T01:00:00"
            )
            
            dm.save_experiment_result(result)
            
            # Load and check
            result_files = list(dm.experiment_path.glob("experiment_results*.json"))
            assert len(result_files) == 1
            
            with open(result_files[0], 'r') as f:
                data = json.load(f)
            
            assert data["scenario_name"] == "test_scenario"
    
    def test_get_scenario_experiments(self):
        """Test retrieving all experiments for a scenario."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            
            # Create multiple experiments for a scenario
            scenario_path = base_path / "scenarios" / "test_scenario"
            for i in range(3):
                exp_path = scenario_path / f"exp_2024010{i}_120000"
                exp_path.mkdir(parents=True)
                
                # Add a result file
                with open(exp_path / "experiment_results.json", 'w') as f:
                    json.dump({"experiment_id": f"exp_{i}"}, f)
            
            dm = DataManager(base_path=tmpdir)
            experiments = dm.get_scenario_experiments("test_scenario")
            
            assert len(experiments) == 3
            assert all(exp.name.startswith("exp_") for exp in experiments)
    
    def test_load_scenario_data(self):
        """Test loading aggregated data for a scenario."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            
            # Create scenario with experiments
            scenario_path = base_path / "scenarios" / "test_scenario"
            exp_path = scenario_path / "exp_20240101_120000"
            exp_path.mkdir(parents=True)
            
            # Save experiment result
            result_data = {
                "experiment_id": "test_exp",
                "round_summaries": [{"round": 1, "cooperation_rate": 0.7}]
            }
            with open(exp_path / "experiment_results.json", 'w') as f:
                json.dump(result_data, f)
            
            dm = DataManager(base_path=tmpdir)
            loaded_data = dm.load_scenario_data("test_scenario", "experiment_results")
            
            assert len(loaded_data) == 1
            assert loaded_data[0]["experiment_id"] == "test_exp"
            assert "experiment_path" in loaded_data[0]


class TestScenarioComparator:
    """Test suite for ScenarioComparator."""
    
    def create_mock_scenario_structure(self, tmpdir: Path) -> None:
        """Create mock scenario directory structure."""
        scenarios_path = tmpdir / "scenarios"
        
        # Create multiple scenarios
        for scenario_name in ["homogeneous_gpt4", "mixed_5_5", "diverse_3_3_4"]:
            scenario_path = scenarios_path / scenario_name
            
            # Create experiments
            for i in range(2):
                exp_path = scenario_path / f"exp_2024010{i}_120000"
                exp_path.mkdir(parents=True)
                
                # Create experiment results
                cooperation_rate = 0.5 + (i * 0.1)  # Vary by experiment
                if "homogeneous" in scenario_name:
                    cooperation_rate += 0.1
                elif "diverse" in scenario_name:
                    cooperation_rate -= 0.05
                
                result = {
                    "experiment_id": f"{scenario_name}_exp_{i}",
                    "scenario_name": scenario_name,
                    "round_summaries": [
                        {"round": r, "cooperation_rate": cooperation_rate + r * 0.01}
                        for r in range(1, 6)
                    ]
                }
                
                with open(exp_path / f"experiment_results_{scenario_name}.json", 'w') as f:
                    json.dump(result, f)
                
                # Create scenario config
                model_dist = {
                    "homogeneous_gpt4": {"gpt-4": 10},
                    "mixed_5_5": {"gpt-4": 5, "claude-3": 5},
                    "diverse_3_3_4": {"gpt-4": 3, "claude-3": 3, "gemini": 4}
                }[scenario_name]
                
                diversity = {
                    "homogeneous_gpt4": 0.0,
                    "mixed_5_5": 0.693,
                    "diverse_3_3_4": 1.099
                }[scenario_name]
                
                config = {
                    "scenario_name": scenario_name,
                    "model_distribution": model_dist,
                    "diversity_score": diversity
                }
                
                with open(exp_path / "scenario_config.json", 'w') as f:
                    json.dump(config, f)
    
    def test_load_scenario_results(self):
        """Test loading results from scenario structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self.create_mock_scenario_structure(tmppath)
            
            comparator = ScenarioComparator(base_path=tmpdir)
            results = comparator.load_scenario_results()
            
            assert len(results) == 3  # Three scenarios
            assert "homogeneous_gpt4" in results
            assert "mixed_5_5" in results
            assert "diverse_3_3_4" in results
            
            # Check each scenario has experiments
            for scenario_name, experiments in results.items():
                assert len(experiments) == 2
                assert all(isinstance(exp, ScenarioResult) for exp in experiments)
    
    def test_compare_scenarios_by_cooperation(self):
        """Test comparing scenarios by cooperation rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self.create_mock_scenario_structure(tmppath)
            
            comparator = ScenarioComparator(base_path=tmpdir)
            comparator.load_scenario_results()
            
            comparison = comparator.compare_scenarios("overall_cooperation_rate")
            
            assert comparison["metric"] == "overall_cooperation_rate"
            assert comparison["best_scenario"] == "homogeneous_gpt4"  # Has highest rates
            assert len(comparison["rankings"]) == 3
            assert comparison["scenario_averages"]["homogeneous_gpt4"] > \
                   comparison["scenario_averages"]["diverse_3_3_4"]
    
    def test_analyze_diversity_impact(self):
        """Test diversity impact analysis across scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self.create_mock_scenario_structure(tmppath)
            
            comparator = ScenarioComparator(base_path=tmpdir)
            comparator.load_scenario_results()
            
            impact = comparator.analyze_diversity_impact()
            
            assert impact["total_experiments"] == 6  # 2 per scenario, 3 scenarios
            assert "diversity_groups" in impact
            assert "homogeneous" in impact["diversity_groups"]
            assert "low_diversity" in impact["diversity_groups"]  # 0.693 falls into low
            assert "high_diversity" in impact["diversity_groups"]
    
    def test_generate_comparison_report(self):
        """Test full comparison report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self.create_mock_scenario_structure(tmppath)
            
            comparator = ScenarioComparator(base_path=tmpdir)
            report = comparator.generate_comparison_report()
            
            assert "summary" in report
            assert report["summary"]["total_scenarios"] == 3
            assert report["summary"]["total_experiments"] == 6
            
            assert "cooperation_comparison" in report
            assert "diversity_impact" in report
            assert "scenario_profiles" in report
            assert "recommendations" in report
            
            # Test saving report
            report_path = tmppath / "comparison_report.json"
            comparator.generate_comparison_report(report_path)
            assert report_path.exists()
    
    def test_find_best_scenario_for_metric(self):
        """Test finding optimal scenario for target metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self.create_mock_scenario_structure(tmppath)
            
            comparator = ScenarioComparator(base_path=tmpdir)
            comparator.load_scenario_results()
            
            # Find scenario closest to 0.6 cooperation rate
            best = comparator.find_best_scenario_for_metric(
                "overall_cooperation_rate", 0.6
            )
            
            assert best is not None
            assert isinstance(best, str)
    
    def test_scenario_filtering_options(self):
        """Test getting available filtering options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self.create_mock_scenario_structure(tmppath)
            
            comparator = ScenarioComparator(base_path=tmpdir)
            comparator.load_scenario_results()
            
            options = comparator.get_scenario_filtering_options()
            
            assert "scenarios" in options
            assert len(options["scenarios"]) == 3
            assert "model_types" in options
            assert "gpt-4" in options["model_types"]
            assert "diversity_levels" in options
            assert "metrics" in options