"""Unit tests for ReportGeneratorNode."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from src.nodes.report_generator import ReportGeneratorNode
from src.nodes.base import ContextKeys
from src.utils.data_manager import DataManager


class TestReportGeneratorNode:
    """Test suite for ReportGeneratorNode."""
    
    @pytest.fixture
    def mock_data_manager(self, tmp_path):
        """Create a mock data manager with temporary directory."""
        dm = Mock(spec=DataManager)
        dm.experiment_path = tmp_path
        dm._write_json = Mock()
        return dm
    
    @pytest.fixture
    def basic_context(self, mock_data_manager):
        """Create a basic context with minimal data."""
        return {
            ContextKeys.EXPERIMENT_ID: "test_exp_001",
            ContextKeys.DATA_MANAGER: mock_data_manager,
            "config": {
                "model": "gpt-4",
                "n_agents": 4,
                "n_rounds": 10,
                "temperature": 0.7
            },
            "transcript_analysis": {},
            "similarity_analysis": {},
            "statistical_analysis": {}
        }
    
    @pytest.fixture
    def full_context(self, mock_data_manager):
        """Create a full context with all analysis results."""
        return {
            ContextKeys.EXPERIMENT_ID: "test_exp_001",
            ContextKeys.DATA_MANAGER: mock_data_manager,
            "config": {
                "model": "gpt-4",
                "n_agents": 4,
                "n_rounds": 10,
                "temperature": 0.7
            },
            "transcript_analysis": {
                "marker_frequencies": {
                    "identity_reasoning": 0.73,
                    "logical_correlation": 0.45,
                    "mutual_benefit": 0.62,
                    "cooperation_despite_asymmetry": 0.38,
                    "precommitment": 0.15,
                    "simulation_awareness": 0.08
                },
                "agent_analyses": {
                    "0": {
                        "marker_frequencies": {
                            "cooperation_despite_asymmetry": 0.4
                        }
                    }
                }
            },
            "similarity_analysis": {
                "strategy_convergence": 0.81,
                "convergence_round": 6,
                "final_similarity": 0.85,
                "convergence_achieved": True,
                "strategy_evolution": [
                    {
                        "round": 1,
                        "avg_similarity": 0.67,
                        "clusters": [{"cluster_id": 0}, {"cluster_id": 1}],
                        "projection_2d": [
                            {"x": 0.12, "y": 0.45, "agent_id": 0, "cluster": 0},
                            {"x": 0.34, "y": 0.21, "agent_id": 1, "cluster": 1}
                        ]
                    }
                ],
                "strategy_clusters": {
                    "optimal_clusters": 2
                }
            },
            "statistical_analysis": {
                "overall_cooperation_rate": 0.72,
                "cooperation_trend": {
                    "direction": "increasing",
                    "slope": 0.035,
                    "p_value": 0.001,
                    "intercept": 0.6
                },
                "round_summaries": [
                    {
                        "round": 1,
                        "cooperation_rate": 0.60,
                        "mutual_cooperation_rate": 0.48,
                        "total_games": 6,
                        "avg_payoff": 2.5
                    },
                    {
                        "round": 10,
                        "cooperation_rate": 0.76,
                        "mutual_cooperation_rate": 0.65,
                        "total_games": 6,
                        "avg_payoff": 3.1
                    }
                ],
                "agent_statistics": {
                    "0": {
                        "cooperation_rate": 0.75,
                        "avg_power": 1.1,
                        "total_games": 30
                    },
                    "1": {
                        "cooperation_rate": 0.68,
                        "avg_power": 0.9,
                        "total_games": 30
                    }
                },
                "anomalies": [
                    {
                        "round": 4,
                        "type": "cooperation_spike",
                        "severity": "medium",
                        "description": "Unusual cooperation increase",
                        "cooperation_rate": 0.85,
                        "expected_rate": 0.68,
                        "deviation": 0.17
                    }
                ],
                "agent_differences": {
                    "p_value": 0.018
                },
                "total_games": 60
            }
        }
    
    @pytest.mark.asyncio
    async def test_synthesis_with_all_analyses(self, full_context):
        """Test synthesis logic with complete analysis results."""
        node = ReportGeneratorNode()
        
        # Execute node
        result = await node.execute(full_context)
        
        # Check synthesis was created
        assert "unified_report" in result
        synthesis = result["unified_report"]["synthesis"]
        
        # Check acausal score calculation
        assert "unified_metrics" in synthesis
        metrics = synthesis["unified_metrics"]
        assert metrics["acausal_cooperation_score"] > 0
        assert 0 <= metrics["acausal_cooperation_score"] <= 1
        
        # Check evidence assessment
        assert synthesis["acausal_cooperation_evidence"]["strength"] == "Strong"
        assert synthesis["acausal_cooperation_evidence"]["score"] > 0.7
        
        # Check correlations
        correlations = synthesis["correlations"]
        assert correlations["identity_cooperation"]["correlation"] == "positive_strong"
        assert correlations["similarity_convergence"]["correlation"] == "positive_strong"
        
        # Check key findings
        assert len(synthesis["key_findings"]) >= 4
        assert any("Acausal Cooperation Evidence" in f for f in synthesis["key_findings"])
    
    @pytest.mark.asyncio
    async def test_synthesis_with_missing_analyses(self, basic_context):
        """Test synthesis handles missing analysis components gracefully."""
        node = ReportGeneratorNode()
        
        # Execute with empty analyses
        result = await node.execute(basic_context)
        
        # Should still generate report
        assert "unified_report" in result
        synthesis = result["unified_report"]["synthesis"]
        
        # Should have default values
        assert synthesis["unified_metrics"]["acausal_cooperation_score"] == 0.0
        assert synthesis["acausal_cooperation_evidence"]["strength"] == "Weak"
        assert synthesis["acausal_cooperation_evidence"]["confidence"] == 0.0
        
        # Correlations should indicate insufficient data
        correlations = synthesis["correlations"]
        assert correlations["identity_cooperation"]["interpretation"] == "Insufficient data"
    
    @pytest.mark.asyncio
    async def test_executive_summary_generation(self, full_context):
        """Test executive summary generation."""
        node = ReportGeneratorNode()
        
        result = await node.execute(full_context)
        exec_summary = result["unified_report"]["executive_summary"]
        
        # Check all required sections
        assert "overview" in exec_summary
        assert "hypothesis_outcome" in exec_summary
        assert "statistical_highlights" in exec_summary
        assert "acausal_evidence" in exec_summary
        assert "conclusions" in exec_summary
        assert "key_findings" in exec_summary
        
        # Check hypothesis assessment
        hypothesis = exec_summary["hypothesis_outcome"]
        assert "strongly supports" in hypothesis["outcome"]
        assert hypothesis["evidence_components"]["acausal_score"] > 0.7
        
        # Check statistical highlights
        highlights = exec_summary["statistical_highlights"]
        assert len(highlights) >= 3
        assert any(h["metric"] == "Overall Cooperation Rate" for h in highlights)
        
        # Check conclusions
        conclusions = exec_summary["conclusions"]
        assert len(conclusions) >= 3
        assert any("Strong evidence" in c for c in conclusions)
    
    @pytest.mark.asyncio
    async def test_acausal_score_calculation(self, full_context):
        """Test acausal score calculation with different weights."""
        # Test with custom weights
        config = {
            "acausal_weights": {
                "identity_reasoning": 0.4,
                "cooperation_rate": 0.3,
                "strategy_convergence": 0.2,
                "cooperation_trend": 0.1
            }
        }
        node = ReportGeneratorNode(config=config)
        
        result = await node.execute(full_context)
        score = result["unified_report"]["synthesis"]["unified_metrics"]["acausal_cooperation_score"]
        
        # Calculate expected score
        expected = (
            0.4 * 0.73 +  # identity_reasoning
            0.3 * 0.72 +  # cooperation_rate
            0.2 * 0.81 +  # strategy_convergence
            0.1 * node._score_trend(full_context["statistical_analysis"]["cooperation_trend"])
        )
        
        assert abs(score - expected) < 0.001
    
    @pytest.mark.asyncio
    async def test_visualization_data_structure(self, full_context):
        """Test visualization data generation."""
        node = ReportGeneratorNode()
        
        result = await node.execute(full_context)
        visualizations = result["unified_report"]["visualizations"]
        
        # Check all visualizations are present
        expected_viz = [
            "cooperation_evolution",
            "strategy_clusters", 
            "power_dynamics",
            "correlation_matrix",
            "anomaly_timeline",
            "acausal_score_breakdown"
        ]
        for viz in expected_viz:
            assert viz in visualizations
        
        # Check cooperation evolution structure
        coop_evo = visualizations["cooperation_evolution"]
        assert coop_evo["type"] == "line_chart"
        assert "series" in coop_evo
        assert len(coop_evo["series"]) >= 1
        assert "confidence_intervals" in coop_evo["series"][0]
        
        # Check strategy clusters structure
        clusters = visualizations["strategy_clusters"]
        assert clusters["type"] == "scatter_plot"
        assert "frames" in clusters
        assert len(clusters["frames"]) >= 1
        
        # Check correlation matrix
        corr_matrix = visualizations["correlation_matrix"]
        assert corr_matrix["type"] == "correlation_matrix"
        assert len(corr_matrix["variables"]) == 5
        assert len(corr_matrix["matrix"]) == 5
    
    @pytest.mark.asyncio
    async def test_latex_generation_escaping(self, full_context):
        """Test LaTeX generation with special character escaping."""
        # Add some special characters to test
        full_context["config"]["model"] = "gpt_4_turbo"  # Has underscore
        
        node = ReportGeneratorNode()
        result = await node.execute(full_context)
        
        latex = result["unified_report"]["latex_sections"]
        
        # Check sections exist
        assert "methods" in latex["sections"]
        assert "results" in latex["sections"]
        assert "discussion" in latex["sections"]
        assert "tables" in latex["sections"]
        assert "figure_captions" in latex["sections"]
        
        # Check escaping in methods section
        methods = latex["sections"]["methods"]
        assert r"gpt\_4\_turbo" in methods
        
        # Check table generation
        tables = latex["sections"]["tables"]
        assert "cooperation_by_round" in tables
        assert "acausal_score_breakdown" in tables
        
        # Check full document
        assert "full_document" in latex
        assert r"\section{Methods}" in latex["full_document"]
    
    @pytest.mark.asyncio 
    async def test_markdown_formatting(self, full_context):
        """Test markdown report generation."""
        node = ReportGeneratorNode()
        
        result = await node.execute(full_context)
        markdown = result["unified_report"]["markdown_report"]
        
        # Check structure
        assert "# Acausal Cooperation Experiment Report" in markdown
        assert "## Executive Summary" in markdown
        assert "## Key Findings" in markdown
        assert "## Detailed Analysis" in markdown
        assert "## Statistical Results" in markdown
        assert "## Interpretation and Implications" in markdown
        assert "## Appendices" in markdown
        
        # Check formatting
        assert "**" in markdown  # Bold text
        assert "###" in markdown  # Subheadings
        assert "|" in markdown  # Tables
        assert "1." in markdown  # Numbered lists
        assert "-" in markdown  # Bullet points
        
        # Check data inclusion
        assert "73%" in markdown or "73.0%" in markdown  # Identity reasoning
        assert "72%" in markdown or "72.0%" in markdown  # Cooperation rate
        assert "round 6" in markdown  # Convergence round
    
    @pytest.mark.asyncio
    async def test_correlation_calculations(self, full_context):
        """Test correlation analysis calculations."""
        node = ReportGeneratorNode()
        
        # Test specific correlation methods
        transcript = full_context["transcript_analysis"]
        statistics = full_context["statistical_analysis"]
        similarity = full_context["similarity_analysis"]
        
        # Test identity-cooperation correlation
        id_coop = node._correlate_identity_cooperation(transcript, statistics)
        assert id_coop["correlation"] == "positive_strong"
        assert id_coop["identity_frequency"] == 0.73
        assert id_coop["cooperation_rate"] == 0.72
        
        # Test similarity-convergence correlation
        sim_conv = node._correlate_similarity_convergence(similarity, statistics)
        assert sim_conv["correlation"] == "positive_strong"
        assert sim_conv["convergence_round"] == 6
        
        # Test power-strategy correlation
        power_strat = node._correlate_power_strategy(transcript, statistics)
        assert power_strat["correlation"] == "negative_overcome"
        assert power_strat["asymmetry_recognition"] == 0.38
    
    @pytest.mark.asyncio
    async def test_report_file_generation(self, full_context, tmp_path):
        """Test that report files are saved correctly."""
        # Create real DataManager with temp directory
        dm = DataManager(base_path=str(tmp_path))
        dm.experiment_path = tmp_path / "test_exp"
        dm.experiment_path.mkdir(parents=True, exist_ok=True)
        full_context[ContextKeys.DATA_MANAGER] = dm
        
        node = ReportGeneratorNode()
        result = await node.execute(full_context)
        
        # Check files were created
        assert (dm.experiment_path / "unified_report.json").exists()
        assert (dm.experiment_path / "experiment_report.md").exists()
        assert (dm.experiment_path / "paper_sections.tex").exists()
        assert (dm.experiment_path / "visualization_data.json").exists()
        
        # Check JSON content
        with open(dm.experiment_path / "unified_report.json") as f:
            report_data = json.load(f)
            assert "synthesis" in report_data
            assert "executive_summary" in report_data
        
        # Check markdown content
        with open(dm.experiment_path / "experiment_report.md") as f:
            markdown = f.read()
            assert "# Acausal Cooperation Experiment Report" in markdown
        
        # Check LaTeX content
        with open(dm.experiment_path / "paper_sections.tex") as f:
            latex = f.read()
            assert r"\section{Methods}" in latex
    
    @pytest.mark.asyncio
    async def test_integration_with_experiment(self, full_context):
        """Test integration with full experiment context."""
        # Add round summaries to context
        full_context[ContextKeys.ROUND_SUMMARIES] = [
            Mock(round=i, cooperation_rate=0.6 + i*0.02) for i in range(1, 11)
        ]
        
        node = ReportGeneratorNode()
        result = await node.execute(full_context)
        
        # Should complete successfully
        assert "unified_report" in result
        assert result["unified_report"]["metadata"]["experiment_id"] == "test_exp_001"
        
        # Check all major components
        report = result["unified_report"]
        assert report["synthesis"]["acausal_cooperation_evidence"]["score"] > 0
        assert len(report["executive_summary"]["key_findings"]) > 0
        assert len(report["visualizations"]) == 6
        assert "full_document" in report["latex_sections"]
        assert len(report["markdown_report"]) > 1000  # Substantial content
    
    @pytest.mark.asyncio
    async def test_edge_case_no_cooperation(self, basic_context):
        """Test edge case with zero cooperation."""
        # Set up context with no cooperation
        basic_context["statistical_analysis"] = {
            "overall_cooperation_rate": 0.0,
            "cooperation_trend": {
                "direction": "stable",
                "slope": 0.0,
                "p_value": 0.8
            },
            "round_summaries": []
        }
        basic_context["transcript_analysis"] = {
            "marker_frequencies": {
                "identity_reasoning": 0.0
            }
        }
        basic_context["similarity_analysis"] = {
            "strategy_convergence": 0.0,
            "convergence_achieved": False
        }
        
        node = ReportGeneratorNode()
        result = await node.execute(basic_context)
        
        # Should handle gracefully
        synthesis = result["unified_report"]["synthesis"]
        assert synthesis["acausal_cooperation_evidence"]["strength"] == "Weak"
        assert synthesis["acausal_cooperation_evidence"]["score"] == 0.0
        
        # Check conclusions reflect no cooperation
        conclusions = result["unified_report"]["executive_summary"]["conclusions"]
        assert any("Limited evidence" in c for c in conclusions)
    
    @pytest.mark.asyncio
    async def test_edge_case_perfect_cooperation(self, basic_context):
        """Test edge case with perfect cooperation."""
        # Set up context with perfect cooperation
        basic_context["statistical_analysis"] = {
            "overall_cooperation_rate": 1.0,
            "cooperation_trend": {
                "direction": "increasing",
                "slope": 0.05,
                "p_value": 0.001
            },
            "round_summaries": [
                {"round": i, "cooperation_rate": 1.0, "mutual_cooperation_rate": 1.0, "total_games": 6}
                for i in range(1, 11)
            ],
            "agent_statistics": {
                str(i): {"cooperation_rate": 1.0, "avg_power": 1.0, "total_games": 30}
                for i in range(4)
            }
        }
        basic_context["transcript_analysis"] = {
            "marker_frequencies": {
                "identity_reasoning": 1.0,
                "logical_correlation": 1.0,
                "mutual_benefit": 1.0
            }
        }
        basic_context["similarity_analysis"] = {
            "strategy_convergence": 1.0,
            "convergence_achieved": True,
            "convergence_round": 1,
            "final_similarity": 1.0
        }
        
        node = ReportGeneratorNode()
        result = await node.execute(basic_context)
        
        # Should handle gracefully
        synthesis = result["unified_report"]["synthesis"]
        assert synthesis["acausal_cooperation_evidence"]["strength"] == "Strong"
        assert synthesis["acausal_cooperation_evidence"]["score"] > 0.9
        
        # Check perfect scores in metrics
        metrics = synthesis["unified_metrics"]
        assert metrics["overall_cooperation_rate"] == 1.0
        assert metrics["identity_reasoning_frequency"] == 1.0
        assert metrics["strategy_convergence"] == 1.0
    
    def test_section_toggle_configuration(self):
        """Test that report sections can be enabled/disabled via config."""
        config = {
            "enabled_sections": {
                "executive_summary": True,
                "detailed_findings": False,
                "visualizations": True,
                "latex_sections": False,
                "correlation_analysis": True
            }
        }
        
        node = ReportGeneratorNode(config=config)
        
        # Check configuration was applied
        assert node.enabled_sections["executive_summary"] is True
        assert node.enabled_sections["detailed_findings"] is False
        assert node.enabled_sections["latex_sections"] is False
    
    def test_weight_normalization(self):
        """Test that acausal weights are normalized to sum to 1.0."""
        config = {
            "acausal_weights": {
                "identity_reasoning": 1.0,
                "cooperation_rate": 1.0,
                "strategy_convergence": 1.0,
                "cooperation_trend": 1.0
            }
        }
        
        node = ReportGeneratorNode(config=config)
        
        # Weights should be normalized
        total = sum(node.acausal_weights.values())
        assert abs(total - 1.0) < 0.001
        
        # Each weight should be 0.25
        for weight in node.acausal_weights.values():
            assert abs(weight - 0.25) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])