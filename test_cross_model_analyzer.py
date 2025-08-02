"""Tests for CrossModelAnalyzer utility."""

import pytest
import numpy as np
import pandas as pd
from src.utils.cross_model_analyzer import CrossModelAnalyzer, CooperationStats


class TestCrossModelAnalyzer:
    """Test suite for CrossModelAnalyzer."""
    
    @pytest.fixture
    def sample_game_results(self):
        """Sample game results for testing."""
        return [
            {
                "game_id": "g1",
                "round": 1,
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE"
            },
            {
                "game_id": "g2",
                "round": 1,
                "player1_id": 3,
                "player2_id": 4,
                "player1_action": "DEFECT",
                "player2_action": "COOPERATE"
            },
            {
                "game_id": "g3",
                "round": 1,
                "player1_id": 1,
                "player2_id": 3,
                "player1_action": "COOPERATE",
                "player2_action": "DEFECT"
            },
            {
                "game_id": "g4",
                "round": 2,
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE"
            },
            {
                "game_id": "g5",
                "round": 2,
                "player1_id": 3,
                "player2_id": 4,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE"
            }
        ]
    
    @pytest.fixture
    def sample_strategy_records(self):
        """Sample strategy records with model information."""
        return [
            {"agent_id": 1, "model": "gpt-4", "round": 1},
            {"agent_id": 2, "model": "gpt-4", "round": 1},
            {"agent_id": 3, "model": "claude-3", "round": 1},
            {"agent_id": 4, "model": "claude-3", "round": 1},
            {"agent_id": 1, "model": "gpt-4", "round": 2},
            {"agent_id": 2, "model": "gpt-4", "round": 2},
            {"agent_id": 3, "model": "claude-3", "round": 2},
            {"agent_id": 4, "model": "claude-3", "round": 2}
        ]
    
    @pytest.fixture
    def analyzer_with_data(self, sample_game_results, sample_strategy_records):
        """Analyzer loaded with sample data."""
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(sample_game_results, sample_strategy_records)
        return analyzer
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = CrossModelAnalyzer()
        assert analyzer.game_results == []
        assert analyzer.strategy_records == []
        assert analyzer.model_map == {}
    
    def test_load_data(self, sample_game_results, sample_strategy_records):
        """Test data loading and model mapping."""
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(sample_game_results, sample_strategy_records)
        
        assert len(analyzer.game_results) == 5
        assert len(analyzer.strategy_records) == 8
        assert analyzer.model_map[1] == "gpt-4"
        assert analyzer.model_map[2] == "gpt-4"
        assert analyzer.model_map[3] == "claude-3"
        assert analyzer.model_map[4] == "claude-3"
    
    def test_get_model_for_agent(self, analyzer_with_data):
        """Test model lookup with error handling."""
        assert analyzer_with_data._get_model_for_agent(1) == "gpt-4"
        assert analyzer_with_data._get_model_for_agent(3) == "claude-3"
        assert analyzer_with_data._get_model_for_agent(99) == "unknown"
    
    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        analyzer = CrossModelAnalyzer()
        
        # Test normal case
        ci = analyzer._calculate_confidence_interval(7, 10)
        assert 0.35 < ci[0] < 0.45  # Lower bound
        assert 0.85 < ci[1] < 0.95  # Upper bound
        
        # Test edge cases
        ci_zero = analyzer._calculate_confidence_interval(0, 10)
        assert ci_zero[0] == 0.0
        
        ci_all = analyzer._calculate_confidence_interval(10, 10)
        assert ci_all[1] == 1.0
        
        # Test empty case
        ci_empty = analyzer._calculate_confidence_interval(0, 0)
        assert ci_empty == (0.0, 1.0)
    
    def test_calculate_cooperation_matrix(self, analyzer_with_data):
        """Test cooperation matrix calculation."""
        matrix = analyzer_with_data.calculate_cooperation_matrix()
        
        assert isinstance(matrix, pd.DataFrame)
        assert "gpt-4" in matrix.columns
        assert "claude-3" in matrix.columns
        assert "gpt-4" in matrix.index
        assert "claude-3" in matrix.index
        
        # Check specific values
        # gpt-4 vs gpt-4: 2 cooperations out of 2 games = 1.0
        assert matrix.loc["gpt-4", "gpt-4"] == 1.0
        
        # claude-3 vs claude-3: 1 cooperation out of 2 games = 0.5
        assert matrix.loc["claude-3", "claude-3"] == 0.5
        
        # gpt-4 vs claude-3: 1 cooperation out of 1 game = 1.0
        assert matrix.loc["gpt-4", "claude-3"] == 1.0
        
        # claude-3 vs gpt-4: 0 cooperations out of 1 game = 0.0
        assert matrix.loc["claude-3", "gpt-4"] == 0.0
    
    def test_cooperation_matrix_missing_data(self):
        """Test cooperation matrix with missing model data."""
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            [{"player1_id": 1, "player2_id": 2, "player1_action": "COOPERATE", "player2_action": "DEFECT"}],
            []  # No strategy records
        )
        
        matrix = analyzer.calculate_cooperation_matrix()
        # Should have "unknown" model
        assert "unknown" in matrix.columns
        assert matrix.loc["unknown", "unknown"] == 0.5  # 1 cooperation out of 2 actions
    
    def test_get_cooperation_stats(self, analyzer_with_data):
        """Test detailed cooperation statistics."""
        stats = analyzer_with_data.get_cooperation_stats()
        
        assert "gpt-4" in stats
        assert "claude-3" in stats
        
        # Check gpt-4 vs gpt-4 stats
        gpt_self_stats = stats["gpt-4"]["gpt-4"]
        assert isinstance(gpt_self_stats, CooperationStats)
        assert gpt_self_stats.cooperation_count == 2
        assert gpt_self_stats.total_games == 2
        assert gpt_self_stats.cooperation_rate == 1.0
        assert len(gpt_self_stats.confidence_interval) == 2
        assert gpt_self_stats.sample_size == 2
    
    def test_detect_in_group_bias(self, analyzer_with_data):
        """Test in-group bias detection."""
        bias_result = analyzer_with_data.detect_in_group_bias()
        
        assert "same_model_rate" in bias_result
        assert "cross_model_rate" in bias_result
        assert "bias" in bias_result
        assert "p_value" in bias_result
        assert "effect_size" in bias_result
        assert "confidence_interval" in bias_result
        assert "sample_sizes" in bias_result
        
        # In this sample, same-model cooperation is higher
        assert bias_result["same_model_rate"] > bias_result["cross_model_rate"]
        assert bias_result["bias"] > 0
        assert bias_result["sample_sizes"]["same_model"] == 4  # 2 gpt-4 + 2 claude-3 games
        assert bias_result["sample_sizes"]["cross_model"] == 2  # 2 cross-model games
    
    def test_detect_in_group_bias_no_data(self):
        """Test in-group bias with no games."""
        analyzer = CrossModelAnalyzer()
        analyzer.load_data([], [])
        
        bias_result = analyzer.detect_in_group_bias()
        assert bias_result["same_model_rate"] is None
        assert bias_result["cross_model_rate"] is None
        assert bias_result["bias"] is None
        assert bias_result["p_value"] is None
    
    def test_detect_in_group_bias_single_model(self):
        """Test in-group bias with only one model type."""
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            [
                {"player1_id": 1, "player2_id": 2, "player1_action": "COOPERATE", "player2_action": "COOPERATE", "round": 1},
                {"player1_id": 1, "player2_id": 2, "player1_action": "DEFECT", "player2_action": "COOPERATE", "round": 1}
            ],
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"}
            ]
        )
        
        bias_result = analyzer.detect_in_group_bias()
        # Should have same_model data but no cross_model data
        assert bias_result["same_model_rate"] is not None
        assert bias_result["cross_model_rate"] is None
        assert bias_result["bias"] is None
    
    def test_analyze_model_coalitions(self, analyzer_with_data):
        """Test coalition detection."""
        coalition_result = analyzer_with_data.analyze_model_coalitions()
        
        assert "detected" in coalition_result
        assert "coalition_groups" in coalition_result
        assert "all_pairings" in coalition_result
        assert "strongest_pair" in coalition_result
        assert "weakest_pair" in coalition_result
        assert "total_rounds_analyzed" in coalition_result
        
        assert coalition_result["total_rounds_analyzed"] == 2
        
        # Check that pairings are sorted by strength
        pairings = coalition_result["all_pairings"]
        if len(pairings) > 1:
            strengths = [p["strength"] for p in pairings]
            assert strengths == sorted(strengths, reverse=True)
    
    def test_analyze_model_coalitions_insufficient_rounds(self):
        """Test coalition detection with too few rounds."""
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            [{"player1_id": 1, "player2_id": 2, "player1_action": "COOPERATE", "player2_action": "COOPERATE", "round": 1}],
            [{"agent_id": 1, "model": "gpt-4"}, {"agent_id": 2, "model": "gpt-4"}]
        )
        
        coalition_result = analyzer.analyze_model_coalitions()
        assert coalition_result["detected"] is False
        assert len(coalition_result["coalition_groups"]) == 0
    
    def test_analyze_model_coalitions_consistency(self):
        """Test coalition consistency calculation."""
        # Create highly consistent cooperation pattern
        games = []
        for round_num in range(5):
            games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE",
                "round": round_num
            })
            games.append({
                "player1_id": 3,
                "player2_id": 4,
                "player1_action": "DEFECT",
                "player2_action": "DEFECT",
                "round": round_num
            })
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"}
            ]
        )
        
        coalition_result = analyzer.analyze_model_coalitions()
        
        # gpt-4 pair should have high strength (high cooperation + high consistency)
        gpt_coalition = next(p for p in coalition_result["all_pairings"] if "gpt-4" in p["models"])
        assert gpt_coalition["average_cooperation"] == 1.0
        assert gpt_coalition["consistency"] == 1.0
        assert gpt_coalition["strength"] == 1.0
        
        # claude-3 pair should have low strength (no cooperation)
        claude_coalition = next(p for p in coalition_result["all_pairings"] if "claude-3" in p["models"])
        assert claude_coalition["average_cooperation"] == 0.0
    
    def test_cooperation_matrix_statistical_significance(self):
        """Test that cooperation matrix handles statistical calculations."""
        # Create dataset with clear patterns
        games = []
        # GPT-4 always cooperates with itself
        for i in range(20):
            games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE",
                "round": i // 5
            })
        
        # Claude-3 sometimes cooperates with itself
        for i in range(20):
            action = "COOPERATE" if i % 3 == 0 else "DEFECT"
            games.append({
                "player1_id": 3,
                "player2_id": 4,
                "player1_action": action,
                "player2_action": action,
                "round": i // 5
            })
        
        # Mixed model interactions
        for i in range(10):
            games.append({
                "player1_id": 1,
                "player2_id": 3,
                "player1_action": "COOPERATE",
                "player2_action": "DEFECT",
                "round": i // 3
            })
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"}
            ]
        )
        
        matrix = analyzer.calculate_cooperation_matrix()
        stats = analyzer.get_cooperation_stats()
        
        # Verify matrix values
        assert matrix.loc["gpt-4", "gpt-4"] == 1.0
        assert 0.3 < matrix.loc["claude-3", "claude-3"] < 0.4
        assert matrix.loc["gpt-4", "claude-3"] == 1.0
        assert matrix.loc["claude-3", "gpt-4"] == 0.0
        
        # Check confidence intervals
        gpt_stats = stats["gpt-4"]["gpt-4"]
        assert gpt_stats.confidence_interval[0] > 0.8  # High confidence it's near 1.0
        
        claude_stats = stats["claude-3"]["claude-3"]
        assert claude_stats.confidence_interval[1] < 0.6  # Confidence it's below 0.6
    
    def test_cooperation_matrix_chi_square(self):
        """Test chi-square calculation for cooperation differences."""
        analyzer = CrossModelAnalyzer()
        
        # Create data with significant difference
        games = []
        # Model A cooperates 90% of the time
        for i in range(100):
            action = "COOPERATE" if i < 90 else "DEFECT"
            games.append({
                "player1_id": 1,
                "player2_id": 1,
                "player1_action": action,
                "player2_action": action,
                "round": i // 10
            })
        
        # Model B cooperates 10% of the time  
        for i in range(100):
            action = "COOPERATE" if i < 10 else "DEFECT"
            games.append({
                "player1_id": 2,
                "player2_id": 2,
                "player1_action": action,
                "player2_action": action,
                "round": i // 10
            })
        
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "model_a"},
                {"agent_id": 2, "model": "model_b"}
            ]
        )
        
        stats = analyzer.get_cooperation_stats()
        
        # Both models should have narrow confidence intervals due to large sample
        model_a_stats = stats["model_a"]["model_a"]
        model_b_stats = stats["model_b"]["model_b"]
        
        ci_width_a = model_a_stats.confidence_interval[1] - model_a_stats.confidence_interval[0]
        ci_width_b = model_b_stats.confidence_interval[1] - model_b_stats.confidence_interval[0]
        
        assert ci_width_a < 0.1  # Narrow CI
        assert ci_width_b < 0.1  # Narrow CI
        
        # The confidence intervals should not overlap
        assert model_a_stats.confidence_interval[0] > model_b_stats.confidence_interval[1]
    
    def test_in_group_bias_edge_cases(self):
        """Test in-group bias detection with edge cases."""
        analyzer = CrossModelAnalyzer()
        
        # Edge case 1: Only one game total
        analyzer.load_data(
            [{"player1_id": 1, "player2_id": 2, "player1_action": "COOPERATE", "player2_action": "COOPERATE", "round": 1}],
            [{"agent_id": 1, "model": "gpt-4"}, {"agent_id": 2, "model": "gpt-4"}]
        )
        
        bias_result = analyzer.detect_in_group_bias()
        assert bias_result["same_model_rate"] == 1.0
        assert bias_result["cross_model_rate"] is None  # No cross-model games
        
        # Edge case 2: All defections
        analyzer.load_data(
            [
                {"player1_id": 1, "player2_id": 2, "player1_action": "DEFECT", "player2_action": "DEFECT", "round": 1},
                {"player1_id": 3, "player2_id": 4, "player1_action": "DEFECT", "player2_action": "DEFECT", "round": 1},
                {"player1_id": 1, "player2_id": 3, "player1_action": "DEFECT", "player2_action": "DEFECT", "round": 1}
            ],
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"}
            ]
        )
        
        bias_result = analyzer.detect_in_group_bias()
        assert bias_result["same_model_rate"] == 0.0
        assert bias_result["cross_model_rate"] == 0.0
        assert bias_result["bias"] == 0.0
    
    def test_in_group_bias_statistical_power(self):
        """Test that bias detection properly calculates statistical power."""
        analyzer = CrossModelAnalyzer()
        
        # Small sample - low power
        small_games = [
            {"player1_id": 1, "player2_id": 2, "player1_action": "COOPERATE", "player2_action": "COOPERATE", "round": 1},
            {"player1_id": 1, "player2_id": 3, "player1_action": "DEFECT", "player2_action": "COOPERATE", "round": 1}
        ]
        
        analyzer.load_data(
            small_games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"}
            ]
        )
        
        small_bias = analyzer.detect_in_group_bias()
        
        # Large sample - high power
        large_games = []
        # 100 same-model games with 80% cooperation
        for i in range(100):
            action = "COOPERATE" if i < 80 else "DEFECT"
            large_games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": action,
                "player2_action": action,
                "round": i // 10
            })
        
        # 100 cross-model games with 20% cooperation
        for i in range(100):
            action = "COOPERATE" if i < 20 else "DEFECT"
            large_games.append({
                "player1_id": 1,
                "player2_id": 3,
                "player1_action": action,
                "player2_action": "DEFECT",
                "round": i // 10
            })
        
        analyzer.load_data(
            large_games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"}
            ]
        )
        
        large_bias = analyzer.detect_in_group_bias()
        
        # Large sample should have much smaller p-value (more significant)
        assert large_bias["p_value"] < small_bias["p_value"]
        
        # Large sample should have narrower confidence interval
        small_ci_width = small_bias["confidence_interval"][1] - small_bias["confidence_interval"][0]
        large_ci_width = large_bias["confidence_interval"][1] - large_bias["confidence_interval"][0]
        assert large_ci_width < small_ci_width
    
    def test_in_group_bias_three_models(self):
        """Test in-group bias with three different model types."""
        games = []
        
        # GPT-4 self games - high cooperation
        for i in range(10):
            games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE",
                "round": i
            })
        
        # Claude-3 self games - medium cooperation
        for i in range(10):
            action = "COOPERATE" if i < 5 else "DEFECT"
            games.append({
                "player1_id": 3,
                "player2_id": 4,
                "player1_action": action,
                "player2_action": action,
                "round": i
            })
        
        # Gemini self games - low cooperation
        for i in range(10):
            action = "COOPERATE" if i < 2 else "DEFECT"
            games.append({
                "player1_id": 5,
                "player2_id": 6,
                "player1_action": action,
                "player2_action": action,
                "round": i
            })
        
        # Cross-model games - very low cooperation
        models = [(1, 3), (1, 5), (3, 5)]
        for p1, p2 in models:
            for i in range(5):
                games.append({
                    "player1_id": p1,
                    "player2_id": p2,
                    "player1_action": "DEFECT",
                    "player2_action": "DEFECT",
                    "round": i
                })
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"},
                {"agent_id": 5, "model": "gemini"},
                {"agent_id": 6, "model": "gemini"}
            ]
        )
        
        bias_result = analyzer.detect_in_group_bias()
        
        # Should detect significant in-group bias
        assert bias_result["bias"] > 0.3  # Substantial bias
        assert bias_result["same_model_rate"] > bias_result["cross_model_rate"]
        assert bias_result["p_value"] < 0.05  # Statistically significant
    
    def test_model_coalition_persistence(self):
        """Test detection of persistent model coalitions across rounds."""
        games = []
        
        # Create persistent coalition between GPT-4 agents (always cooperate)
        for round_num in range(10):
            games.append({
                "game_id": f"g{round_num}_1",
                "round": round_num,
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE"
            })
        
        # Create less consistent coalition between Claude agents
        for round_num in range(10):
            action = "COOPERATE" if round_num % 2 == 0 else "DEFECT"
            games.append({
                "game_id": f"g{round_num}_2",
                "round": round_num,
                "player1_id": 3,
                "player2_id": 4,
                "player1_action": action,
                "player2_action": action
            })
        
        # Cross-model interactions are mostly defections
        for round_num in range(10):
            games.append({
                "game_id": f"g{round_num}_3",
                "round": round_num,
                "player1_id": 1,
                "player2_id": 3,
                "player1_action": "DEFECT",
                "player2_action": "DEFECT"
            })
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"}
            ]
        )
        
        coalition_result = analyzer.analyze_model_coalitions()
        
        # Should detect coalition
        assert coalition_result["detected"] is True
        assert len(coalition_result["coalition_groups"]) >= 1
        
        # GPT-4 coalition should be strongest
        strongest = coalition_result["strongest_pair"]
        assert "gpt-4" in strongest
        
        # Check coalition strength calculation
        gpt_coalition = next(
            c for c in coalition_result["all_pairings"] 
            if set(c["models"]) == {"gpt-4"}
        )
        assert gpt_coalition["average_cooperation"] == 1.0
        assert gpt_coalition["consistency"] == 1.0
        assert gpt_coalition["strength"] == 1.0
    
    def test_coalition_network_data(self):
        """Test coalition network data generation for visualization."""
        # Create complex coalition patterns
        games = []
        
        # Strong coalition: GPT-4 models
        for r in range(5):
            games.extend([
                {"game_id": f"g{r}_1", "round": r, "player1_id": 1, "player2_id": 2,
                 "player1_action": "COOPERATE", "player2_action": "COOPERATE"},
                {"game_id": f"g{r}_2", "round": r, "player1_id": 3, "player2_id": 4,
                 "player1_action": "COOPERATE", "player2_action": "COOPERATE"}
            ])
        
        # Weak coalition: Claude models  
        for r in range(5):
            action = "COOPERATE" if r < 2 else "DEFECT"
            games.append({
                "game_id": f"g{r}_3", "round": r, "player1_id": 5, "player2_id": 6,
                "player1_action": action, "player2_action": action
            })
        
        # Mixed interactions
        for r in range(5):
            games.extend([
                {"game_id": f"g{r}_4", "round": r, "player1_id": 1, "player2_id": 5,
                 "player1_action": "COOPERATE", "player2_action": "DEFECT"},
                {"game_id": f"g{r}_5", "round": r, "player1_id": 3, "player2_id": 6,
                 "player1_action": "DEFECT", "player2_action": "COOPERATE"}
            ])
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4-1"},
                {"agent_id": 2, "model": "gpt-4-1"},
                {"agent_id": 3, "model": "gpt-4-2"},
                {"agent_id": 4, "model": "gpt-4-2"},
                {"agent_id": 5, "model": "claude-3"},
                {"agent_id": 6, "model": "claude-3"}
            ]
        )
        
        result = analyzer.analyze_model_coalitions()
        
        # Check all pairings are analyzed
        assert len(result["all_pairings"]) >= 3
        
        # Check network data structure
        for pairing in result["all_pairings"]:
            assert "models" in pairing
            assert len(pairing["models"]) == 2
            assert "average_cooperation" in pairing
            assert "consistency" in pairing
            assert "strength" in pairing
            assert 0 <= pairing["strength"] <= 1
    
    def test_coalition_temporal_evolution(self):
        """Test tracking coalition evolution over time."""
        games = []
        
        # Coalition starts weak and grows stronger
        for round_num in range(10):
            # Early rounds: occasional cooperation
            if round_num < 3:
                action = "COOPERATE" if round_num == 0 else "DEFECT"
            # Middle rounds: increasing cooperation
            elif round_num < 7:
                action = "COOPERATE" if round_num % 2 == 0 else "DEFECT"
            # Late rounds: consistent cooperation
            else:
                action = "COOPERATE"
                
            games.append({
                "game_id": f"g{round_num}",
                "round": round_num,
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": action,
                "player2_action": action
            })
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"}
            ]
        )
        
        result = analyzer.analyze_model_coalitions()
        
        # Should detect the coalition
        assert len(result["all_pairings"]) > 0
        
        # Average cooperation should reflect the overall pattern
        coalition = result["all_pairings"][0]
        assert 0.5 < coalition["average_cooperation"] < 0.8
        
        # Consistency should be lower due to changing pattern
        assert coalition["consistency"] < 0.8
    
    def test_coalition_edge_cases(self):
        """Test coalition detection edge cases."""
        # Edge case 1: Single round
        analyzer1 = CrossModelAnalyzer()
        analyzer1.load_data(
            [{"game_id": "g1", "round": 1, "player1_id": 1, "player2_id": 2,
              "player1_action": "COOPERATE", "player2_action": "COOPERATE"}],
            [{"agent_id": 1, "model": "gpt-4"}, {"agent_id": 2, "model": "gpt-4"}]
        )
        
        result1 = analyzer1.analyze_model_coalitions()
        assert result1["detected"] is False  # Not enough rounds
        
        # Edge case 2: No mutual cooperation
        games2 = []
        for r in range(5):
            games2.append({
                "game_id": f"g{r}", "round": r, "player1_id": 1, "player2_id": 2,
                "player1_action": "COOPERATE", "player2_action": "DEFECT"
            })
        
        analyzer2 = CrossModelAnalyzer()
        analyzer2.load_data(
            games2,
            [{"agent_id": 1, "model": "gpt-4"}, {"agent_id": 2, "model": "claude-3"}]
        )
        
        result2 = analyzer2.analyze_model_coalitions()
        # Should still analyze but show low cooperation
        assert result2["all_pairings"][0]["average_cooperation"] == 0.0
    
    def test_generate_heatmap_data(self, analyzer_with_data):
        """Test heatmap visualization data generation."""
        heatmap_data = analyzer_with_data.generate_heatmap_data()
        
        assert "heatmap" in heatmap_data
        assert "time_series" in heatmap_data
        assert "metadata" in heatmap_data
        
        # Check heatmap structure
        heatmap = heatmap_data["heatmap"]
        assert "matrix" in heatmap
        assert "labels" in heatmap
        assert "significance" in heatmap
        assert "color_scale" in heatmap
        assert "title" in heatmap
        assert "x_label" in heatmap
        assert "y_label" in heatmap
        
        # Check matrix dimensions
        matrix = heatmap["matrix"]
        labels = heatmap["labels"]
        assert len(matrix) == len(labels)
        assert all(len(row) == len(labels) for row in matrix)
        
        # Check color scale
        color_scale = heatmap["color_scale"]
        assert color_scale["min"] == 0.0
        assert color_scale["max"] == 1.0
        assert color_scale["midpoint"] == 0.5
        assert "colormap" in color_scale
        assert "null_color" in color_scale
        
        # Check metadata
        metadata = heatmap_data["metadata"]
        assert "total_games" in metadata
        assert "total_models" in metadata
        assert "generated_at" in metadata
    
    def test_generate_heatmap_data_with_missing_data(self):
        """Test heatmap generation with missing model pairings."""
        analyzer = CrossModelAnalyzer()
        # Create data where some model pairings never play
        games = [
            {"player1_id": 1, "player2_id": 2, "player1_action": "COOPERATE", "player2_action": "COOPERATE", "round": 1},
            {"player1_id": 3, "player2_id": 4, "player1_action": "DEFECT", "player2_action": "COOPERATE", "round": 1},
            # No games between gpt-4 and gemini
        ]
        
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"},
                {"agent_id": 5, "model": "gemini"},  # Never plays
                {"agent_id": 6, "model": "gemini"}
            ]
        )
        
        heatmap_data = analyzer.generate_heatmap_data()
        matrix = heatmap_data["heatmap"]["matrix"]
        labels = heatmap_data["heatmap"]["labels"]
        
        # Should have all models in labels
        assert "gpt-4" in labels
        assert "claude-3" in labels
        
        # Missing data should be None
        gpt_idx = labels.index("gpt-4")
        claude_idx = labels.index("claude-3")
        
        # Check known values
        assert matrix[gpt_idx][gpt_idx] == 1.0
        assert matrix[claude_idx][claude_idx] == 0.5
    
    def test_time_series_data_generation(self):
        """Test time series visualization data generation."""
        # Create games with evolving cooperation patterns
        games = []
        for round_num in range(5):
            # GPT-4 self: starts defecting, ends cooperating
            action = "COOPERATE" if round_num >= 3 else "DEFECT"
            games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": action,
                "player2_action": action,
                "round": round_num
            })
            
            # Claude self: consistent cooperation
            games.append({
                "player1_id": 3,
                "player2_id": 4,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE",
                "round": round_num
            })
            
            # Cross-model: always defect
            games.append({
                "player1_id": 1,
                "player2_id": 3,
                "player1_action": "DEFECT",
                "player2_action": "DEFECT",
                "round": round_num
            })
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"}
            ]
        )
        
        heatmap_data = analyzer.generate_heatmap_data()
        time_series = heatmap_data["time_series"]
        
        assert "series" in time_series
        assert "x_label" in time_series
        assert "y_label" in time_series
        assert "title" in time_series
        assert "y_range" in time_series
        assert "show_sample_sizes" in time_series
        
        # Check series data
        series = time_series["series"]
        assert len(series) > 0
        
        for s in series:
            assert "name" in s
            assert "data" in s
            assert "total_games" in s
            
            data = s["data"]
            assert "rounds" in data
            assert "cooperation_rates" in data
            assert "sample_sizes" in data
            
            # Check data consistency
            assert len(data["rounds"]) == len(data["cooperation_rates"])
            assert len(data["rounds"]) == len(data["sample_sizes"])
        
        # Check y_range
        assert time_series["y_range"] == [0, 1]
    
    def test_time_series_filtering(self):
        """Test that time series only includes pairs with sufficient data."""
        analyzer = CrossModelAnalyzer()
        
        # Create sparse data
        games = []
        # Pair 1: Many games (should be included)
        for r in range(10):
            games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE",
                "round": r
            })
        
        # Pair 2: Few games (should be excluded)
        games.append({
            "player1_id": 3,
            "player2_id": 4,
            "player1_action": "DEFECT",
            "player2_action": "DEFECT",
            "round": 1
        })
        
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"}
            ]
        )
        
        heatmap_data = analyzer.generate_heatmap_data()
        series = heatmap_data["time_series"]["series"]
        
        # Should only include pairs with 5+ games
        assert all(s["total_games"] >= 5 for s in series)
        
        # Should include GPT-4 pair
        gpt_series = [s for s in series if "gpt-4" in s["name"]]
        assert len(gpt_series) > 0
    
    def test_heatmap_significance_calculation(self):
        """Test significance values in heatmap data."""
        # Create data with different sample sizes
        games = []
        
        # High sample size for GPT-4 (narrow CI = high significance)
        for i in range(100):
            games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE",
                "round": i // 10
            })
        
        # Low sample size for Claude-3 (wide CI = low significance)
        games.append({
            "player1_id": 3,
            "player2_id": 4,
            "player1_action": "COOPERATE",
            "player2_action": "DEFECT",
            "round": 1
        })
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"}
            ]
        )
        
        heatmap_data = analyzer.generate_heatmap_data()
        significance = heatmap_data["heatmap"]["significance"]
        labels = heatmap_data["heatmap"]["labels"]
        
        gpt_idx = labels.index("gpt-4")
        claude_idx = labels.index("claude-3")
        
        # GPT-4 self should have high significance
        assert significance[gpt_idx][gpt_idx] > 0.8
        
        # Claude-3 self should have low significance
        assert significance[claude_idx][claude_idx] < 0.5 or significance[claude_idx][claude_idx] is None
    
    def test_visualization_edge_cases(self):
        """Test visualization generation with edge cases."""
        # Edge case 1: No games
        analyzer1 = CrossModelAnalyzer()
        analyzer1.load_data([], [])
        
        heatmap_data1 = analyzer1.generate_heatmap_data()
        assert heatmap_data1["heatmap"]["matrix"] == []
        assert heatmap_data1["heatmap"]["labels"] == []
        assert heatmap_data1["metadata"]["total_games"] == 0
        
        # Edge case 2: Single model type
        analyzer2 = CrossModelAnalyzer()
        analyzer2.load_data(
            [{"player1_id": 1, "player2_id": 2, "player1_action": "COOPERATE", "player2_action": "COOPERATE", "round": 1}],
            [{"agent_id": 1, "model": "gpt-4"}, {"agent_id": 2, "model": "gpt-4"}]
        )
        
        heatmap_data2 = analyzer2.generate_heatmap_data()
        assert len(heatmap_data2["heatmap"]["labels"]) == 1
        assert heatmap_data2["heatmap"]["labels"][0] == "gpt-4"
        assert len(heatmap_data2["heatmap"]["matrix"]) == 1
        assert len(heatmap_data2["heatmap"]["matrix"][0]) == 1
    
    def test_time_series_ordering(self):
        """Test that time series data is properly ordered by round."""
        games = []
        # Add games in non-sequential order
        for round_num in [5, 1, 3, 2, 4]:
            games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE",
                "round": round_num
            })
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [{"agent_id": 1, "model": "gpt-4"}, {"agent_id": 2, "model": "gpt-4"}]
        )
        
        heatmap_data = analyzer.generate_heatmap_data()
        series = heatmap_data["time_series"]["series"][0]
        rounds = series["data"]["rounds"]
        
        # Rounds should be in ascending order
        assert rounds == sorted(rounds)
        assert rounds == [1, 2, 3, 4, 5]
    
    def test_calculate_average_cooperation_by_model(self, analyzer_with_data):
        """Test calculation of average cooperation rates by model."""
        avg_cooperation = analyzer_with_data.calculate_average_cooperation_by_model()
        
        assert "gpt-4" in avg_cooperation
        assert "claude-3" in avg_cooperation
        
        # Check structure
        for model, stats in avg_cooperation.items():
            assert "avg_cooperation" in stats
            assert "total_games" in stats
            assert "cooperation_count" in stats
            assert "confidence_interval" in stats
            assert 0 <= stats["avg_cooperation"] <= 1
    
    def test_compute_model_diversity_impact(self, analyzer_with_data):
        """Test model diversity impact computation."""
        diversity_impact = analyzer_with_data.compute_model_diversity_impact()
        
        assert "diversity_ratio" in diversity_impact
        assert "same_model_mutual_cooperation" in diversity_impact
        assert "cross_model_mutual_cooperation" in diversity_impact
        assert "cooperation_difference" in diversity_impact
        assert "diversity_cost" in diversity_impact
        assert "in_group_bias" in diversity_impact
        assert "statistical_significance" in diversity_impact
        assert "interpretation" in diversity_impact
        
        # Check values are in expected ranges
        assert 0 <= diversity_impact["diversity_ratio"] <= 1
        assert 0 <= diversity_impact["same_model_mutual_cooperation"] <= 1
        assert 0 <= diversity_impact["cross_model_mutual_cooperation"] <= 1
    
    def test_diversity_impact_interpretation(self):
        """Test interpretation of diversity impact."""
        analyzer = CrossModelAnalyzer()
        
        # No significant impact
        assert "No significant" in analyzer._interpret_diversity_impact(0.1, 0.1)
        
        # Minimal impact
        assert "minimal" in analyzer._interpret_diversity_impact(0.03, 0.01)
        
        # Moderate impact
        assert "moderately" in analyzer._interpret_diversity_impact(0.10, 0.01)
        
        # Substantial impact
        assert "substantially" in analyzer._interpret_diversity_impact(0.20, 0.01)
    
    def test_get_sample_size_warnings(self):
        """Test sample size warning generation."""
        analyzer = CrossModelAnalyzer()
        
        # Create data with varying sample sizes
        games = []
        # Many games for gpt-4 self
        for i in range(100):
            games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE",
                "round": i // 10
            })
        
        # Few games for claude-3 self
        games.append({
            "player1_id": 3,
            "player2_id": 4,
            "player1_action": "DEFECT",
            "player2_action": "DEFECT",
            "round": 1
        })
        
        # No games for gemini
        
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"},
                {"agent_id": 5, "model": "gemini"},
                {"agent_id": 6, "model": "gemini"}
            ]
        )
        
        warnings = analyzer.get_sample_size_warnings()
        
        assert "critical" in warnings
        assert "low_confidence" in warnings
        assert "info" in warnings
        
        # Should have critical warning for claude-3 (very low sample)
        assert any("claude-3" in w and "Very low" in w for w in warnings["critical"])
    
    def test_track_coalition_emergence(self):
        """Test tracking of coalition emergence over rounds."""
        # Create data with emerging coalitions
        games = []
        
        # Rounds 1-2: No clear coalitions
        for round_num in [1, 2]:
            games.extend([
                {"player1_id": 1, "player2_id": 2, "player1_action": "DEFECT", "player2_action": "COOPERATE", "round": round_num},
                {"player1_id": 3, "player2_id": 4, "player1_action": "COOPERATE", "player2_action": "DEFECT", "round": round_num}
            ])
        
        # Rounds 3-5: Strong coalitions emerge
        for round_num in [3, 4, 5]:
            games.extend([
                {"player1_id": 1, "player2_id": 2, "player1_action": "COOPERATE", "player2_action": "COOPERATE", "round": round_num},
                {"player1_id": 3, "player2_id": 4, "player1_action": "COOPERATE", "player2_action": "COOPERATE", "round": round_num}
            ])
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"}
            ]
        )
        
        emergence = analyzer.track_coalition_emergence()
        
        assert "round_by_round" in emergence
        assert "stable_coalitions" in emergence
        assert "coalition_formation_round" in emergence
        assert "total_rounds_analyzed" in emergence
        
        # Should detect coalition formation around round 3
        assert emergence["coalition_formation_round"] == 3
        
        # Should have stable coalitions
        assert len(emergence["stable_coalitions"]) > 0
    
    def test_calculate_statistical_power(self):
        """Test statistical power calculation."""
        analyzer = CrossModelAnalyzer()
        
        # Create data with different sample sizes
        games = []
        
        # High sample size for gpt-4 self (high power)
        for i in range(100):
            games.append({
                "player1_id": 1,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE",
                "round": i // 10
            })
        
        # Low sample size for claude-3 self (low power)
        for i in range(3):
            games.append({
                "player1_id": 3,
                "player2_id": 4,
                "player1_action": "DEFECT",
                "player2_action": "COOPERATE",
                "round": 1
            })
        
        analyzer.load_data(
            games,
            [
                {"agent_id": 1, "model": "gpt-4"},
                {"agent_id": 2, "model": "gpt-4"},
                {"agent_id": 3, "model": "claude-3"},
                {"agent_id": 4, "model": "claude-3"}
            ]
        )
        
        power_analysis = analyzer.calculate_statistical_power()
        
        # Check structure
        for pair, analysis in power_analysis.items():
            assert "statistical_power" in analysis
            assert "sample_size" in analysis
            assert "effect_size" in analysis
            assert "interpretation" in analysis
            assert 0 <= analysis["statistical_power"] <= 1
        
        # GPT-4 self should have high power
        gpt_power = power_analysis.get("gpt-4 vs gpt-4", {})
        if gpt_power:
            assert gpt_power["statistical_power"] > 0.8
            assert "Excellent" in gpt_power["interpretation"]
        
        # Claude-3 self should have low power
        claude_power = power_analysis.get("claude-3 vs claude-3", {})
        if claude_power:
            assert claude_power["statistical_power"] < 0.5
    
    def test_comprehensive_statistics_integration(self):
        """Test that all comprehensive statistics work together."""
        # Create realistic multi-round data
        games = []
        strategies = []
        
        models = ["gpt-4", "claude-3", "gemini"]
        agent_id = 0
        
        # Create agents
        agents = {}
        for model in models:
            for i in range(2):
                agents[agent_id] = model
                strategies.append({"agent_id": agent_id, "model": model})
                agent_id += 1
        
        # Create games across rounds
        for round_num in range(5):
            for p1 in range(len(agents)):
                for p2 in range(p1 + 1, len(agents)):
                    # Same model pairs cooperate more
                    if agents[p1] == agents[p2]:
                        action = "COOPERATE" if round_num > 1 else "DEFECT"
                    else:
                        action = "DEFECT" if round_num < 3 else "COOPERATE"
                    
                    games.append({
                        "player1_id": p1,
                        "player2_id": p2,
                        "player1_action": action,
                        "player2_action": action,
                        "round": round_num
                    })
        
        analyzer = CrossModelAnalyzer()
        analyzer.load_data(games, strategies)
        
        # Run all analyses
        matrix = analyzer.calculate_cooperation_matrix()
        bias = analyzer.detect_in_group_bias()
        avg_by_model = analyzer.calculate_average_cooperation_by_model()
        diversity = analyzer.compute_model_diversity_impact()
        warnings = analyzer.get_sample_size_warnings()
        emergence = analyzer.track_coalition_emergence()
        power = analyzer.calculate_statistical_power()
        
        # Verify all return valid data
        assert not matrix.empty
        assert bias["same_model_rate"] is not None
        assert len(avg_by_model) == 3  # Three models
        assert diversity["diversity_ratio"] > 0
        assert isinstance(warnings, dict)
        assert emergence["total_rounds_analyzed"] == 5
        assert len(power) > 0
    
    def test_edge_cases_comprehensive_stats(self):
        """Test edge cases for comprehensive statistics."""
        analyzer = CrossModelAnalyzer()
        
        # Empty data
        analyzer.load_data([], [])
        
        # All methods should handle empty data gracefully
        assert analyzer.calculate_average_cooperation_by_model() == {}
        
        diversity = analyzer.compute_model_diversity_impact()
        assert diversity["diversity_ratio"] == 0
        
        warnings = analyzer.get_sample_size_warnings()
        assert len(warnings["critical"]) == 0
        
        emergence = analyzer.track_coalition_emergence()
        assert emergence["stable_coalitions"] == []
        
        power = analyzer.calculate_statistical_power()
        assert power == {}