"""Tests for enhanced coalition tracking in mixed scenarios."""
import pytest
import numpy as np
from unittest.mock import Mock

from src.utils.coalition_tracker import TemporalCoalitionTracker, CoalitionMetrics
from src.utils.cross_model_analyzer import CrossModelAnalyzer


class TestTemporalCoalitionTracker:
    """Test suite for TemporalCoalitionTracker."""
    
    def test_coalition_formation(self):
        """Test basic coalition formation detection."""
        tracker = TemporalCoalitionTracker(stability_threshold=0.6, formation_threshold=0.7)
        
        # Round 1: Coalition forms
        round1_data = {
            ("gpt-4", "gpt-4"): 0.8,
            ("claude-3", "claude-3"): 0.75,
            ("gpt-4", "claude-3"): 0.4  # No coalition
        }
        
        result = tracker.update_round(1, round1_data)
        
        assert len(result["new_coalitions"]) == 2
        assert ("claude-3", "claude-3") in result["new_coalitions"]
        assert ("gpt-4", "gpt-4") in result["new_coalitions"]
        assert len(result["active_coalitions"]) == 2
    
    def test_coalition_breaking(self):
        """Test coalition breaking detection."""
        tracker = TemporalCoalitionTracker(stability_threshold=0.6, formation_threshold=0.7)
        
        # Form coalition
        tracker.update_round(1, {("gpt-4", "gpt-4"): 0.8})
        
        # Break coalition
        result = tracker.update_round(2, {("gpt-4", "gpt-4"): 0.3})
        
        assert len(result["broken_coalitions"]) == 1
        assert ("gpt-4", "gpt-4") in result["broken_coalitions"]
        assert len(result["active_coalitions"]) == 0
    
    def test_coalition_stability_metrics(self):
        """Test stability metric calculations."""
        tracker = TemporalCoalitionTracker()
        
        # Simulate stable coalition
        for i in range(5):
            tracker.update_round(i+1, {("gpt-4", "claude-3"): 0.75 + 0.02 * i})
        
        stability = tracker.get_coalition_stability_metrics()
        pair = ("claude-3", "gpt-4")  # Ordered
        
        assert pair in stability
        assert stability[pair]["duration"] == 5
        assert stability[pair]["average_strength"] > 0.7
        assert stability[pair]["trend"]["direction"] == "increasing"
        assert stability[pair]["resilience"] == 1.0  # All rounds above threshold
    
    def test_cross_vs_same_model_patterns(self):
        """Test comparison between same and cross model coalitions."""
        tracker = TemporalCoalitionTracker()
        
        # Create mixed scenario data
        for i in range(5):
            tracker.update_round(i+1, {
                ("gpt-4", "gpt-4"): 0.85,  # Same model - stable
                ("claude-3", "claude-3"): 0.80,  # Same model - stable
                ("gpt-4", "claude-3"): 0.65 + 0.05 * np.random.randn(),  # Cross model - variable
                ("claude-3", "gemini"): 0.70  # Cross model
            })
        
        patterns = tracker.identify_cross_model_vs_same_model_patterns()
        
        assert patterns["same_model"]["count"] == 2
        assert patterns["cross_model"]["count"] == 2
        assert patterns["same_model"]["avg_stability"] > patterns["cross_model"]["avg_stability"]
    
    def test_coalition_cascades(self):
        """Test cascade detection."""
        tracker = TemporalCoalitionTracker()
        
        # Round 1: No cascade
        tracker.update_round(1, {("gpt-4", "gpt-4"): 0.8})
        
        # Round 2: Cascade event - multiple coalitions form
        tracker.update_round(2, {
            ("gpt-4", "claude-3"): 0.75,
            ("claude-3", "gemini"): 0.8,
            ("gemini", "gpt-4"): 0.77,
            ("claude-3", "claude-3"): 0.85
        })
        
        cascades = tracker.detect_coalition_cascades()
        
        assert len(cascades) == 1
        assert cascades[0]["round"] == 2
        assert cascades[0]["coalitions_formed"] == 4
        assert cascades[0]["type"] == "network_cascade"
    
    def test_defection_patterns(self):
        """Test defection pattern tracking."""
        tracker = TemporalCoalitionTracker()
        
        # Form and break coalitions
        tracker.update_round(1, {("gpt-4", "claude-3"): 0.8})
        tracker.update_round(2, {("gpt-4", "claude-3"): 0.85})
        tracker.update_round(3, {("gpt-4", "claude-3"): 0.3})  # Defection
        
        defections = tracker.track_defection_patterns()
        
        assert defections["total_defections"] == 1
        assert defections["defection_events"][0]["round"] == 3
        assert defections["defection_events"][0]["drop_magnitude"] > 0.5
    
    def test_coalition_network_data(self):
        """Test network visualization data generation."""
        tracker = TemporalCoalitionTracker()
        
        # Create network
        tracker.update_round(1, {
            ("gpt-4", "claude-3"): 0.8,
            ("claude-3", "gemini"): 0.75,
            ("gpt-4", "gpt-4"): 0.85
        })
        
        network = tracker.generate_coalition_network_data()
        
        assert len(network["nodes"]) == 3
        assert len(network["edges"]) == 3
        assert network["network_metrics"]["total_nodes"] == 3
        assert network["network_metrics"]["network_density"] > 0
        
        # Check node metrics
        gpt4_node = next(n for n in network["nodes"] if n["id"] == "gpt-4")
        assert gpt4_node["degree"] == 2  # Connected to claude-3 and itself


class TestEnhancedCrossModelAnalyzer:
    """Test enhanced CrossModelAnalyzer with temporal tracking."""
    
    def create_mock_game_data(self, rounds: int = 5) -> list:
        """Create mock game data for testing."""
        games = []
        game_id = 0
        
        for round_num in range(1, rounds + 1):
            # Same model games - stable cooperation
            games.append({
                "game_id": f"game_{game_id}",
                "round": round_num,
                "player1_id": 0,
                "player2_id": 1,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE"
            })
            game_id += 1
            
            # Cross model games - variable cooperation
            coop = "COOPERATE" if round_num > 2 else "DEFECT"
            games.append({
                "game_id": f"game_{game_id}",
                "round": round_num,
                "player1_id": 0,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": coop
            })
            game_id += 1
        
        return games
    
    def create_mock_strategy_data(self) -> list:
        """Create mock strategy data with model assignments."""
        return [
            {"agent_id": 0, "model": "gpt-4"},
            {"agent_id": 1, "model": "gpt-4"},
            {"agent_id": 2, "model": "claude-3"}
        ]
    
    def test_enhanced_coalition_analysis(self):
        """Test coalition analysis with temporal tracking."""
        analyzer = CrossModelAnalyzer()
        
        # Load mock data
        games = self.create_mock_game_data(rounds=5)
        strategies = self.create_mock_strategy_data()
        analyzer.load_data(games, strategies)
        
        # Analyze with temporal tracking
        result = analyzer.analyze_model_coalitions(include_temporal=True)
        
        assert result["detected"] is True
        assert "temporal_analysis" in result
        
        temporal = result["temporal_analysis"]
        assert "stability_metrics" in temporal
        assert "cross_vs_same_model" in temporal
        assert "coalition_cascades" in temporal
        assert "defection_patterns" in temporal
        assert "network_data" in temporal
    
    def test_mixed_scenario_coalition_detection(self):
        """Test coalition detection in mixed model scenarios."""
        analyzer = CrossModelAnalyzer()
        
        # Create diverse scenario
        games = []
        strategies = [
            {"agent_id": 0, "model": "gpt-4"},
            {"agent_id": 1, "model": "gpt-4"},
            {"agent_id": 2, "model": "claude-3"},
            {"agent_id": 3, "model": "claude-3"},
            {"agent_id": 4, "model": "gemini"},
            {"agent_id": 5, "model": "gemini"}
        ]
        
        # Create games with model-specific patterns
        for round_num in range(1, 6):
            # Same model pairs cooperate
            for i in range(0, 6, 2):
                games.append({
                    "round": round_num,
                    "player1_id": i,
                    "player2_id": i + 1,
                    "player1_action": "COOPERATE",
                    "player2_action": "COOPERATE"
                })
            
            # Cross model pairs have lower cooperation
            games.append({
                "round": round_num,
                "player1_id": 0,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "DEFECT" if round_num < 3 else "COOPERATE"
            })
        
        analyzer.load_data(games, strategies)
        result = analyzer.analyze_model_coalitions(include_temporal=True)
        
        # Check same vs cross model patterns
        cross_vs_same = result["temporal_analysis"]["cross_vs_same_model"]
        assert cross_vs_same["same_model"]["avg_strength"] > cross_vs_same["cross_model"]["avg_strength"]
    
    def test_stability_tracking_over_rounds(self):
        """Test tracking coalition stability over multiple rounds."""
        analyzer = CrossModelAnalyzer()
        
        # Create data with evolving coalitions
        games = []
        strategies = self.create_mock_strategy_data()
        
        # Early rounds: unstable
        for round_num in range(1, 4):
            games.append({
                "round": round_num,
                "player1_id": 0,
                "player2_id": 2,
                "player1_action": "COOPERATE" if round_num % 2 else "DEFECT",
                "player2_action": "DEFECT" if round_num % 2 else "COOPERATE"
            })
        
        # Later rounds: stable coalition forms
        for round_num in range(4, 8):
            games.append({
                "round": round_num,
                "player1_id": 0,
                "player2_id": 2,
                "player1_action": "COOPERATE",
                "player2_action": "COOPERATE"
            })
        
        analyzer.load_data(games, strategies)
        result = analyzer.analyze_model_coalitions(include_temporal=True)
        
        stability = result["temporal_analysis"]["stability_metrics"]
        # Check that stability improved over time
        assert len(stability) > 0
        
        # The coalition should show improving trend
        for pair, metrics in stability.items():
            if metrics["duration"] > 3:
                assert metrics["trend"]["direction"] == "increasing"