"""Tests for the SimilarityNode class."""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch

from src.nodes.similarity import SimilarityNode
from src.nodes.base import ContextKeys
from src.utils.data_manager import DataManager


class TestSimilarityNode:
    """Test suite for SimilarityNode."""
    
    @pytest.fixture
    def similarity_node(self):
        """Create a SimilarityNode instance for testing."""
        return SimilarityNode()
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock context with required keys."""
        mock_data_manager = Mock(spec=DataManager)
        mock_data_manager.experiment_path = Path("/fake/experiment/path")
        
        return {
            ContextKeys.DATA_MANAGER: mock_data_manager,
            ContextKeys.EXPERIMENT_ID: "test_exp_123"
        }
    
    @pytest.fixture
    def sample_strategies(self):
        """Create sample strategy data for testing."""
        return {
            "strategies": [
                {
                    "strategy_id": "r1_a0_123",
                    "agent_id": 0,
                    "round": 1,
                    "strategy": "Always cooperate with identical agents",
                    "full_reasoning": "Since we are all identical...",
                    "timestamp": "2024-01-15T10:30:00Z"
                },
                {
                    "strategy_id": "r1_a1_123",
                    "agent_id": 1,
                    "round": 1,
                    "strategy": "Defect against weaker agents, cooperate with stronger",
                    "full_reasoning": "Power dynamics suggest...",
                    "timestamp": "2024-01-15T10:30:01Z"
                },
                {
                    "strategy_id": "r1_a2_123",
                    "agent_id": 2,
                    "round": 1,
                    "strategy": "Always defect to maximize individual payoff",
                    "full_reasoning": "Game theory suggests...",
                    "timestamp": "2024-01-15T10:30:02Z"
                }
            ],
            "round": 1,
            "timestamp": "2024-01-15T10:30:00Z"
        }
    
    def test_initialization(self, similarity_node):
        """Test SimilarityNode initialization."""
        assert similarity_node.max_retries == 1
        assert similarity_node.vectorizer is not None
        assert similarity_node.vectorizer.max_features == 100
        assert similarity_node.vectorizer.ngram_range == (1, 2)
        assert similarity_node.clustering_linkage == 'average'
        assert similarity_node.min_clusters == 2
        assert similarity_node.max_clusters == 5
    
    @pytest.mark.asyncio
    async def test_execute_with_no_strategies(self, similarity_node, mock_context):
        """Test execution when no strategies are found."""
        # Mock empty rounds directory
        mock_data_manager = mock_context[ContextKeys.DATA_MANAGER]
        mock_data_manager.experiment_path = Path(tempfile.mkdtemp())
        
        try:
            result = await similarity_node.execute(mock_context)
            
            assert "similarity_analysis" in result
            assert result["similarity_analysis"]["error"] == "No strategies found"
            assert result["similarity_analysis"]["experiment_id"] == "test_exp_123"
        finally:
            shutil.rmtree(mock_data_manager.experiment_path)
    
    @pytest.mark.asyncio
    async def test_load_strategies(self, similarity_node, sample_strategies):
        """Test loading strategies from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data manager
            data_manager = DataManager(tmpdir)
            data_manager.experiment_id = "test_exp"
            data_manager.experiment_path = Path(tmpdir) / "test_exp"
            rounds_path = data_manager.experiment_path / "rounds"
            rounds_path.mkdir(parents=True)
            
            # Write sample strategy files
            for round_num in range(1, 4):
                strategy_file = rounds_path / f"strategies_r{round_num}.json"
                strategies = sample_strategies.copy()
                strategies["round"] = round_num
                with open(strategy_file, 'w') as f:
                    json.dump(strategies, f)
            
            # Load strategies
            strategies_by_round = await similarity_node._load_strategies(data_manager)
            
            assert len(strategies_by_round) == 3
            assert all(round_num in strategies_by_round for round_num in [1, 2, 3])
            assert len(strategies_by_round[1]) == 3
            assert len(similarity_node.strategy_texts[1]) == 3
    
    def test_strategy_vectorization(self, similarity_node):
        """Test TF-IDF vectorization of strategies."""
        # Create sample strategy texts
        similarity_node.strategy_texts = {
            1: [
                "Always cooperate with identical agents",
                "Defect against weaker agents",
                "Always defect to maximize payoff"
            ],
            2: [
                "Cooperate if opponent cooperated last time",
                "Always cooperate to build trust",
                "Defect first then cooperate"
            ]
        }
        
        strategies_by_round = {1: [{}] * 3, 2: [{}] * 3}
        
        # Run vectorization
        pytest.mark.asyncio
        async def run_vectorization():
            await similarity_node._vectorize_strategies(strategies_by_round)
        
        import asyncio
        asyncio.run(run_vectorization())
        
        # Check vectors were created
        assert 1 in similarity_node.strategy_vectors
        assert 2 in similarity_node.strategy_vectors
        
        # Check vector properties
        vectors_r1 = similarity_node.strategy_vectors[1]
        assert vectors_r1.shape == (3, vectors_r1.shape[1])  # 3 strategies
        
        # Check normalization (unit vectors)
        norms = np.linalg.norm(vectors_r1, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)
    
    def test_tfidf_with_empty_strategies(self, similarity_node):
        """Test TF-IDF handles empty strategy texts."""
        similarity_node.strategy_texts = {
            1: ["", "Valid strategy", ""]
        }
        
        strategies_by_round = {1: [{}] * 3}
        
        pytest.mark.asyncio
        async def run_vectorization():
            await similarity_node._vectorize_strategies(strategies_by_round)
        
        import asyncio
        asyncio.run(run_vectorization())
        
        # Should handle empty strings gracefully
        assert 1 in similarity_node.strategy_vectors
        vectors = similarity_node.strategy_vectors[1]
        assert vectors.shape[0] == 3
    
    @pytest.mark.asyncio
    async def test_cosine_similarity_computation(self, similarity_node):
        """Test cosine similarity matrix computation."""
        # Create simple vectors for testing
        similarity_node.strategy_vectors = {
            1: np.array([
                [1.0, 0.0, 0.0],  # Orthogonal vectors
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
        }
        
        await similarity_node._compute_similarity_matrices()
        
        assert 1 in similarity_node.similarity_matrices
        sim_matrix = similarity_node.similarity_matrices[1]
        
        # Check diagonal is 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(np.diag(sim_matrix), np.ones(3))
        
        # Check off-diagonal elements are 0.0 (orthogonal)
        assert sim_matrix[0, 1] == pytest.approx(0.0, abs=1e-5)
        assert sim_matrix[0, 2] == pytest.approx(0.0, abs=1e-5)
        assert sim_matrix[1, 2] == pytest.approx(0.0, abs=1e-5)
    
    def test_similarity_matrix_properties(self, similarity_node):
        """Test properties of similarity matrices."""
        # Create vectors with known similarities
        similarity_node.strategy_vectors = {
            1: np.array([
                [1.0, 0.0],
                [0.8, 0.6],  # Similar to first vector
                [0.0, 1.0]   # Orthogonal to first
            ])
        }
        
        pytest.mark.asyncio
        async def compute():
            await similarity_node._compute_similarity_matrices()
        
        import asyncio
        asyncio.run(compute())
        
        sim_matrix = similarity_node.similarity_matrices[1]
        
        # Check symmetry
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)
        
        # Check bounds [0, 1]
        assert np.all(sim_matrix >= 0.0)
        assert np.all(sim_matrix <= 1.0)
        
        # Check specific similarities
        assert sim_matrix[0, 1] > 0.5  # Similar vectors
        assert sim_matrix[0, 2] < 0.1  # Orthogonal vectors
    
    def test_convergence_metric_calculation(self, similarity_node):
        """Test calculation of convergence metrics."""
        # Test increasing similarity trend (convergence)
        trend = [0.5, 0.6, 0.7, 0.75, 0.76, 0.77, 0.77, 0.78, 0.78, 0.78]
        metrics = similarity_node._calculate_convergence_metrics(trend)
        
        assert metrics["strategy_convergence"] == 0.78
        assert metrics["rounds_to_convergence"] is not None
        assert metrics["convergence_trend"] == trend
        
        # Test empty trend
        empty_metrics = similarity_node._calculate_convergence_metrics([])
        assert empty_metrics["strategy_convergence"] == 0.0
        assert empty_metrics["rounds_to_convergence"] is None
    
    @pytest.mark.asyncio
    async def test_clustering_identification(self, similarity_node):
        """Test clustering algorithm identifies distinct groups."""
        # Create vectors with clear clusters
        cluster1 = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.95, 0.05, 0.0]])
        cluster2 = np.array([[0.0, 1.0, 0.0], [0.1, 0.9, 0.0], [0.05, 0.95, 0.0]])
        vectors = np.vstack([cluster1, cluster2])
        
        similarity_node.strategy_vectors = {1: vectors}
        
        # Compute similarities and perform clustering
        await similarity_node._compute_similarity_matrices()
        await similarity_node._perform_clustering_analysis()
        
        assert 1 in similarity_node.cluster_assignments
        clusters = similarity_node.cluster_assignments[1]
        
        # Should identify 2 distinct clusters
        unique_clusters = np.unique(clusters)
        assert len(unique_clusters) >= 2
        
        # Agents 0,1,2 should be in same cluster, 3,4,5 in another
        assert clusters[0] == clusters[1] == clusters[2]
        assert clusters[3] == clusters[4] == clusters[5]
        assert clusters[0] != clusters[3]
    
    def test_cluster_evolution_tracking(self, similarity_node):
        """Test tracking of cluster membership across rounds."""
        # Mock cluster assignments for multiple rounds
        similarity_node.cluster_assignments = {
            1: np.array([1, 1, 2, 2]),
            2: np.array([1, 2, 2, 1]),
            3: np.array([1, 1, 1, 1])  # All converged to same cluster
        }
        
        analysis = similarity_node._generate_clustering_analysis([1, 2, 3])
        
        assert "cluster_evolution" in analysis
        evolution = analysis["cluster_evolution"]
        
        # Check round 1 clusters
        assert "1" in evolution
        assert set(evolution["1"]["0"]) == {0, 1}  # Cluster 0 (originally 1)
        assert set(evolution["1"]["1"]) == {2, 3}  # Cluster 1 (originally 2)
        
        # Check convergence in round 3
        assert "3" in evolution
        assert len(evolution["3"]) == 1  # Only one cluster
    
    @pytest.mark.asyncio
    async def test_visualization_data_generation(self, similarity_node):
        """Test generation of 2D visualization coordinates."""
        # Create some vectors
        similarity_node.strategy_vectors = {
            1: np.random.rand(5, 10),  # 5 strategies, 10 features
            2: np.random.rand(5, 10)
        }
        
        viz_data = await similarity_node._generate_visualization_data()
        
        assert "strategy_embeddings_2d" in viz_data
        embeddings = viz_data["strategy_embeddings_2d"]
        
        assert "1" in embeddings
        assert "2" in embeddings
        
        # Check 2D coordinates
        round1_coords = embeddings["1"]
        assert len(round1_coords) == 5  # 5 strategies
        assert all(len(coord) == 2 for coord in round1_coords)  # 2D coordinates
    
    @pytest.mark.asyncio
    async def test_handle_missing_rounds(self, similarity_node, mock_context):
        """Test handling of missing round files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_manager = DataManager(tmpdir)
            data_manager.experiment_id = "test_exp"
            data_manager.experiment_path = Path(tmpdir) / "test_exp"
            rounds_path = data_manager.experiment_path / "rounds"
            rounds_path.mkdir(parents=True)
            
            # Only create files for rounds 1 and 3 (skip 2)
            for round_num in [1, 3]:
                strategy_file = rounds_path / f"strategies_r{round_num}.json"
                with open(strategy_file, 'w') as f:
                    json.dump({
                        "round": round_num,
                        "strategies": [{
                            "strategy": f"Strategy for round {round_num}",
                            "agent_id": 0
                        }]
                    }, f)
            
            strategies = await similarity_node._load_strategies(data_manager)
            
            # Should load available rounds
            assert len(strategies) == 2
            assert 1 in strategies
            assert 3 in strategies
            assert 2 not in strategies
    
    @pytest.mark.asyncio
    async def test_similarity_report_generation(self, similarity_node):
        """Test generation of complete similarity report."""
        # Setup mock data
        similarity_node.similarity_matrices = {
            1: np.array([[1.0, 0.8], [0.8, 1.0]]),
            2: np.array([[1.0, 0.9], [0.9, 1.0]])
        }
        similarity_node.cluster_assignments = {1: np.array([1, 1]), 2: np.array([1, 1])}
        similarity_node.strategy_vectors = {1: np.array([[1, 0], [0.9, 0.1]]), 
                                           2: np.array([[1, 0], [0.95, 0.05]])}
        
        report = await similarity_node._generate_similarity_report("test_exp_123")
        
        assert "strategy_similarity_analysis" in report
        analysis = report["strategy_similarity_analysis"]
        
        assert analysis["experiment_id"] == "test_exp_123"
        assert analysis["rounds_analyzed"] == [1, 2]
        assert "similarity_by_round" in analysis
        assert "convergence_metrics" in analysis
        assert "clustering_analysis" in analysis
        assert "visualization_data" in analysis
        assert "metadata" in analysis
        
        # Check similarity trend
        convergence = analysis["convergence_metrics"]["convergence_trend"]
        assert convergence[0] < convergence[1]  # Increasing similarity
    
    @pytest.mark.asyncio
    async def test_identical_strategies_similarity(self, similarity_node):
        """Test that identical strategies have similarity of 1.0."""
        # Create identical strategies - need more varied text to avoid max_df issue
        similarity_node.strategy_texts = {
            1: [
                "Always cooperate with everyone",
                "Always cooperate with everyone",
                "Always cooperate with everyone"
            ],
            2: [
                "Sometimes defect against opponents",
                "Sometimes defect against opponents", 
                "Sometimes defect against opponents"
            ]
        }
        
        strategies_by_round = {1: [{}] * 3, 2: [{}] * 3}
        
        # Vectorize and compute similarities
        await similarity_node._vectorize_strategies(strategies_by_round)
        await similarity_node._compute_similarity_matrices()
        
        sim_matrix = similarity_node.similarity_matrices[1]
        
        # All pairwise similarities should be 1.0
        np.testing.assert_array_almost_equal(sim_matrix, np.ones((3, 3)), decimal=5)
    
    @pytest.mark.asyncio
    async def test_completely_different_strategies(self, similarity_node):
        """Test that completely different strategies have low similarity."""
        # Create very different strategies
        similarity_node.strategy_texts = {
            1: [
                "Always cooperate trust friendship",
                "Always defect betray enemy",
                "Random chaos unpredictable noise"
            ]
        }
        
        strategies_by_round = {1: [{}] * 3}
        
        # Vectorize and compute similarities
        await similarity_node._vectorize_strategies(strategies_by_round)
        await similarity_node._compute_similarity_matrices()
        
        sim_matrix = similarity_node.similarity_matrices[1]
        
        # Off-diagonal similarities should be low
        assert sim_matrix[0, 1] < 0.5
        assert sim_matrix[0, 2] < 0.5
        assert sim_matrix[1, 2] < 0.5
    
    def test_metadata_generation(self, similarity_node):
        """Test that metadata is correctly generated."""
        similarity_node.similarity_matrices = {1: np.eye(2)}
        similarity_node.cluster_assignments = {1: np.array([1, 1])}
        similarity_node.strategy_vectors = {1: np.array([[1, 0], [0, 1]])}
        
        pytest.mark.asyncio
        async def generate_report():
            return await similarity_node._generate_similarity_report("test_exp")
        
        import asyncio
        report = asyncio.run(generate_report())
        
        metadata = report["strategy_similarity_analysis"]["metadata"]
        
        assert metadata["vectorization_method"] == "tfidf"
        assert metadata["similarity_metric"] == "cosine"
        assert metadata["clustering_algorithm"] == "hierarchical"
        assert metadata["parameters"]["tfidf_max_features"] == 100
        assert metadata["parameters"]["tfidf_ngram_range"] == [1, 2]
        assert metadata["parameters"]["clustering_linkage"] == "average"
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, similarity_node):
        """Test various edge cases."""
        # Test with single strategy - need more varied corpus
        similarity_node.strategy_texts = {
            1: ["Only one strategy"],
            2: ["Another different strategy", "Yet another approach"]
        }
        strategies_by_round = {1: [{}], 2: [{}] * 2}
        
        await similarity_node._vectorize_strategies(strategies_by_round)
        await similarity_node._compute_similarity_matrices()
        
        # Should handle single strategy gracefully
        assert 1 in similarity_node.similarity_matrices
        sim_matrix = similarity_node.similarity_matrices[1]
        assert sim_matrix.shape == (1, 1)
        assert sim_matrix[0, 0] == 1.0
        
        # Test with no rounds
        metrics = similarity_node._calculate_convergence_metrics([])
        assert metrics["strategy_convergence"] == 0.0
        assert metrics["rounds_to_convergence"] is None
    
    @pytest.mark.asyncio
    async def test_save_results(self, similarity_node, mock_context):
        """Test saving analysis results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data manager
            data_manager = DataManager(tmpdir)
            data_manager.experiment_id = "test_exp"
            data_manager.experiment_path = Path(tmpdir) / "test_exp"
            data_manager.experiment_path.mkdir(parents=True)
            
            # Mock the _write_json method to track calls
            original_write_json = data_manager._write_json
            write_json_calls = []
            
            def mock_write_json(path, data):
                write_json_calls.append((path, data))
                # Use the original method to actually write the file
                original_write_json(path, data)
            
            data_manager._write_json = mock_write_json
            
            # Create analysis results
            analysis_results = {
                "strategy_similarity_analysis": {
                    "experiment_id": "test_exp",
                    "rounds_analyzed": [1, 2],
                    "similarity_by_round": {}
                }
            }
            
            # Save results
            await similarity_node._save_results(data_manager, analysis_results)
            
            # Check that _write_json was called with correct arguments
            assert len(write_json_calls) == 1
            path_arg, data_arg = write_json_calls[0]
            assert path_arg == data_manager.experiment_path / "strategy_similarity.json"
            assert data_arg == analysis_results
            
            # Check file was created
            output_file = data_manager.experiment_path / "strategy_similarity.json"
            assert output_file.exists()
            
            # Check content
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data == analysis_results

    @pytest.mark.asyncio
    async def test_integration_with_experiment(self, similarity_node, sample_strategies):
        """Test integration with experiment flow - similarity metrics added to context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup data manager and experiment structure
            data_manager = DataManager(tmpdir)
            data_manager.experiment_id = "test_exp"
            data_manager.experiment_path = Path(tmpdir) / "test_exp"
            rounds_path = data_manager.experiment_path / "rounds"
            rounds_path.mkdir(parents=True)
            
            # Create strategy files for multiple rounds
            for round_num in range(1, 4):
                strategy_file = rounds_path / f"strategies_r{round_num}.json"
                strategies = sample_strategies.copy()
                strategies["round"] = round_num
                # Modify strategies slightly for each round to show convergence
                for i, strat in enumerate(strategies["strategies"]):
                    if round_num > 1:
                        strat["strategy"] = strat["strategy"].replace("Always", "Often")
                    if round_num > 2:
                        strat["strategy"] = strat["strategy"].replace("Often", "Usually")
                
                with open(strategy_file, 'w') as f:
                    json.dump(strategies, f)
            
            # Create context as experiment flow would
            context = {
                ContextKeys.DATA_MANAGER: data_manager,
                ContextKeys.EXPERIMENT_ID: "test_exp"
            }
            
            # Execute similarity analysis
            result_context = await similarity_node.execute(context)
            
            # Verify similarity analysis was added to context
            assert "similarity_analysis" in result_context
            assert "strategy_similarity_analysis" in result_context["similarity_analysis"]
            
            analysis = result_context["similarity_analysis"]["strategy_similarity_analysis"]
            
            # Verify key metrics are present
            assert "experiment_id" in analysis
            assert analysis["experiment_id"] == "test_exp"
            assert "rounds_analyzed" in analysis
            assert analysis["rounds_analyzed"] == [1, 2, 3]
            assert "convergence_metrics" in analysis
            assert "strategy_convergence" in analysis["convergence_metrics"]
            assert "similarity_by_round" in analysis
            
            # Verify output file was created
            output_file = data_manager.experiment_path / "strategy_similarity.json"
            assert output_file.exists()
            
            # Verify the convergence metric can be extracted as experiment flow does
            convergence_metrics = analysis.get("convergence_metrics", {})
            strategy_convergence = convergence_metrics.get("strategy_convergence", 0.0)
            assert isinstance(strategy_convergence, float)
            assert 0.0 <= strategy_convergence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])