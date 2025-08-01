"""Similarity analysis node for comparing agent strategies across rounds."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import re
from datetime import datetime

import numpy as np
from scipy.spatial.distance import cosine, squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.nodes.base import AsyncNode, ContextKeys, validate_context
from src.utils.data_manager import DataManager

logger = logging.getLogger(__name__)


class SimilarityNode(AsyncNode):
    """Computes similarity between agent strategies across rounds."""
    
    def __init__(self):
        """Initialize SimilarityNode with vectorizer and configuration."""
        super().__init__(max_retries=1)  # Don't retry analysis failures
        
        # TF-IDF configuration for strategy vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english',
            lowercase=True,
            min_df=1,  # Don't exclude rare terms in small corpus
            max_df=0.95  # Exclude very common terms
        )
        
        # Clustering configuration
        self.clustering_linkage = 'average'
        self.min_clusters = 2
        self.max_clusters = 5
        
        # Storage for analysis results
        self.strategy_texts: Dict[int, List[str]] = {}
        self.strategy_vectors: Dict[int, np.ndarray] = {}
        self.similarity_matrices: Dict[int, np.ndarray] = {}
        self.cluster_assignments: Dict[int, np.ndarray] = {}
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute similarity analysis on strategies from all rounds.
        
        Args:
            context: Experiment context containing data_manager and experiment_id
            
        Returns:
            Updated context with similarity_analysis results
        """
        # Validate required context keys
        validate_context(context, [ContextKeys.DATA_MANAGER, ContextKeys.EXPERIMENT_ID])
        
        data_manager: DataManager = context[ContextKeys.DATA_MANAGER]
        experiment_id = context[ContextKeys.EXPERIMENT_ID]
        
        logger.info(f"Starting similarity analysis for experiment {experiment_id}")
        
        try:
            # Load all strategies from files
            strategies_by_round = await self._load_strategies(data_manager)
            
            if not strategies_by_round:
                logger.warning("No strategies found for similarity analysis")
                context["similarity_analysis"] = {
                    "error": "No strategies found",
                    "experiment_id": experiment_id,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                return context
            
            # Vectorize all strategies
            await self._vectorize_strategies(strategies_by_round)
            
            # Compute similarity matrices for each round
            await self._compute_similarity_matrices()
            
            # Perform clustering analysis
            await self._perform_clustering_analysis()
            
            # Generate similarity report
            analysis_results = await self._generate_similarity_report(experiment_id)
            
            # Save results to file
            await self._save_results(data_manager, analysis_results)
            
            # Add results to context
            context["similarity_analysis"] = analysis_results
            
            logger.info("Similarity analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Similarity analysis failed: {e}")
            context["similarity_analysis"] = {
                "error": str(e),
                "experiment_id": experiment_id,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        return context
    
    async def _load_strategies(self, data_manager: DataManager) -> Dict[int, List[Dict[str, Any]]]:
        """Load strategies from all round files.
        
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
        
        for file_path in strategy_files:
            # Extract round number from filename
            match = re.search(r'strategies_r(\d+)\.json', file_path.name)
            if not match:
                logger.warning(f"Unexpected filename format: {file_path.name}")
                continue
                
            round_num = int(match.group(1))
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    strategies = data.get("strategies", [])
                    
                    if strategies:
                        strategies_by_round[round_num] = strategies
                        # Extract strategy texts for vectorization
                        self.strategy_texts[round_num] = [
                            s.get("strategy", "") for s in strategies
                        ]
                        logger.info(f"Loaded {len(strategies)} strategies from round {round_num}")
                    
            except Exception as e:
                logger.error(f"Failed to load strategies from {file_path}: {e}")
                continue
        
        logger.info(f"Loaded strategies from {len(strategies_by_round)} rounds")
        return strategies_by_round
    
    async def _vectorize_strategies(self, strategies_by_round: Dict[int, List[Dict[str, Any]]]) -> None:
        """Vectorize strategies using TF-IDF.
        
        Args:
            strategies_by_round: Dictionary of strategies by round
        """
        # Collect all strategy texts for fitting vectorizer
        all_texts = []
        for round_num in sorted(strategies_by_round.keys()):
            all_texts.extend(self.strategy_texts[round_num])
        
        if not all_texts:
            logger.warning("No strategy texts to vectorize")
            return
        
        # Fit vectorizer on all texts
        logger.info(f"Fitting vectorizer on {len(all_texts)} strategy texts")
        self.vectorizer.fit(all_texts)
        
        # Transform strategies for each round
        for round_num in sorted(strategies_by_round.keys()):
            texts = self.strategy_texts[round_num]
            if texts:
                # Handle empty strategies
                texts = [text if text else "No strategy provided" for text in texts]
                vectors = self.vectorizer.transform(texts)
                # Normalize to unit vectors for cosine similarity
                normalized_vectors = vectors.toarray()
                norms = np.linalg.norm(normalized_vectors, axis=1, keepdims=True)
                # Avoid division by zero
                norms[norms == 0] = 1.0
                self.strategy_vectors[round_num] = normalized_vectors / norms
                logger.debug(f"Vectorized {len(texts)} strategies for round {round_num}")
    
    async def _compute_similarity_matrices(self) -> None:
        """Compute pairwise cosine similarity matrices for each round."""
        for round_num, vectors in self.strategy_vectors.items():
            n_strategies = vectors.shape[0]
            
            # Initialize similarity matrix
            similarity_matrix = np.zeros((n_strategies, n_strategies))
            
            # Compute pairwise similarities
            for i in range(n_strategies):
                for j in range(i, n_strategies):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Cosine similarity = 1 - cosine distance
                        sim = 1.0 - cosine(vectors[i], vectors[j])
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim  # Symmetric
            
            self.similarity_matrices[round_num] = similarity_matrix
            
            # Log average similarity for this round
            # Exclude diagonal (self-similarity)
            mask = ~np.eye(n_strategies, dtype=bool)
            avg_similarity = similarity_matrix[mask].mean()
            logger.info(f"Round {round_num}: Average similarity = {avg_similarity:.3f}")
    
    async def _perform_clustering_analysis(self) -> None:
        """Perform hierarchical clustering on strategy similarities."""
        for round_num, similarity_matrix in self.similarity_matrices.items():
            n_strategies = similarity_matrix.shape[0]
            
            if n_strategies < 2:
                logger.warning(f"Not enough strategies for clustering in round {round_num}")
                continue
            
            # Convert similarity to distance for clustering
            distance_matrix = 1.0 - similarity_matrix
            
            # Ensure distance matrix is valid (no negative values)
            distance_matrix = np.maximum(distance_matrix, 0)
            
            # Convert to condensed distance matrix
            condensed_dist = squareform(distance_matrix, checks=False)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_dist, method=self.clustering_linkage)
            
            # Find optimal number of clusters using silhouette score
            best_n_clusters = 2
            best_score = -1
            
            max_possible_clusters = min(self.max_clusters, n_strategies - 1)
            
            for n_clusters in range(self.min_clusters, max_possible_clusters + 1):
                clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                # Only compute silhouette if we have enough samples
                if len(np.unique(clusters)) > 1:
                    score = silhouette_score(distance_matrix, clusters, metric='precomputed')
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            
            # Get final cluster assignments
            self.cluster_assignments[round_num] = fcluster(
                linkage_matrix, best_n_clusters, criterion='maxclust'
            )
            
            logger.info(f"Round {round_num}: Identified {best_n_clusters} clusters "
                       f"(silhouette score: {best_score:.3f})")
    
    async def _generate_similarity_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate comprehensive similarity analysis report.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary containing complete analysis results
        """
        rounds_analyzed = sorted(self.similarity_matrices.keys())
        
        # Compute similarity metrics by round
        similarity_by_round = {}
        convergence_trend = []
        
        for round_num in rounds_analyzed:
            sim_matrix = self.similarity_matrices[round_num]
            n = sim_matrix.shape[0]
            
            # Calculate metrics (excluding self-similarity)
            mask = ~np.eye(n, dtype=bool)
            similarities = sim_matrix[mask]
            
            avg_sim = float(similarities.mean())
            convergence_trend.append(avg_sim)
            
            similarity_by_round[str(round_num)] = {
                "average_similarity": avg_sim,
                "similarity_matrix": sim_matrix.tolist(),
                "min_similarity": float(similarities.min()),
                "max_similarity": float(similarities.max())
            }
        
        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(convergence_trend)
        
        # Generate clustering analysis
        clustering_analysis = self._generate_clustering_analysis(rounds_analyzed)
        
        # Generate visualization data
        visualization_data = await self._generate_visualization_data()
        
        # Compile metadata
        metadata = {
            "vectorization_method": "tfidf",
            "similarity_metric": "cosine",
            "clustering_algorithm": "hierarchical",
            "parameters": {
                "tfidf_max_features": self.vectorizer.max_features,
                "tfidf_ngram_range": list(self.vectorizer.ngram_range),
                "clustering_linkage": self.clustering_linkage
            }
        }
        
        return {
            "strategy_similarity_analysis": {
                "experiment_id": experiment_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "rounds_analyzed": rounds_analyzed,
                "similarity_by_round": similarity_by_round,
                "convergence_metrics": convergence_metrics,
                "clustering_analysis": clustering_analysis,
                "visualization_data": visualization_data,
                "metadata": metadata
            }
        }
    
    def _calculate_convergence_metrics(self, convergence_trend: List[float]) -> Dict[str, Any]:
        """Calculate strategy convergence metrics.
        
        Args:
            convergence_trend: List of average similarities by round
            
        Returns:
            Dictionary of convergence metrics
        """
        if not convergence_trend:
            return {
                "strategy_convergence": 0.0,
                "rounds_to_convergence": None,
                "convergence_trend": []
            }
        
        # Strategy convergence is the final average similarity
        strategy_convergence = convergence_trend[-1]
        
        # Find rounds to convergence (when similarity stabilizes)
        # Define convergence as similarity change < 0.02 for 2 consecutive rounds
        rounds_to_convergence = None
        if len(convergence_trend) >= 3:
            for i in range(2, len(convergence_trend)):
                if (abs(convergence_trend[i] - convergence_trend[i-1]) < 0.02 and
                    abs(convergence_trend[i-1] - convergence_trend[i-2]) < 0.02):
                    rounds_to_convergence = i - 1  # Convert to 1-based round number
                    break
        
        return {
            "strategy_convergence": float(strategy_convergence),
            "rounds_to_convergence": rounds_to_convergence,
            "convergence_trend": [float(x) for x in convergence_trend]
        }
    
    def _generate_clustering_analysis(self, rounds_analyzed: List[int]) -> Dict[str, Any]:
        """Generate clustering analysis results.
        
        Args:
            rounds_analyzed: List of round numbers analyzed
            
        Returns:
            Dictionary of clustering analysis
        """
        if not self.cluster_assignments:
            return {
                "optimal_clusters": 0,
                "cluster_descriptions": {},
                "cluster_evolution": {}
            }
        
        # Find most common number of clusters
        cluster_counts = [len(np.unique(self.cluster_assignments[r])) 
                         for r in rounds_analyzed if r in self.cluster_assignments]
        optimal_clusters = int(np.median(cluster_counts)) if cluster_counts else 0
        
        # Generate cluster descriptions (placeholder - would need actual strategy analysis)
        cluster_descriptions = {}
        for i in range(optimal_clusters):
            cluster_descriptions[str(i)] = f"Strategy cluster {i+1}"
        
        # Track cluster evolution
        cluster_evolution = {}
        for round_num in rounds_analyzed:
            if round_num in self.cluster_assignments:
                clusters = self.cluster_assignments[round_num]
                round_clusters = {}
                for cluster_id in np.unique(clusters):
                    # Get agent indices in this cluster
                    agent_indices = np.where(clusters == cluster_id)[0].tolist()
                    round_clusters[str(cluster_id - 1)] = agent_indices  # 0-based cluster IDs
                cluster_evolution[str(round_num)] = round_clusters
        
        return {
            "optimal_clusters": optimal_clusters,
            "cluster_descriptions": cluster_descriptions,
            "cluster_evolution": cluster_evolution
        }
    
    async def _generate_visualization_data(self) -> Dict[str, Any]:
        """Generate 2D coordinates for strategy visualization.
        
        Returns:
            Dictionary with 2D embeddings for each round
        """
        strategy_embeddings_2d = {}
        
        for round_num, vectors in self.strategy_vectors.items():
            n_strategies = vectors.shape[0]
            
            if n_strategies < 2:
                # Not enough data for PCA
                strategy_embeddings_2d[str(round_num)] = [[0.0, 0.0]] * n_strategies
                continue
            
            # Use PCA for dimensionality reduction
            n_components = min(2, n_strategies, vectors.shape[1])
            pca = PCA(n_components=n_components)
            
            try:
                embeddings = pca.fit_transform(vectors)
                
                # If only 1 component, add zeros for second dimension
                if n_components == 1:
                    embeddings = np.column_stack([embeddings, np.zeros(n_strategies)])
                
                strategy_embeddings_2d[str(round_num)] = embeddings.tolist()
                
            except Exception as e:
                logger.warning(f"PCA failed for round {round_num}: {e}")
                # Fallback to random positions
                strategy_embeddings_2d[str(round_num)] = np.random.randn(n_strategies, 2).tolist()
        
        return {
            "strategy_embeddings_2d": strategy_embeddings_2d
        }
    
    async def _save_results(self, data_manager: DataManager, analysis_results: Dict[str, Any]) -> None:
        """Save similarity analysis results to file.
        
        Args:
            data_manager: DataManager instance for file operations
            analysis_results: Complete analysis results
        """
        output_path = data_manager.experiment_path / "strategy_similarity.json"
        data_manager._write_json(output_path, analysis_results)
        logger.info(f"Saved similarity analysis to {output_path}")