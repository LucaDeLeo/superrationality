"""Statistical tests for diversity-cooperation correlation."""
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DiversityStatistics:
    """Statistical analysis for model diversity impacts."""
    
    @staticmethod
    def test_diversity_cooperation_correlation(
        diversity_scores: List[float], 
        cooperation_rates: List[float]
    ) -> Dict:
        """
        Test correlation between diversity and cooperation.
        
        Args:
            diversity_scores: List of diversity scores from experiments
            cooperation_rates: List of corresponding cooperation rates
            
        Returns:
            Dictionary with statistical test results
        """
        if len(diversity_scores) != len(cooperation_rates):
            raise ValueError("Diversity scores and cooperation rates must have same length")
        
        if len(diversity_scores) < 3:
            return {
                "error": "Insufficient data points for correlation analysis (need at least 3)",
                "n_samples": len(diversity_scores)
            }
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(diversity_scores, cooperation_rates)
        
        # Spearman rank correlation (non-parametric)
        spearman_r, spearman_p = stats.spearmanr(diversity_scores, cooperation_rates)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(diversity_scores, cooperation_rates)
        
        # Effect size (Cohen's d for high vs low diversity)
        median_diversity = np.median(diversity_scores)
        high_div_indices = [i for i, d in enumerate(diversity_scores) if d >= median_diversity]
        low_div_indices = [i for i, d in enumerate(diversity_scores) if d < median_diversity]
        
        if high_div_indices and low_div_indices:
            high_div_coop = [cooperation_rates[i] for i in high_div_indices]
            low_div_coop = [cooperation_rates[i] for i in low_div_indices]
            cohens_d = DiversityStatistics._calculate_cohens_d(high_div_coop, low_div_coop)
        else:
            cohens_d = None
        
        return {
            "n_samples": len(diversity_scores),
            "pearson_correlation": {
                "coefficient": pearson_r,
                "p_value": pearson_p,
                "significant": pearson_p < 0.05,
                "interpretation": DiversityStatistics._interpret_correlation(pearson_r)
            },
            "spearman_correlation": {
                "coefficient": spearman_r,
                "p_value": spearman_p,
                "significant": spearman_p < 0.05
            },
            "linear_regression": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "std_error": std_err,
                "prediction_formula": f"cooperation = {slope:.3f} * diversity + {intercept:.3f}"
            },
            "effect_size": {
                "cohens_d": cohens_d,
                "interpretation": DiversityStatistics._interpret_cohens_d(cohens_d) if cohens_d else None
            }
        }
    
    @staticmethod
    def test_diversity_groups_anova(scenario_results: List[Dict]) -> Dict:
        """
        ANOVA test for differences between diversity groups.
        
        Args:
            scenario_results: List of scenario results with diversity and cooperation
            
        Returns:
            Dictionary with ANOVA results
        """
        # Group scenarios by diversity level
        homogeneous = []
        balanced = []
        diverse = []
        
        for result in scenario_results:
            diversity = result.get('diversity_score', 0)
            cooperation = result.get('overall_cooperation_rate', 0)
            
            if diversity == 0:
                homogeneous.append(cooperation)
            elif diversity < 0.8:
                balanced.append(cooperation)
            else:
                diverse.append(cooperation)
        
        # Need at least 2 groups with data
        groups = [g for g in [homogeneous, balanced, diverse] if g]
        if len(groups) < 2:
            return {
                "error": "Insufficient groups for ANOVA (need at least 2)",
                "n_groups": len(groups)
            }
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Post-hoc tests if significant
        post_hoc = {}
        if p_value < 0.05 and len(groups) >= 2:
            # Tukey HSD would be ideal but requires more setup
            # Using pairwise t-tests with Bonferroni correction
            comparisons = []
            if homogeneous and balanced:
                t_stat, t_p = stats.ttest_ind(homogeneous, balanced)
                comparisons.append({
                    "comparison": "homogeneous_vs_balanced",
                    "t_statistic": t_stat,
                    "p_value": t_p,
                    "corrected_p": min(t_p * 3, 1.0)  # Bonferroni
                })
            if homogeneous and diverse:
                t_stat, t_p = stats.ttest_ind(homogeneous, diverse)
                comparisons.append({
                    "comparison": "homogeneous_vs_diverse",
                    "t_statistic": t_stat,
                    "p_value": t_p,
                    "corrected_p": min(t_p * 3, 1.0)
                })
            if balanced and diverse:
                t_stat, t_p = stats.ttest_ind(balanced, diverse)
                comparisons.append({
                    "comparison": "balanced_vs_diverse",
                    "t_statistic": t_stat,
                    "p_value": t_p,
                    "corrected_p": min(t_p * 3, 1.0)
                })
            post_hoc["pairwise_comparisons"] = comparisons
        
        return {
            "test": "One-way ANOVA",
            "groups": {
                "homogeneous": {"n": len(homogeneous), "mean": np.mean(homogeneous) if homogeneous else None},
                "balanced": {"n": len(balanced), "mean": np.mean(balanced) if balanced else None},
                "diverse": {"n": len(diverse), "mean": np.mean(diverse) if diverse else None}
            },
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "post_hoc": post_hoc if p_value < 0.05 else None
        }
    
    @staticmethod
    def test_diversity_stability(cooperation_by_round_data: Dict[float, List[float]]) -> Dict:
        """
        Test if diversity affects cooperation stability across rounds.
        
        Args:
            cooperation_by_round_data: Dict mapping diversity score to cooperation rates by round
            
        Returns:
            Dictionary with stability analysis
        """
        stability_results = {}
        
        for diversity, cooperation_rates in cooperation_by_round_data.items():
            if len(cooperation_rates) < 2:
                continue
            
            # Calculate stability metrics
            variance = np.var(cooperation_rates)
            std_dev = np.std(cooperation_rates)
            
            # Calculate autocorrelation (lag-1)
            if len(cooperation_rates) > 2:
                autocorr = np.corrcoef(cooperation_rates[:-1], cooperation_rates[1:])[0, 1]
            else:
                autocorr = None
            
            # Test for trend
            rounds = np.arange(len(cooperation_rates))
            slope, _, r_value, p_value, _ = stats.linregress(rounds, cooperation_rates)
            
            stability_results[f"diversity_{diversity:.3f}"] = {
                "variance": variance,
                "std_deviation": std_dev,
                "coefficient_of_variation": std_dev / np.mean(cooperation_rates) if np.mean(cooperation_rates) > 0 else None,
                "autocorrelation": autocorr,
                "trend": {
                    "slope": slope,
                    "r_squared": r_value ** 2,
                    "p_value": p_value,
                    "has_significant_trend": p_value < 0.05
                }
            }
        
        # Compare stability across diversity levels
        diversity_levels = sorted(cooperation_by_round_data.keys())
        variances = [stability_results[f"diversity_{d:.3f}"]["variance"] 
                    for d in diversity_levels 
                    if f"diversity_{d:.3f}" in stability_results]
        
        if len(diversity_levels) >= 2 and len(variances) >= 2:
            # Test if variance correlates with diversity
            var_corr, var_p = stats.pearsonr(diversity_levels[:len(variances)], variances)
            stability_correlation = {
                "variance_diversity_correlation": var_corr,
                "p_value": var_p,
                "interpretation": "Higher diversity " + 
                               ("increases" if var_corr > 0 else "decreases") + 
                               " cooperation volatility" if var_p < 0.05 else "No significant relationship"
            }
        else:
            stability_correlation = {"error": "Insufficient data for correlation"}
        
        return {
            "individual_stability": stability_results,
            "stability_correlation": stability_correlation
        }
    
    @staticmethod
    def _calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def _interpret_correlation(r: float) -> str:
        """Interpret correlation coefficient strength."""
        abs_r = abs(r)
        if abs_r < 0.1:
            strength = "negligible"
        elif abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.5:
            strength = "moderate"
        elif abs_r < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if r > 0 else "negative"
        return f"{strength} {direction} correlation"
    
    @staticmethod
    def _interpret_cohens_d(d: Optional[float]) -> str:
        """Interpret Cohen's d effect size."""
        if d is None:
            return "Cannot calculate"
        
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible effect"
        elif abs_d < 0.5:
            return "small effect"
        elif abs_d < 0.8:
            return "medium effect"
        else:
            return "large effect"