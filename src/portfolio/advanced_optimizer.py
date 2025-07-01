"""
Advanced portfolio optimizer for large-scale strategy and parameter optimization.
Supports genetic algorithms, grid search, Bayesian optimization, and ensemble methods.
"""

from __future__ import annotations

import itertools
import json
import logging
import multiprocessing as mp
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import warnings

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core.backtest_engine import UnifiedBacktestEngine as OptimizedBacktestEngine, BacktestConfig
from src.data_scraper.advanced_cache import advanced_cache

warnings.filterwarnings('ignore')


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""
    symbols: List[str]
    strategies: List[str]
    parameter_ranges: Dict[str, Dict[str, List]]  # strategy -> param -> range
    optimization_metric: str = 'sharpe_ratio'
    start_date: str = '2020-01-01'
    end_date: str = None  # Will default to today if None
    interval: str = '1d'
    initial_capital: float = 10000
    max_iterations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    early_stopping_patience: int = 20
    n_jobs: int = -1
    use_cache: bool = True
    constraint_functions: List[Callable] = None


@dataclass
class OptimizationResult:
    """Result from optimization run."""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    convergence_generation: int
    final_population: List[Dict[str, Any]]
    strategy: str
    symbol: str
    config: OptimizationConfig


class OptimizationMethod(ABC):
    """Abstract base class for optimization methods."""
    
    @abstractmethod
    def optimize(self, objective_function: Callable, config: OptimizationConfig) -> OptimizationResult:
        """Run optimization using this method."""
        pass


class GridSearchOptimizer(OptimizationMethod):
    """Grid search optimization - exhaustive search over parameter space."""
    
    def __init__(self, engine: OptimizedBacktestEngine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, objective_function: Callable, config: OptimizationConfig,
                symbol: str, strategy: str) -> OptimizationResult:
        """Run grid search optimization."""
        start_time = time.time()
        
        param_ranges = config.parameter_ranges.get(strategy, {})
        if not param_ranges:
            raise ValueError(f"No parameter ranges defined for strategy {strategy}")
        
        # Generate all parameter combinations
        param_combinations = self._generate_combinations(param_ranges)
        self.logger.info(f"Grid search: {len(param_combinations)} combinations for {symbol}/{strategy}")
        
        # Evaluate all combinations
        history = []
        best_score = float('-inf')
        best_params = None
        
        # Use parallel processing
        with ProcessPoolExecutor(max_workers=config.n_jobs if config.n_jobs > 0 else mp.cpu_count()) as executor:
            futures = {
                executor.submit(objective_function, symbol, strategy, params, config): params
                for params in param_combinations
            }
            
            for future in as_completed(futures):
                params = futures[future]
                try:
                    score = future.result()
                    history.append({'parameters': params, 'score': score, 'generation': 0})
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for {params}: {e}")
                    history.append({'parameters': params, 'score': float('-inf'), 'generation': 0, 'error': str(e)})
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=len(param_combinations),
            optimization_time=time.time() - start_time,
            convergence_generation=0,
            final_population=history,
            strategy=strategy,
            symbol=symbol,
            config=config
        )
    
    def _generate_combinations(self, param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations


class GeneticAlgorithmOptimizer(OptimizationMethod):
    """Genetic algorithm optimizer for parameter optimization."""
    
    def __init__(self, engine: OptimizedBacktestEngine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, objective_function: Callable, config: OptimizationConfig,
                symbol: str, strategy: str) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        start_time = time.time()
        
        param_ranges = config.parameter_ranges.get(strategy, {})
        if not param_ranges:
            raise ValueError(f"No parameter ranges defined for strategy {strategy}")
        
        self.logger.info(f"GA optimization for {symbol}/{strategy}: "
                        f"pop_size={config.population_size}, max_iter={config.max_iterations}")
        
        # Initialize population
        population = self._initialize_population(param_ranges, config.population_size)
        history = []
        best_score = float('-inf')
        best_params = None
        generations_without_improvement = 0
        convergence_generation = -1
        
        for generation in range(config.max_iterations):
            # Evaluate population
            scores = self._evaluate_population(population, objective_function, symbol, strategy, config)
            
            # Track best individual
            gen_best_idx = np.argmax(scores)
            gen_best_score = scores[gen_best_idx]
            gen_best_params = population[gen_best_idx]
            
            # Update global best
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_params = gen_best_params.copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Record generation statistics
            history.append({
                'generation': generation,
                'best_score': gen_best_score,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'best_parameters': gen_best_params
            })
            
            self.logger.info(f"Generation {generation}: best={gen_best_score:.4f}, "
                           f"mean={np.mean(scores):.4f}")
            
            # Early stopping
            if generations_without_improvement >= config.early_stopping_patience:
                convergence_generation = generation
                self.logger.info(f"Early stopping at generation {generation}")
                break
            
            # Create next generation
            if generation < config.max_iterations - 1:
                population = self._create_next_generation(population, scores, param_ranges, config)
        
        # Final population evaluation for reporting
        final_scores = self._evaluate_population(population, objective_function, symbol, strategy, config)
        final_population = [
            {'parameters': params, 'score': score}
            for params, score in zip(population, final_scores)
        ]
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=len(history) * config.population_size,
            optimization_time=time.time() - start_time,
            convergence_generation=convergence_generation,
            final_population=final_population,
            strategy=strategy,
            symbol=symbol,
            config=config
        )
    
    def _initialize_population(self, param_ranges: Dict[str, List], 
                             population_size: int) -> List[Dict[str, Any]]:
        """Initialize random population."""
        population = []
        for _ in range(population_size):
            individual = {}
            for param, values in param_ranges.items():
                if isinstance(values[0], (int, float)):
                    # Numeric parameter - sample from range
                    individual[param] = random.uniform(min(values), max(values))
                else:
                    # Categorical parameter - sample from list
                    individual[param] = random.choice(values)
            population.append(individual)
        return population
    
    def _evaluate_population(self, population: List[Dict[str, Any]], 
                           objective_function: Callable, symbol: str, strategy: str,
                           config: OptimizationConfig) -> List[float]:
        """Evaluate fitness of entire population."""
        with ProcessPoolExecutor(max_workers=config.n_jobs if config.n_jobs > 0 else mp.cpu_count()) as executor:
            futures = {
                executor.submit(objective_function, symbol, strategy, params, config): i
                for i, params in enumerate(population)
            }
            
            scores = [0.0] * len(population)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for individual {idx}: {e}")
                    scores[idx] = float('-inf')
        
        return scores
    
    def _create_next_generation(self, population: List[Dict[str, Any]], scores: List[float],
                              param_ranges: Dict[str, List], config: OptimizationConfig) -> List[Dict[str, Any]]:
        """Create next generation using selection, crossover, and mutation."""
        new_population = []
        
        # Elitism - keep best individuals
        elite_count = max(1, int(0.1 * len(population)))
        elite_indices = np.argsort(scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < len(population):
            # Tournament selection
            parent1 = self._tournament_selection(population, scores)
            parent2 = self._tournament_selection(population, scores)
            
            # Crossover
            if random.random() < config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, param_ranges)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < config.mutation_rate:
                child1 = self._mutate(child1, param_ranges)
            if random.random() < config.mutation_rate:
                child2 = self._mutate(child2, param_ranges)
            
            new_population.extend([child1, child2])
        
        return new_population[:len(population)]
    
    def _tournament_selection(self, population: List[Dict[str, Any]], 
                            scores: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection for parent selection."""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                  param_ranges: Dict[str, List]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uniform crossover between two parents."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param in param_ranges.keys():
            if random.random() < 0.5:
                child1[param], child2[param] = child2[param], child1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], 
               param_ranges: Dict[str, List], mutation_strength: float = 0.1) -> Dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        
        for param, values in param_ranges.items():
            if random.random() < 0.1:  # 10% chance to mutate each parameter
                if isinstance(values[0], (int, float)):
                    # Numeric parameter - add Gaussian noise
                    current_value = mutated[param]
                    range_size = max(values) - min(values)
                    noise = random.gauss(0, range_size * mutation_strength)
                    new_value = current_value + noise
                    mutated[param] = max(min(values), min(max(values), new_value))
                else:
                    # Categorical parameter - random choice
                    mutated[param] = random.choice(values)
        
        return mutated


class BayesianOptimizer(OptimizationMethod):
    """Bayesian optimization using Gaussian Processes."""
    
    def __init__(self, engine: OptimizedBacktestEngine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, objective_function: Callable, config: OptimizationConfig,
                symbol: str, strategy: str) -> OptimizationResult:
        """Run Bayesian optimization."""
        start_time = time.time()
        
        param_ranges = config.parameter_ranges.get(strategy, {})
        if not param_ranges:
            raise ValueError(f"No parameter ranges defined for strategy {strategy}")
        
        # Only support numeric parameters for now
        numeric_params = {k: v for k, v in param_ranges.items() 
                         if isinstance(v[0], (int, float))}
        
        if not numeric_params:
            self.logger.warning(f"No numeric parameters found for {strategy}, falling back to grid search")
            return GridSearchOptimizer(self.engine).optimize(objective_function, config, symbol, strategy)
        
        self.logger.info(f"Bayesian optimization for {symbol}/{strategy}: "
                        f"max_iter={config.max_iterations}")
        
        # Initialize with random samples
        n_initial = min(10, config.max_iterations // 2)
        X_sample = []
        y_sample = []
        history = []
        
        # Random initialization
        for i in range(n_initial):
            params = self._sample_random_params(numeric_params)
            score = objective_function(symbol, strategy, params, config)
            
            X_sample.append(list(params.values()))
            y_sample.append(score)
            history.append({'parameters': params, 'score': score, 'iteration': i, 'type': 'random'})
        
        X_sample = np.array(X_sample)
        y_sample = np.array(y_sample)
        
        # Gaussian Process model
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        best_score = np.max(y_sample)
        best_params = history[np.argmax(y_sample)]['parameters']
        
        # Bayesian optimization loop
        for iteration in range(n_initial, config.max_iterations):
            # Fit GP model
            gp.fit(X_sample, y_sample)
            
            # Find next point using acquisition function
            next_params = self._optimize_acquisition(gp, numeric_params, best_score)
            next_x = np.array([list(next_params.values())])
            
            # Evaluate objective
            score = objective_function(symbol, strategy, next_params, config)
            
            # Update data
            X_sample = np.vstack([X_sample, next_x])
            y_sample = np.append(y_sample, score)
            history.append({'parameters': next_params, 'score': score, 'iteration': iteration, 'type': 'bayes'})
            
            # Update best
            if score > best_score:
                best_score = score
                best_params = next_params
            
            self.logger.info(f"Iteration {iteration}: score={score:.4f}, best={best_score:.4f}")
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=config.max_iterations,
            optimization_time=time.time() - start_time,
            convergence_generation=-1,
            final_population=[],
            strategy=strategy,
            symbol=symbol,
            config=config
        )
    
    def _sample_random_params(self, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Sample random parameters from ranges."""
        params = {}
        for param, values in param_ranges.items():
            params[param] = random.uniform(min(values), max(values))
        return params
    
    def _optimize_acquisition(self, gp: GaussianProcessRegressor, param_ranges: Dict[str, List],
                            current_best: float) -> Dict[str, Any]:
        """Optimize acquisition function to find next point."""
        bounds = [(min(values), max(values)) for values in param_ranges.values()]
        param_names = list(param_ranges.keys())
        
        def acquisition_function(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            # Expected Improvement
            improvement = mu - current_best
            Z = improvement / (sigma + 1e-9)
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei[0]  # Minimize negative EI
        
        # Multiple random starts for optimization
        best_x = None
        best_ei = float('inf')
        
        for _ in range(10):
            x0 = [random.uniform(bound[0], bound[1]) for bound in bounds]
            result = optimize.minimize(acquisition_function, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.fun < best_ei:
                best_ei = result.fun
                best_x = result.x
        
        return dict(zip(param_names, best_x))


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimizer supporting multiple optimization methods
    and large-scale parameter optimization across thousands of assets.
    """
    
    def __init__(self, engine: OptimizedBacktestEngine = None):
        self.engine = engine or OptimizedBacktestEngine()
        self.logger = logging.getLogger(__name__)
        
        # Available optimization methods
        self.optimizers = {
            'grid_search': GridSearchOptimizer(self.engine),
            'genetic_algorithm': GeneticAlgorithmOptimizer(self.engine),
            'bayesian': BayesianOptimizer(self.engine),
        }
    
    def optimize_portfolio(self, config: OptimizationConfig, 
                          method: str = 'genetic_algorithm') -> Dict[str, Dict[str, OptimizationResult]]:
        """
        Optimize entire portfolio of symbols and strategies.
        
        Args:
            config: Optimization configuration
            method: Optimization method to use
            
        Returns:
            Nested dictionary: {symbol: {strategy: OptimizationResult}}
        """
        if method not in self.optimizers:
            raise ValueError(f"Unknown optimization method: {method}")
        
        start_time = time.time()
        self.logger.info(f"Portfolio optimization: {len(config.symbols)} symbols, "
                        f"{len(config.strategies)} strategies, method={method}")
        
        results = {}
        total_combinations = len(config.symbols) * len(config.strategies)
        completed = 0
        
        for symbol in config.symbols:
            results[symbol] = {}
            
            for strategy in config.strategies:
                self.logger.info(f"Optimizing {symbol}/{strategy} ({completed+1}/{total_combinations})")
                
                try:
                    # Check cache first
                    cache_key = self._get_optimization_cache_key(symbol, strategy, config, method)
                    cached_result = advanced_cache.get_optimization_result(symbol, strategy, cache_key, config.interval)
                    
                    if cached_result and config.use_cache:
                        self.logger.info(f"Using cached optimization for {symbol}/{strategy}")
                        results[symbol][strategy] = self._dict_to_optimization_result(cached_result)
                    else:
                        # Run optimization
                        optimizer = self.optimizers[method]
                        result = optimizer.optimize(self._objective_function, config, symbol, strategy)
                        results[symbol][strategy] = result
                        
                        # Cache result
                        if config.use_cache:
                            advanced_cache.cache_optimization_result(
                                symbol, strategy, cache_key, asdict(result), config.interval
                            )
                    
                    completed += 1
                    
                except Exception as e:
                    self.logger.error(f"Optimization failed for {symbol}/{strategy}: {e}")
                    results[symbol][strategy] = OptimizationResult(
                        best_parameters={},
                        best_score=float('-inf'),
                        optimization_history=[],
                        total_evaluations=0,
                        optimization_time=0,
                        convergence_generation=-1,
                        final_population=[],
                        strategy=strategy,
                        symbol=symbol,
                        config=config
                    )
                    completed += 1
        
        total_time = time.time() - start_time
        self.logger.info(f"Portfolio optimization completed in {total_time:.2f}s")
        
        return results
    
    def optimize_single_strategy(self, symbol: str, strategy: str, config: OptimizationConfig,
                               method: str = 'genetic_algorithm') -> OptimizationResult:
        """Optimize a single symbol/strategy combination."""
        if method not in self.optimizers:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimizer = self.optimizers[method]
        return optimizer.optimize(self._objective_function, config, symbol, strategy)
    
    def _objective_function(self, symbol: str, strategy: str, parameters: Dict[str, Any],
                          config: OptimizationConfig) -> float:
        """Objective function for optimization."""
        try:
            # Create backtest config
            backtest_config = BacktestConfig(
                symbols=[symbol],
                strategies=[strategy],
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=config.initial_capital,
                interval=config.interval,
                use_cache=config.use_cache,
                save_trades=False,
                save_equity_curve=False
            )
            
            # Run backtest with custom parameters
            result = self.engine._run_single_backtest(symbol, strategy, backtest_config, None, parameters)
            
            if result.error:
                return float('-inf')
            
            # Apply constraint functions
            if config.constraint_functions:
                for constraint_func in config.constraint_functions:
                    if not constraint_func(result.metrics, parameters):
                        return float('-inf')
            
            # Return optimization metric
            return result.metrics.get(config.optimization_metric, float('-inf'))
            
        except Exception as e:
            self.logger.warning(f"Objective function failed for {symbol}/{strategy}: {e}")
            return float('-inf')
    
    def _get_optimization_cache_key(self, symbol: str, strategy: str, 
                                  config: OptimizationConfig, method: str) -> Dict[str, Any]:
        """Generate cache key for optimization result."""
        return {
            'method': method,
            'parameter_ranges': config.parameter_ranges.get(strategy, {}),
            'optimization_metric': config.optimization_metric,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'interval': config.interval,
            'max_iterations': config.max_iterations,
            'population_size': config.population_size if method == 'genetic_algorithm' else None
        }
    
    def _dict_to_optimization_result(self, cached_dict: Dict) -> OptimizationResult:
        """Convert cached dictionary to OptimizationResult object."""
        return OptimizationResult(
            best_parameters=cached_dict.get('best_parameters', {}),
            best_score=cached_dict.get('best_score', float('-inf')),
            optimization_history=cached_dict.get('optimization_history', []),
            total_evaluations=cached_dict.get('total_evaluations', 0),
            optimization_time=cached_dict.get('optimization_time', 0),
            convergence_generation=cached_dict.get('convergence_generation', -1),
            final_population=cached_dict.get('final_population', []),
            strategy=cached_dict.get('strategy', ''),
            symbol=cached_dict.get('symbol', ''),
            config=OptimizationConfig(**cached_dict.get('config', {}))
        )
    
    def create_ensemble_strategy(self, optimization_results: Dict[str, Dict[str, OptimizationResult]],
                               top_n: int = 5) -> Dict[str, Any]:
        """
        Create ensemble strategy from optimization results.
        
        Args:
            optimization_results: Results from portfolio optimization
            top_n: Number of top strategies to include in ensemble
            
        Returns:
            Ensemble strategy configuration
        """
        # Flatten results and sort by score
        all_results = []
        for symbol, strategies in optimization_results.items():
            for strategy, result in strategies.items():
                if result.best_score > float('-inf'):
                    all_results.append({
                        'symbol': symbol,
                        'strategy': strategy,
                        'score': result.best_score,
                        'parameters': result.best_parameters
                    })
        
        # Sort by score and take top N
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_strategies = all_results[:top_n]
        
        # Calculate weights based on scores
        scores = [r['score'] for r in top_strategies]
        min_score = min(scores)
        adjusted_scores = [s - min_score + 1 for s in scores]  # Ensure positive weights
        total_score = sum(adjusted_scores)
        weights = [s / total_score for s in adjusted_scores]
        
        ensemble_config = {
            'strategies': top_strategies,
            'weights': weights,
            'creation_date': time.time(),
            'total_score': sum(scores),
            'diversity_score': len(set(r['strategy'] for r in top_strategies))
        }
        
        return ensemble_config
    
    def get_optimization_summary(self, results: Dict[str, Dict[str, OptimizationResult]]) -> Dict[str, Any]:
        """Generate summary statistics from optimization results."""
        all_scores = []
        strategy_performance = defaultdict(list)
        symbol_performance = defaultdict(list)
        
        for symbol, strategies in results.items():
            for strategy, result in strategies.items():
                if result.best_score > float('-inf'):
                    all_scores.append(result.best_score)
                    strategy_performance[strategy].append(result.best_score)
                    symbol_performance[symbol].append(result.best_score)
        
        summary = {
            'total_optimizations': sum(len(strategies) for strategies in results.values()),
            'successful_optimizations': len(all_scores),
            'overall_stats': {
                'mean_score': np.mean(all_scores) if all_scores else 0,
                'std_score': np.std(all_scores) if all_scores else 0,
                'min_score': np.min(all_scores) if all_scores else 0,
                'max_score': np.max(all_scores) if all_scores else 0,
                'median_score': np.median(all_scores) if all_scores else 0
            },
            'strategy_stats': {
                strategy: {
                    'count': len(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'best_score': np.max(scores)
                }
                for strategy, scores in strategy_performance.items()
            },
            'symbol_stats': {
                symbol: {
                    'count': len(scores),
                    'mean_score': np.mean(scores),
                    'best_strategy': max(results[symbol].items(), key=lambda x: x[1].best_score)[0]
                }
                for symbol, scores in symbol_performance.items()
            }
        }
        
        return summary


# Import for Bayesian optimization
try:
    from scipy.stats import norm
except ImportError:
    # Fallback implementation
    class norm:
        @staticmethod
        def cdf(x):
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        @staticmethod
        def pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
