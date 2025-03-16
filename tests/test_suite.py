import unittest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from tests.data_scraper.test_data_loader import TestDataLoader
from tests.backtesting_engine.test_engine import TestBacktestEngine
from tests.backtesting_engine.strategies.test_strategies import TestStrategies
from tests.backtesting_engine.strategies.test_strategy_factory import TestStrategyFactory
from tests.optimizer.test_parameter_tuner import TestParameterTuner
from tests.reports.test_report_generator import TestReportGenerator
from tests.portfolio.test_portfolio_analyzer import TestPortfolioAnalyzer
from tests.portfolio.test_parameter_optimizer import TestParameterOptimizer
from tests.portfolio.test_metrics_processor import TestMetricsProcessor
from tests.cli.config.test_config_loader import TestConfigLoader
from tests.cli.test_cli import TestCLI
from tests.integration.test_workflow import TestWorkflow

def create_test_suite():
    """Create a test suite containing all tests."""
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestBacktestEngine))
    test_suite.addTest(unittest.makeSuite(TestStrategies))
    test_suite.addTest(unittest.makeSuite(TestStrategyFactory))
    test_suite.addTest(unittest.makeSuite(TestParameterTuner))
    test_suite.addTest(unittest.makeSuite(TestReportGenerator))
    test_suite.addTest(unittest.makeSuite(TestPortfolioAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestParameterOptimizer))
    test_suite.addTest(unittest.makeSuite(TestMetricsProcessor))
    test_suite.addTest(unittest.makeSuite(TestConfigLoader))
    test_suite.addTest(unittest.makeSuite(TestCLI))
    test_suite.addTest(unittest.makeSuite(TestWorkflow))
    
    return test_suite

if __name__ == '__main__':
    # Create the test suite
    suite = create_test_suite()
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
