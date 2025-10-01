"""
Unit tests for MLModule
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ml_module import MLModule
from config import Config

class TestMLModule(unittest.TestCase):
    """Test cases for MLModule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.ml_module = MLModule(self.config)
        
        # Create sample dataset
        np.random.seed(42)
        n_samples = 100
        
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(2, 1, n_samples),
            'feature3': np.random.randint(0, 3, n_samples)
        })
        
        # Classification target
        self.y_classification = pd.Series(np.random.choice([0, 1], n_samples))
        
        # Regression target
        self.y_regression = pd.Series(
            self.X['feature1'] * 2 + self.X['feature2'] * 3 + np.random.normal(0, 0.5, n_samples)
        )
    
    def test_task_type_detection(self):
        """Test automatic task type detection"""
        # Test classification detection
        task_type_class = self.ml_module._detect_task_type(self.y_classification)
        self.assertEqual(task_type_class, 'classification')
        
        # Test regression detection
        task_type_reg = self.ml_module._detect_task_type(self.y_regression)
        self.assertEqual(task_type_reg, 'regression')
    
    def test_auto_ml_pipeline_classification(self):
        """Test Auto-ML pipeline for classification"""
        results = self.ml_module.auto_ml_pipeline(
            self.X, self.y_classification, 
            task_type='classification',
            test_size=0.3
        )
        
        self.assertIn('best_model', results)
        self.assertIn('all_models', results)
        self.assertIn('test_metrics', results)
        self.assertIn('accuracy', results['test_metrics'])
    
    def test_auto_ml_pipeline_regression(self):
        """Test Auto-ML pipeline for regression"""
        results = self.ml_module.auto_ml_pipeline(
            self.X, self.y_regression,
            task_type='regression',
            test_size=0.3
        )
        
        self.assertIn('best_model', results)
        self.assertIn('all_models', results)
        self.assertIn('test_metrics', results)
        self.assertIn('r2', results['test_metrics'])
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        results = self.ml_module.auto_ml_pipeline(
            self.X, self.y_classification,
            task_type='classification'
        )
        
        self.assertIn('feature_importance', results)
        importance = results['feature_importance']
        self.assertEqual(len(importance), len(self.X.columns))

if __name__ == '__main__':
    unittest.main()
