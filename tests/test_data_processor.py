"""
Unit tests for DataProcessor module
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processor import DataProcessor
from config import Config

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.processor = DataProcessor(self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'missing_col': [1, None, 3, None, 5]
        })
    
    def test_data_quality_analysis(self):
        """Test data quality analysis"""
        analysis = self.processor.analyze_data_quality(self.sample_data)
        
        self.assertIn('basic_info', analysis)
        self.assertIn('missing_data', analysis)
        self.assertIn('quality_score', analysis)
        self.assertEqual(analysis['basic_info']['rows'], 5)
        self.assertEqual(analysis['basic_info']['columns'], 3)
    
    def test_preprocessing(self):
        """Test data preprocessing"""
        processed_df = self.processor.preprocess_data(
            self.sample_data,
            handle_missing='fill_median',
            encode_categorical=True
        )
        
        # Check that missing values are handled
        self.assertEqual(processed_df.isnull().sum().sum(), 0)
        
        # Check that categorical variables are encoded
        self.assertTrue(processed_df['categorical_col'].dtype in ['int64', 'int32'])
    
    def test_create_data_profile(self):
        """Test data profiling"""
        profile = self.processor.create_data_profile(self.sample_data)
        
        self.assertIn('overview', profile)
        self.assertIn('features', profile)
        self.assertEqual(profile['overview']['dataset_size'], 5)

if __name__ == '__main__':
    unittest.main()
