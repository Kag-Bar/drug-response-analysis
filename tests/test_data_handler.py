import unittest
from unittest.mock import patch
from DataHandler import DataHandler
import os
import pandas as pd

class TestDataHandler(unittest.TestCase):

    def setUp(self):
        """Setup a config and inputs."""
        self.mock_gene_data = pd.DataFrame({
            'ID_REF': ['sample_1', 'sample_2'],
            'Gene1': [2.3, 2],
            'Gene2': [7.1, 5.1],
            'gene3': [9.6, 8.8],
        }).set_index('ID_REF').T
        self.mock_gene_data.to_csv("mocks/gene.csv")

        self.mock_metadata = pd.DataFrame({
            'SampleID': ["sample_1", "sample_2"],
            'Gender': ['Male', 'Female'],
            'das': [2.5, 3.1],
            'Response status': ['Responder', 'non_responder']
        })
        self.mock_metadata.to_csv("mocks/metadata.csv", index=False)

        self.config_path = os.path.abspath('mocks/mock_cfg_load_data.json')

    @patch('builtins.open')
    def test_init_invalid_config(self, mock_open):
        # Test invalid config path (e.g., file not found)
        invalid_config_path = 'tests/non_existent_config.json'
        mock_cfg = 'mocks/mock_cfg_TypeError'

        with self.assertRaises(ValueError):
            DataHandler(4)
        with self.assertRaises(FileNotFoundError):
            DataHandler(invalid_config_path)
        with self.assertRaises(TypeError):
            DataHandler(mock_cfg)

    def test_load_data(self):
        handler = DataHandler(self.config_path)
        self.assertIsNotNone(handler.data)
        self.assertEqual(len(handler.data), 2)
        self.assertIn('Gene1', handler.data.columns)
        self.assertIn('Gender', handler.data.columns)

    # @patch('builtins.open')
    # @patch('pandas.read_csv')
    def test_preprocess_metadata(self):
        # Test preprocessing of metadata (removing singular columns and renaming)
        handler = DataHandler(self.config_path)
        # After preprocessing
        self.assertNotIn('Response status', handler.metadata.columns)
        self.assertIn('y', handler.metadata.columns)
        self.assertEqual(handler.metadata['y'].iloc[0], 1)
        self.assertTrue(
            not any(isinstance(val, str) for val in handler.data['Gender']),
            "The 'Gender' column contains string values but should not."
        )

    def test_normalize_data(self):
        handler = DataHandler(self.config_path)
        # Before normalization
        original_mean_gene1 = self.mock_gene_data.T['Gene1'].mean()
        # After normalization
        normalized_mean_gene1 = handler.data['Gene1'].mean()

        self.assertNotEqual(original_mean_gene1, normalized_mean_gene1)

    def test_analyze_data(self):
        handler = DataHandler(self.config_path)
        top_correlations = handler.analyze_data()

        self.assertGreater(len(top_correlations), 0)


if __name__ == '__main__':
    unittest.main()
