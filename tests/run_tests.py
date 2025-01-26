import unittest

def run_tests():
    # Create a test suite
    suite = unittest.TestSuite()

    suite.addTest(unittest.defaultTestLoader.loadTestsFromName('tests.test_requirements.TestRequirements'))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromName('tests.test_data_handler.TestDataHandler'))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromName('tests.test_feature_extractor.TestFeatureExtractor'))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromName('tests.test_simple_classification_network.TestSimpleClassificationNetwork'))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromName('tests.test_model_trainer.TestModelTrainer'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    run_tests()
