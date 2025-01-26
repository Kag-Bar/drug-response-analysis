import unittest
import torch
from SimpleClassificationNetwork import SimpleClassificationNetwork

class Config:
    input_size = 100  # Example feature input size
    num_classes = 2   # Binary classification

class TestSimpleClassificationNetwork(unittest.TestCase):

    def setUp(self):
        """Setup a sample model for testing."""
        self.model = SimpleClassificationNetwork(Config.input_size, Config.num_classes)

    def test_forward_pass(self):
        """Test the forward pass of the neural network."""
        input_data = torch.randn([Config.input_size, Config.input_size])  # Example input data shape
        output = self.model(input_data)
        self.assertEqual(output.shape, (Config.input_size, 2))  # Assuming binary classification output shape (batch_size, 2)

    def test_model_initialization(self):
        """Test if the model is properly initialized."""
        # Check if the model contains expected layers
        self.assertTrue(hasattr(self.model, 'model'))  # Check that the 'model' attribute exists
        self.assertTrue(len(self.model.model) > 0)  # Check that there are layers in the model
        self.assertTrue(isinstance(self.model.model[0], torch.nn.Linear))  # Check that the first layer is Linear
        self.assertTrue(isinstance(self.model.model[-1], torch.nn.Linear))  # Check that the last layer is Linear
        self.assertTrue(hasattr(self.model, 'softmax')) # Check if the softmax layer exists

if __name__ == "__main__":
    unittest.main()
