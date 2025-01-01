import unittest
from tinygrad import Tensor
from tinygrad.engine.schedule import create_schedule

class TestIntegration(unittest.TestCase):
    def test_complex_computation(self):
        # Test a more complex computation
        x = Tensor.randn(10, 10)
        y = Tensor.randn(10, 10)
        
        z = (x @ y).relu()
        w = z.sum(axis=0)
        
        schedule = create_schedule([w])
        
        # Run the computation
        result = w.numpy()
        
        # Verify shape and non-zero values
        self.assertEqual(result.shape, (10,))
        self.assertTrue(any(result > 0)) 