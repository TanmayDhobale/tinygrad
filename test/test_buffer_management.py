import unittest
from tinygrad import Tensor
from tinygrad.engine.schedule import create_schedule

class TestBufferManagement(unittest.TestCase):
    def test_buffer_lifecycle(self):
        # Create tensor
        t1 = Tensor([1,2,3])
        t1.realize()
        
        # Verify buffer exists
        self.assertIsNotNone(t1._uop.buffer)
        
        # Create new tensor reusing buffer
        t2 = t1 + 1
        schedule = create_schedule([t2])
        
        # Verify buffer properly managed
        self.assertIsNotNone(t2._uop.buffer)
        
    def test_buffer_reuse(self):
        # Test that buffers are properly reused
        t1 = Tensor([1,2,3])
        t2 = t1 * 2
        t3 = t2 + 1
        
        schedule = create_schedule([t3])
        
        # Count unique buffers
        buffers = set()
        for item in schedule:
            for buf in item.bufs:
                buffers.add(buf)
                
        # Should reuse buffers efficiently
        self.assertLess(len(buffers), 3) 