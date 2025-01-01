import unittest
from tinygrad import Tensor
from tinygrad.engine.schedule import create_schedule

class TestScheduleImmutability(unittest.TestCase):
    def test_schedule_preserves_relationships(self):
        # Create a simple computation graph
        t1 = Tensor([1,2,3])
        t2 = Tensor([4,5,6])
        t3 = t1 + t2
        t4 = t3 * 2
        
        # Get original UOps
        original_uops = {
            't1': t1._uop,
            't2': t2._uop,
            't3': t3._uop,
            't4': t4._uop
        }
        
        # Create schedule
        schedule = create_schedule([t4])
        
        # Verify tensors still point to correct transformed UOps
        self.assertNotEqual(t4._uop, original_uops['t4'])
        self.assertTrue(hasattr(t4._uop, 'buffer'))
        
    def test_multiple_tensor_references(self):
        # Test case where multiple tensors reference same UOp
        t1 = Tensor([1,2,3])
        t2 = t1.reshape(3,1)
        t3 = t1.reshape(1,3)
        
        # Both reshapes should reference same base UOp
        self.assertEqual(t2._uop.base, t3._uop.base)
        
        schedule = create_schedule([t2, t3])
        
        # Verify relationships preserved after scheduling
        self.assertEqual(t2._uop.base, t3._uop.base) 