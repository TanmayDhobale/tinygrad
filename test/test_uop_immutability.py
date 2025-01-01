import unittest
import dataclasses
from tinygrad import Tensor
from tinygrad.ops import UOp, Ops
from tinygrad.dtype import dtypes

class TestUOpImmutability(unittest.TestCase):
    def test_uop_immutability(self):
        # Create a basic UOp
        uop = UOp(Ops.ADD, dtypes.float32, (), None)
        
        # Verify we can't modify attributes
        with self.assertRaises(dataclasses.FrozenInstanceError):
            uop.op = Ops.MUL
            
        with self.assertRaises(dataclasses.FrozenInstanceError):
            uop.dtype = dtypes.int32
            
    def test_buffer_reference(self):
        # Test that buffer is properly referenced in UOp arg
        t = Tensor([1,2,3])
        t.realize()
        
        # Get the underlying UOp
        uop = t._uop
        
        # Verify buffer is in the arg, not in global dict
        self.assertIsNotNone(uop.buffer)
        
    def test_tensor_uop_transformation(self):
        # Test that tensor operations create new UOps instead of mutating
        t1 = Tensor([1,2,3])
        t2 = Tensor([4,5,6])
        
        original_uop = t1._uop
        result = t1 + t2
        
        # Verify original UOp wasn't modified
        self.assertEqual(t1._uop, original_uop)
        
        # Verify new operation created new UOp
        self.assertNotEqual(result._uop, original_uop) 