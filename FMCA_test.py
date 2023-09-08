import FMCA
import utils as ut
import pytest
import numpy as np
import pandas as pd
import warnings
class TestUtils:
    def test_safe_cast_good_input(self):
        assert True == np.equal(ut.safe_cast([1,2,3]), np.array([1,2,3]))[0]
        
    def test_safe_cast_bad_input(self):
        with pytest.raises(RuntimeError) as exc:
            ut.safe_cast({'a' : 1})
        print(str(exc.value))
        assert str(exc.value) == "Data cannot be cast to numpy array."
    
    
    def test_set_labels_good_input(self):
        assert True == np.equal(ut.set_labels([2, 3]), np.array([0,1]))[0]
    
    
    def test_set_labels_bad_input(self):
        with pytest.raises(RuntimeError) as exc:
            ut.set_labels(["A", "B", "C"])
        assert str(exc.value) == "Only two-class datasets are currently supported"
    
    
