import numpy as np
from defense.defenses import erosion_alpha

def test_erosion_basic():
    a = np.zeros((64,64), dtype=np.float32)
    cv = 1.0
    a[16:48,16:48] = cv
    e = erosion_alpha(a, radius=3)
    assert e.sum() < a.sum()
    assert e.max() <= 1.0 and e.min() >= 0.0
