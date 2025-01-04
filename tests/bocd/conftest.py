import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(43)
