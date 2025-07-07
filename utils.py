import pandas as pd
import numpy as np

def load_sample_data():
    np.random.seed(42)
    data = {
        'open': np.random.rand(100) * 1.1,
        'high': np.random.rand(100) * 1.2,
        'low': np.random.rand(100) * 1.0,
        'close': np.random.rand(100) * 1.1,
        'volume': np.random.randint(100, 1000, size=100)
    }
    return pd.DataFrame(data)