import sys
import math


def get_tqdm():
    """Return tqdm that works for notebook or terminal."""
    try:
        # If running inside Jupyter/IPython kernel
        if "ipykernel" in sys.modules:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
    except ImportError:
        # fallback in case notebook extra is not installed
        from tqdm import tqdm
    return tqdm

def replace_nan_with_none(data):
    """
    Replace NaN or Inf values with None in a list of lists.
    
    Args:
        data (list[list]): Nested list of values.
    Returns:
        list[list]: Same structure but with NaN/Inf replaced by None.
    """
    cleaned = []
    for row in data:
        new_row = []
        for v in row:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                new_row.append(None)
            else:
                new_row.append(v)
        cleaned.append(new_row)
    return cleaned
