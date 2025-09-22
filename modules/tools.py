import sys


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