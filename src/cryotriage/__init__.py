"""CryoTriage Live — real-time cryo-EM particle-picking + junk-removal copilot.

Submodules are imported lazily; only `metrics` and `features` depend solely on
numpy/scipy/scikit-learn. Heavy/optional deps (mrcfile, starfile, torch, topaz,
streamlit, aspire) are imported inside functions so `import cryotriage` is cheap.
"""

__version__ = "0.1.0"
__all__ = ["metrics", "features", "junk_classifier"]
