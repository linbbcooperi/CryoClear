# Roles & Ownership

## Bindu — CS (the brain + the app)
- Picking-engine integration (`picker.py`): run Topaz/CryoSegNet, parse boxes.
- **Junk classifier** (`junk_classifier.py` + `features.py`) — the novel ML piece. Start with a scikit-learn RandomForest on CryoPPP keep/junk labels.
- Active-learning loop (`active_learning.py`): update from user corrections.
- UI (`app/streamlit_app.py`): image canvas, boxes, click-to-reject, live metrics.
- Metrics (`metrics.py`, already done): precision/recall/F1 vs CryoPPP.

## Tony — hardware / systems (make it real-time and reliable)
- Environment + GPU setup on the venue workstation (CUDA/PyTorch); Docker image.
- **MRC I/O + preprocessing** (`io_mrc.py`): normalize, downsample to 8-bit, tiling.
- **Stream simulator** (`stream.py`): feed micrographs on a timer to mimic a live session; throughput dashboard (micrographs/min, GPU use).
- Performance: batching, caching, keep per-micrograph inference to a few seconds.

## Eva — chemistry (the science, the user, the story)
- Choose the demo protein(s); define real particle vs ice/carbon/aggregate.
- Be the **human-in-the-loop** in the demo (the corrections that improve the model).
- Run/interpret **2D class averages** (ASPIRE); confirm they look like real protein.
- Lead the scientific framing + slides + "why this matters for structure determination."

> This split nails the "professional-background variety" judging criterion: hardware + chemistry + CS, each owning a layer.
