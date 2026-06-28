# Demo samples

## `CryoClear_sample_betagal_EMPIAR-10017.mrc`
A real β-galactosidase micrograph (EMPIAR-10017, 4096×4096) for the **"bring your own
micrograph"** part of the demo.

**How to use it in the demo**
1. In the app, click **Upload MRC** (top toolbar).
2. Select this file. A progress bar runs while it uploads; then it appears in the
   micrograph dropdown as an uploaded micrograph.
3. The app picks it live (blob picker, ~700 candidates) and the β-gal-trained junk
   classifier flags keep (green) / junk (red) — no ground truth needed.
4. Switch the junk classifier (LightGBM / RandomForest / SGD) to show the purity change,
   or toggle **Show uncertain** to highlight the candidates the model is least sure about.
5. Export the kept coordinates (`.star` / `.box`) to show the clean stack you'd hand to
   RELION/cryoSPARC.

This is the strongest upload story because the classifier is trained on β-gal, so the
keep/junk calls are the most meaningful. (It's stored as int16 — visually identical after
the app's normalization — to keep the file ~33 MB instead of 64 MB.)
