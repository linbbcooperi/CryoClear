# RunPod configuration for CryoClear

Concrete settings to stand up the GPU box. Pairs with `docs/07_runpod_build_plan.md`
(the milestone plan) and `scripts/setup_pod.sh` (the bootstrap).

## 1. Deploy the pod
- **GPU:** RTX 4090 (24 GB) recommended — SAM ViT-H + U-Net fit in ~10 GB, 24 GB gives headroom.
  A5000/A40 also fine. (Venue RTX 2080 Ti, 11 GB, works but tighter.)
- **Template:** RunPod official **"RunPod PyTorch 2.x"** (CUDA 12.x, Ubuntu 22.04). It ships
  torch+CUDA so you don't fight driver/runtime mismatches.
- **Pod type:** Secure Cloud or Community Cloud. On-demand is fine for a 48 h hackathon.

## 2. Storage — this is the important one
- Create a **Network Volume** (~50–100 GB) in the **same region** as the pod and **mount it at
  `/workspace`**. Everything lives there: the repo, datasets, CryoSegNet + weights, the uv `.venv`.
- Why: the **container disk is wiped** when a pod is stopped/recreated; the Network Volume persists.
  Put the repo at `/workspace/CryoClear` and you can stop the pod overnight without re-downloading.
- Container disk: 20–30 GB is plenty (just the OS/CUDA image) since data is on the volume.

## 3. Expose ports
In the pod's template / "Edit Pod" → **Expose HTTP Ports**: add **8501** (Streamlit).
Under **Expose TCP Ports** keep **22** (SSH). RunPod gives each HTTP port a proxy URL:

```
https://<POD_ID>-8501.proxy.runpod.net
```

Streamlit **must bind 0.0.0.0** for the proxy to reach it — `make app` already does
(`--server.address 0.0.0.0 --server.port 8501`). The proxy terminates TLS, which is why
`.streamlit/config.toml` sets `enableXsrfProtection=false` / `enableCORS=false`.

## 4. First-boot setup
SSH in (or use the web terminal), then:
```bash
cd /workspace
git clone https://github.com/linbbcooperi/CryoClear.git
cd CryoClear
bash scripts/setup_pod.sh --cryosegnet      # uv env + tests + clone CryoSegNet + weights
```
`setup_pod.sh` installs uv, runs `uv sync`, runs the tests, and (with `--cryosegnet`) clones the
picker and pulls its weights. It then prints the exact data/pick/train/run commands.

## 5. CryoSegNet's conda env (separate from uv)
CryoClear uses **uv**; CryoSegNet needs its **own conda env** (torch + SAM). Keep them separate:
```bash
cd /workspace/CryoSegNet
conda env create -f environment.yml && conda activate cryosegnet
# install the torch CUDA build matching this pod from pytorch.org, then:
python predict_new_data_mrc.py --my_dataset_path <mics> --output_path output   # smoke test
```
You only use that env to **cache picks once** (`scripts/run_cryosegnet.py`); the app/metrics run
under uv and read the cached `.star`.

## 6. Environment variables / secrets
- **No secrets are required** to run CryoClear, and none belong in this repo (see `.gitignore`).
- If you want `gh`/private pushes from the pod, set `GH_TOKEN` as a **RunPod Secret / env var in the
  pod config** — never commit it. The HF token (if you use the Hub) likewise goes in the pod env,
  not the code.
- Optional convenience env vars: `RUNPOD_POD_ID` is set automatically (the setup script uses it to
  print your proxy URL).

## 7. Keeping the app running
The web terminal closes its processes when you leave. Run Streamlit so it survives:
```bash
nohup make app > /workspace/streamlit.log 2>&1 &      # or use tmux
```
Then open `https://<POD_ID>-8501.proxy.runpod.net`.

## 8. Optional: `runpodctl`
Install RunPod's CLI locally to manage pods / move files:
```bash
# send a file from the pod to your laptop without committing it
runpodctl send /workspace/CryoClear/data/processed/10017/junk_classifier.joblib
```
Handy for pulling the trained model or class-average montage off the pod for slides.
