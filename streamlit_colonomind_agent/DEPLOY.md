# Deploying the ColonoMind app (clean, no Colab)

The Streamlit app runs in a container built from `requirements.txt`, so none of
the Colab dependency problems happen here. You need to give it 3 things it can't
get from the repo:

- `deployment_report.json`  (small — metrics + class names + model registry)
- `preprocess_artifacts.joblib`  (small–medium — scaler, UMAP, label encoder)
- `weights/*.weights.h5`  (large — one per trained model, ~280 MB each)

All of these were produced on your Google Drive at
`MyDrive/ColonoMind/streamlit_colonomind_agent/`. Download them from Drive to
your computer first.

---

## Option A — Hugging Face Spaces  ★ recommended

Free CPU Spaces get ~16 GB RAM (Streamlit Community Cloud only ~1 GB, which is
tight for TensorFlow + these models), and a Space can hold the weights directly.

1. Create a Space: https://huggingface.co/new-space → SDK = **Streamlit**.
2. In the Space's **Files** tab, upload:
   - `app.py`
   - `requirements.txt`
   - `deployment_report.json`
   - `preprocess_artifacts.joblib`
   - a `weights/` folder containing your `*.weights.h5` files
     (drag the whole folder, or use `git lfs` — HF handles large files natively)
3. The Space builds and starts automatically. Done — you get a permanent URL.

No `ASSET_BASE_URL` needed here because the files live in the Space.

---

## Option B — Streamlit Community Cloud

Repo file-size limits mean the big weights can't live in the repo, so host them
on a **GitHub Release** and let the app download them at startup.

1. **Host the weights.** In this GitHub repo → **Releases** → **Draft a new
   release** → tag e.g. `weights-v1` → attach your `*.weights.h5` files → publish.
   Each file's URL will be:
   `https://github.com/colonoscopyai-agent/CNN-Comparison-in-Colonoscopy/releases/download/weights-v1/<file>.weights.h5`
2. **Commit the two small files** to this folder (via GitHub *Add file → Upload*):
   `deployment_report.json` and `preprocess_artifacts.joblib`.
   (They're under GitHub's 100 MB limit; if `preprocess_artifacts.joblib` is
   bigger, attach it to the same Release and it'll be downloaded too.)
3. Deploy: https://share.streamlit.io → **New app** → this repo/branch →
   **Main file path** = `streamlit_colonomind_agent/app.py`.
4. In the app's **Settings → Secrets**, add:
   ```toml
   ASSET_BASE_URL = "https://github.com/colonoscopyai-agent/CNN-Comparison-in-Colonoscopy/releases/download/weights-v1"
   ```
   The app downloads any missing `weights/<file>` (and the metadata files if you
   didn't commit them) from that base URL on first use.

> ⚠️ Community Cloud's ~1 GB RAM may not fit TensorFlow + a 280 MB hybrid model.
> If the app crashes with a memory error, use **Option A (Hugging Face Spaces)**.

---

## How the app finds its files

`app.py` calls `ensure_asset(path)` for each required file:
1. if the file is already next to `app.py` (committed / uploaded), it's used as-is;
2. otherwise it's downloaded from `ASSET_BASE_URL/<path>`.

So Option A (everything uploaded) needs no URL, and Option B (weights on a
Release) just needs `ASSET_BASE_URL` in secrets.

Research/educational tool — not a clinical diagnosis.
