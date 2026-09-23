# 🌿 Plant Disease Detection Using Leaf Images

An end-to-end deep learning project that identifies plant diseases from leaf
images using transfer learning (MobileNetV2), served via a FastAPI backend
and a Streamlit web frontend, with prediction history stored in SQLite.

---

## 📌 Project Description

Farmers and gardeners often struggle to identify plant diseases early,
leading to crop loss. This project uses a Convolutional Neural Network (CNN),
built with transfer learning on **MobileNetV2**, trained on the
**PlantVillage dataset** (38 classes of healthy and diseased leaves across
14 crop species) to automatically detect plant diseases from a leaf photo
and suggest a remedy.

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Dataset | PlantVillage (Kaggle), 38 classes |
| Model | TensorFlow / Keras, MobileNetV2 (transfer learning) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Database | SQLite |
| Deployment | Streamlit Community Cloud / Hugging Face Spaces |

## 📁 Project Structure

```
plant_disease_detection/
├── dataset/
│   └── download_dataset.py      # downloads + splits PlantVillage dataset
├── model/
│   ├── data_utils.py            # tf.data pipeline + augmentation
│   ├── build_model.py           # MobileNetV2 transfer-learning model
│   └── train.py                 # training script (configurable)
├── backend/
│   ├── main.py                  # FastAPI app (/predict, /history)
│   ├── database.py              # SQLite helper
│   └── remedies.json            # disease -> remedy lookup
├── frontend/
│   └── app.py                   # Streamlit web app
├── saved_model/                 # trained model + class_names.json (generated)
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Setup Steps

### 1. Clone and install dependencies
```bash
git clone <your-repo-url>
cd plant_disease_detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset
- Get your Kaggle API token (`kaggle.json`) from your Kaggle account
  settings and place it at `~/.kaggle/kaggle.json`.
```bash
cd dataset
python download_dataset.py
```
This creates `data/split/{train,val,test}/<class_name>/*.jpg`.

### 3. Train the model
```bash
cd ../model
python train.py --epochs 15 --batch_size 32 --lr 0.0001
```
This saves:
- `saved_model/plant_disease_model.keras` — the trained model
- `saved_model/class_names.json` — class index → label mapping
- `saved_model/training_history.png` — accuracy/loss plots
- `saved_model/test_results.json` — final test accuracy/loss

### 4. Run the backend API (optional, for REST API access)
```bash
cd ../backend
uvicorn main:app --reload --port 8000
```
Test it at `http://localhost:8000/docs` (Swagger UI).

### 5. Run the frontend app
```bash
cd ../frontend
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 🚀 Deployment

### Push code to GitHub
```bash
git init
git add .
git commit -m "Initial commit: plant disease detection project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
> Note: `saved_model/plant_disease_model.keras` can be large. If it's over
> 100 MB, use [Git LFS](https://git-lfs.github.com/) (`git lfs track "*.keras"`)
> before committing it.

### Option A: Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, select your repo, branch `main`.
3. Set the **main file path** to `frontend/app.py`.
4. Under **Advanced settings**, make sure the Python version matches
   your `requirements.txt` (Python 3.10/3.11 recommended for TensorFlow).
5. Click **Deploy**. Streamlit installs `requirements.txt` and launches the app.

### Option B: Hugging Face Spaces (free, alternative)
1. Go to https://huggingface.co/new-space
2. Choose **Streamlit** as the Space SDK, name your Space.
3. Either:
   - Push your GitHub repo to the Space's git remote:
     ```bash
     git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
     git push hf main
     ```
   - Or upload files directly via the Hugging Face web UI.
4. Make sure `frontend/app.py` is set as the app's entry file
   (Hugging Face Streamlit Spaces looks for `app.py` in the repo root —
   you may need to copy/rename `frontend/app.py` to the repo root, or set
   `app_file: frontend/app.py` in the Space's `README.md` metadata block).
5. The Space builds automatically from `requirements.txt` and deploys.

---

## 📸 Screenshots

*(Add screenshots here after running the app)*

- `screenshots/predict_page.png` — Upload & prediction screen
- `screenshots/history_page.png` — Prediction history table

---

## 📄 License

This project is for academic/educational purposes.

## check live status
https://plantcare-ai-app.streamlit.app/
