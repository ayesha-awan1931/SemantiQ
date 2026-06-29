# 🔮 NLP Semantic Similarity Explorer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-all--MiniLM--L6--v2-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-Apache_2.0-green)
![Model](https://img.shields.io/badge/Model-sentence--transformers-blue)

**A premium AI-powered web application that computes semantic similarity between texts using state-of-the-art sentence embeddings — no preprocessing, no training, no paid APIs.**

</div>

---

## 📖 Project Description

**NLP Semantic Similarity Explorer** is a full-featured, production-quality Streamlit application built for a BS Artificial Intelligence quiz. It allows users to enter words, sentences, or paragraphs and instantly measure how semantically similar they are using the free, pretrained `sentence-transformers/all-MiniLM-L6-v2` model from HuggingFace.

The app features a premium dark glassmorphism UI with animated gradient backgrounds, floating blobs, interactive Plotly charts, Paul's Critical Thinking Standards analysis, session history, CSV export, and much more.

---

## ✨ Features

### 🧠 Core NLP
- **Semantic Similarity** via cosine similarity on dense 384-dim embeddings
- **Zero Preprocessing** — raw text is fed directly to the model
- **Pairwise Analysis** — compare up to 10 texts simultaneously
- **Top-K Results** — ranked list of most similar texts

### 📊 Visualisations (Plotly Interactive)
| Chart | Description |
|-------|-------------|
| Animated Bar Chart | Top-K similarity scores with grow animation |
| Interactive Heatmap | Full N×N pairwise similarity matrix |
| 2D PCA Scatter | Embedding space projection with hover labels |
| Radar Chart | Paul's 7 Intellectual Standards scores |
| Gauge Chart | Single score displayed as a speedometer |
| Distribution Chart | Histogram of all pairwise scores |
| Score Trend | Cross-session score trend line |

### 🎨 UI Design
- **Dark Glassmorphism** theme with backdrop-filter blur
- **Animated gradient background** (CSS keyframe animation)
- **Floating glowing blobs** (CSS pseudo-elements)
- **Fade-in / slide-up** card animations
- **Shimmer animated dividers**
- **Custom CSS** progress bars, buttons, inputs, sidebar
- **Google Fonts**: Inter, Space Grotesk, JetBrains Mono
- **Fully responsive** layout

### 🛠️ Extra Features
- 📜 Session History (last 50 runs)
- 📥 Download Results as CSV
- 📋 Copy embedding vector to clipboard
- 🔄 Reset button
- 📈 Performance statistics (runs, avg inference time)
- 🧪 Paul's 7 Critical Thinking Standards (expandable cards + radar)
- ℹ️ About & Model info page with full quiz checklist

---

## 🤖 Model Used

| Property | Value |
|----------|-------|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Provider** | sentence-transformers / HuggingFace |
| **License** | Apache 2.0 (Free) |
| **Embedding Dim** | 384 |
| **Max Tokens** | 256 |
| **Size** | ~80 MB |
| **Cost** | FREE — no API key required |

The model is downloaded automatically on first run and cached locally.

---

## 🗂️ Project Structure

```
NLP_Similarity_App/
│
├── app.py              ← Main Streamlit entry point (run this)
├── similarity.py       ← Model loading, embeddings, cosine similarity
├── graphs.py           ← All Plotly interactive charts
├── components.py       ← Reusable HTML/CSS UI components
├── utils.py            ← Session state, CSV export, Paul's scores
├── style.css           ← Global glassmorphism dark theme CSS
├── requirements.txt    ← Pinned Python dependencies
├── README.md           ← This file
└── assets/
    └── screenshots/    ← App screenshots (add your own)
```

---

## ⚙️ Installation

### Prerequisites
- Python **3.10 or higher**
- pip
- (Optional) Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/NLP_Similarity_App.git
cd NLP_Similarity_App
```

### Step 2 — Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The first install downloads PyTorch (~700 MB). Subsequent runs use the local cache.

---

## 🚀 Run Commands

### Local Development
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501**

### Custom Port
```bash
streamlit run app.py --server.port 8080
```

### Headless (Server / CI)
```bash
streamlit run app.py --server.headless true
```

---

## ☁️ Deployment

### Streamlit Community Cloud (Recommended — Free)

1. Push your code to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repository, branch, and set **main file path** to `app.py`.
4. Click **"Deploy!"** — Streamlit Cloud installs requirements automatically.
5. Your app is live at `https://YOUR_APP.streamlit.app`

> **Tip:** Streamlit Cloud uses CPU-only PyTorch. The app is optimised for this and works perfectly.

### GitHub Actions (CI/CD)
No additional configuration needed — Streamlit Cloud handles deployment on every push to `main`.

---

## 📸 Screenshots

> Add screenshots to `assets/screenshots/` and reference them here.

```markdown
![Main Page](assets/screenshots/main.png)
![Heatmap](assets/screenshots/heatmap.png)
![PCA Plot](assets/screenshots/pca.png)
![Critical Thinking](assets/screenshots/critical.png)
```

---

## 🐙 GitHub Instructions

### First-time Setup
```bash
git init
git add .
git commit -m "Initial commit: NLP Similarity Explorer"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/NLP_Similarity_App.git
git push -u origin main
```

### Push Updates
```bash
git add .
git commit -m "Your update message"
git push
```

### Recommended `.gitignore`
```
__pycache__/
*.py[cod]
.env
venv/
.venv/
*.log
.streamlit/secrets.toml
```

---

## 📐 How It Works

```
User Input (raw text)
        ↓
SentenceTransformer.encode()
        ↓
384-dim L2-normalised embedding vectors
        ↓
Cosine Similarity = dot(vec_a, vec_b)  [since ||v||=1]
        ↓
Pairwise Matrix  →  Heatmap
Top-K Ranking    →  Bar Chart
PCA(2D)          →  Scatter Plot
Paul's Scores    →  Radar Chart
```

No tokenization, stemming, lemmatization, or preprocessing is performed by the application code. The SentenceTransformer model handles all internal processing.

---

## 🎓 Academic Compliance

| Requirement | Implemented |
|-------------|-------------|
| One FREE pretrained NLP model | ✅ |
| No preprocessing | ✅ |
| No tokenization | ✅ |
| No stemming / lemmatization | ✅ |
| No model training | ✅ |
| No paid APIs | ✅ |
| Cosine similarity | ✅ |
| Similarity Score display | ✅ |
| Top Similar Results | ✅ |
| Embedding vectors | ✅ |
| Animated Bar Chart | ✅ |
| Interactive Heatmap | ✅ |
| 2D PCA Plot | ✅ |
| Radar Chart | ✅ |
| Gauge Chart | ✅ |
| Distribution Chart | ✅ |
| Paul's 7 Standards | ✅ |
| Glassmorphism UI | ✅ |
| Dark theme | ✅ |
| Animated background | ✅ |
| Download CSV | ✅ |
| Session History | ✅ |
| Sidebar navigation | ✅ |
| Footer | ✅ |
| PEP-8 code quality | ✅ |
| Modular structure | ✅ |

---

## 📄 License

Apache License 2.0 — same as the model used.

---

<div align="center">
Built with ❤️ using <a href="https://streamlit.io">Streamlit</a>,
<a href="https://www.sbert.net">SentenceTransformers</a>, and
<a href="https://plotly.com">Plotly</a>
</div>
