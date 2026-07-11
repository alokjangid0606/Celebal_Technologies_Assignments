# 📄 AI Resume Screening System

An End-to-End AI Resume Screening System that evaluates resumes against job descriptions using Machine Learning, Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), and Streamlit.

---

## 🚀 Features

- 📄 Resume Upload (PDF)
- 🤖 Resume Category Classification
- 📊 ATS Score Calculation
- ✅ Skill Gap Analysis
- 🔍 Semantic Resume Retrieval using FAISS
- 🧠 Rule-Based AI Resume Feedback
- 📥 Download Resume Analysis Report
- 🌐 Interactive Streamlit Web Application

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Sentence Transformers
- FAISS
- Pandas
- NumPy
- Joblib
- PDFPlumber
- NLP
- Machine Learning

---

## 📂 Project Structure

```
AI-Resume-Screening-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── data/
│   └── final_feature_dataset.csv
│
├── models/
│   ├── resume_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── resume_index.faiss
│   └── resume_chunks.pkl
│
├── notebooks/
│   ├── 01_Data_Loading.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Resume_Classification.ipynb
│   ├── 06_Resume_Retrieval.ipynb
│   ├── 07_ATS_Scoring.ipynb
│   ├── 08_RAG.ipynb
│   ├── 09_AI_Resume_Analyzer.ipynb
│
└── outputs/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/AI-Resume-Screening-System.git
```

Move into the project folder

```bash
cd AI-Resume-Screening-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 🔄 Project Workflow

```
Resume PDF
        │
        ▼
Text Extraction
        │
        ▼
Text Preprocessing
        │
        ▼
TF-IDF Feature Extraction
        │
        ▼
Resume Classification
        │
        ▼
ATS Score Calculation
        │
        ▼
Skill Gap Analysis
        │
        ▼
Sentence Embeddings
        │
        ▼
FAISS Vector Search
        │
        ▼
Retrieved Resume Context
        │
        ▼
AI Resume Feedback
        │
        ▼
Final Resume Report
```

---

## 📊 Output

The application provides:

- Resume Category Prediction
- ATS Score
- Matched Skills
- Missing Skills
- Resume Strengths
- Resume Weaknesses
- AI Feedback
- Retrieved Resume Context
- Downloadable Resume Report

---

## 📸 Screenshots

Add screenshots of:

- Home Page
- Resume Upload
- ATS Score
- Skill Gap Analysis
- Resume Report

---

## 🔮 Future Improvements

- Gemini/OpenAI Integration
- Better ATS Scoring
- DOCX Resume Support
- Dashboard Analytics
- Resume Ranking
- Multiple Resume Comparison

---

## 👨‍💻 Author

**Alok Jangid**

B.Tech Computer Science & Artificial Intelligence

JECRC Foundation, Jaipur


---
