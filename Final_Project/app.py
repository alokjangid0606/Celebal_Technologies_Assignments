# ============================================================
# AI Resume Screening System
# Part 1 : UI + Model Loading + PDF Reader
# ============================================================

import streamlit as st
import pdfplumber
import joblib
import pickle
import faiss
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# Load ML Models
# ============================================================

@st.cache_resource
def load_models():

    classifier = joblib.load("models/resume_classifier.pkl")

    tfidf = joblib.load("models/tfidf_vectorizer.pkl")

    label_encoder = joblib.load("models/label_encoder.pkl")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    index = faiss.read_index(
        "models/resume_index.faiss"
    )

    with open(
        "models/resume_chunks.pkl",
        "rb"
    ) as file:

        resume_chunks = pickle.load(file)

    return (
        classifier,
        tfidf,
        label_encoder,
        embedding_model,
        index,
        resume_chunks
    )


(
    classifier,
    tfidf,
    label_encoder,
    embedding_model,
    index,
    resume_chunks
) = load_models()


# ============================================================
# PDF Reader
# ============================================================

def extract_text_from_pdf(uploaded_file):

    text = ""

    with pdfplumber.open(uploaded_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text


# ============================================================
# Feature Extraction
# ============================================================

def extract_features(text):

    resume_length = len(text)

    word_count = len(text.split())

    sentence_count = text.count(".")

    skill_count = text.count(",")

    education_length = len(
        text.lower().split("education")
    ) if "education" in text.lower() else 1

    experience_length = len(
        text.lower().split("experience")
    ) if "experience" in text.lower() else 1

    return np.array([[
        resume_length,
        word_count,
        sentence_count,
        skill_count,
        education_length,
        experience_length
    ]])


# ============================================================
# Header
# ============================================================

st.title("📄 AI Resume Screening System")

st.markdown("""
Analyze resumes using Machine Learning, NLP, ATS Scoring and RAG.
""")


# ============================================================
# Layout
# ============================================================

left, right = st.columns([2,1])


with left:

    uploaded_resume = st.file_uploader(

        "Upload Resume",

        type=["pdf"]

    )

    job_description = st.text_area(

        "Paste Job Description",

        height=250

    )

    analyze = st.button(

        "Analyze Resume",

        use_container_width=True

    )


with right:

    st.info("""

### Features

✅ Resume Classification

✅ ATS Score

✅ Skill Gap

✅ RAG Retrieval

✅ AI Feedback

""")


# ============================================================
# Analyze
# ============================================================

if analyze:

    if uploaded_resume is None:

        st.warning("Please upload a PDF Resume.")

        st.stop()

    if job_description.strip() == "":

        st.warning("Please paste Job Description.")

        st.stop()

    resume_text = extract_text_from_pdf(uploaded_resume)

    st.success("Resume Uploaded Successfully")

    st.subheader("Extracted Resume")

    st.write(resume_text[:1500])
    # ============================================================
    # Resume Category Prediction
    # ============================================================

    resume_vector = tfidf.transform([resume_text])

    extra_features = extract_features(resume_text)

    final_vector = hstack([
        resume_vector,
        extra_features
    ])

    prediction = classifier.predict(final_vector)

    predicted_category = label_encoder.inverse_transform(
        prediction
    )[0]

    st.markdown("---")

    st.subheader("🎯 Predicted Resume Category")

    st.success(predicted_category)


    # ============================================================
    # Resume Skills
    # ============================================================

    common_skills = [
        "python","java","sql","mysql","mongodb","c","c++",
        "javascript","html","css","react","angular","node",
        "machine learning","deep learning","tensorflow","keras",
        "pandas","numpy","matplotlib","seaborn","scikit-learn",
        "git","github","docker","kubernetes","aws","azure",
        "gcp","linux","flask","django","fastapi","power bi",
        "tableau","excel","rest api","spring","spring boot"
    ]

    resume_lower = resume_text.lower()

    resume_skills = []

    for skill in common_skills:

        if skill in resume_lower:

            resume_skills.append(skill)


    # ============================================================
    # Job Description Skills
    # ============================================================

    jd_lower = job_description.lower()

    jd_skills = []

    for skill in common_skills:

        if skill in jd_lower:

            jd_skills.append(skill)


    # ============================================================
    # ATS Score
    # ============================================================

    matched_skills = list(
        set(resume_skills).intersection(jd_skills)
    )

    missing_skills = list(
        set(jd_skills) - set(resume_skills)
    )

    if len(jd_skills) > 0:

        ats_score = (
            len(matched_skills)
            /
            len(jd_skills)
        ) * 100

    else:

        ats_score = 0


    # ============================================================
    # ATS Score Display
    # ============================================================

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "ATS Score",
            f"{ats_score:.2f}%"
        )

    with c2:

        st.metric(
            "Matched Skills",
            len(matched_skills)
        )

    with c3:

        st.metric(
            "Missing Skills",
            len(missing_skills)
        )


    # ============================================================
    # Skill Display
    # ============================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("✅ Matched Skills")

        if matched_skills:

            for skill in matched_skills:

                st.success(skill)

        else:

            st.warning("No skills matched.")


    with col2:

        st.subheader("❌ Missing Skills")

        if missing_skills:

            for skill in missing_skills:

                st.error(skill)

        else:

            st.success("No missing skills.")
            # ============================================================
    # RAG Retrieval using FAISS
    # ============================================================

    query_embedding = embedding_model.encode(
        [job_description],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        k=2
    )

    retrieved_context = ""

    for idx in indices[0]:

        retrieved_context += resume_chunks[idx] + "\n\n"


    # ============================================================
    # Resume Strengths
    # ============================================================

    strengths = []

    if ats_score >= 70:
        strengths.append("Excellent ATS compatibility.")

    elif ats_score >= 50:
        strengths.append("Good ATS compatibility.")

    else:
        strengths.append("Resume needs ATS improvement.")

    if len(matched_skills) >= 5:
        strengths.append("Strong technical skill set.")

    elif len(matched_skills) >= 3:
        strengths.append("Relevant technical skills available.")

    if len(resume_text) > 1500:
        strengths.append("Resume contains detailed information.")


    # ============================================================
    # Resume Weaknesses
    # ============================================================

    weaknesses = []

    if ats_score < 70:
        weaknesses.append("ATS score is below recommended level.")

    if len(missing_skills) > 0:
        weaknesses.append("Some required skills are missing.")

    if len(resume_text) < 800:
        weaknesses.append("Resume content is too short.")


    # ============================================================
    # Rule-Based AI Feedback
    # ============================================================

    feedback = []

    if ats_score >= 80:

        feedback.append(
            "Excellent resume. Your profile is highly suitable for this role."
        )

    elif ats_score >= 60:

        feedback.append(
            "Good resume. Improve missing skills to increase your chances."
        )

    else:

        feedback.append(
            "Resume needs improvement before applying for this position."
        )

    if missing_skills:

        feedback.append(
            "Recommended skills to learn: "
            + ", ".join(missing_skills)
        )

    feedback.append(
        "Add more projects and measurable achievements."
    )


    # ============================================================
    # Display Analysis
    # ============================================================

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("💪 Strengths")

        for item in strengths:

            st.success(item)


    with right:

        st.subheader("⚠ Weaknesses")

        for item in weaknesses:

            st.error(item)


    st.markdown("---")

    st.subheader("🤖 AI Resume Feedback")

    for item in feedback:

        st.info(item)


    st.markdown("---")

    st.subheader("📄 Retrieved Resume Context (RAG)")

    st.write(retrieved_context[:1200])
    # ============================================================
    # Final Report
    # ============================================================

    st.markdown("---")

    report = f"""
    ==============================
    AI RESUME ANALYSIS REPORT
    ==============================
    
    Predicted Category:
    {predicted_category}
    
    ATS Score:
    {ats_score:.2f} %
    
    Matched Skills:
    {', '.join(matched_skills)}
    
    Missing Skills:
    {', '.join(missing_skills)}
    
    Strengths:
    {chr(10).join(strengths)}
    
    Weaknesses:
    {chr(10).join(weaknesses)}
    
    AI Feedback:
    {chr(10).join(feedback)}
    
    Retrieved Resume Context:
    {retrieved_context}
    """

    st.download_button(

        label="📥 Download Resume Report",

        data=report,

        file_name="Resume_Report.txt",

        mime="text/plain"

    )


    # ============================================================
    # Save Report
    # ============================================================

    with open(
        "outputs/Resume_Report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)


    # ============================================================
    # Footer
    # ============================================================

    st.markdown("---")

    st.success("Resume Analysis Completed Successfully.")

    st.caption(
        "AI Resume Screening System | Built using Machine Learning, NLP, RAG and Streamlit"
    )
