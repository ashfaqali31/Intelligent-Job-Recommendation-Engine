import streamlit as st
import pandas as pd
import pickle
import re
import time

# --- 1. SETUP & MODEL LOADING ---
st.set_page_config(page_title="AI Job Matcher", layout="wide")

@st.cache_resource
def load_model():
    # Update this path to your actual .pkl location
    path = 'C:/Users/ashua/Desktop/Inelligent Job Recomendation Engine/data/Supervised Training/final_random_forest_model.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

rf_model = load_model()

SKILL_CODES = ['ACCT', 'ADM', 'ADVR', 'ANLS', 'ART', 'BD', 'CNST', 'DSGN', 'EDCN', 'ENG', 
               'FASH', 'FIN', 'GENB', 'HCPR', 'HR', 'IT', 'LGL', 'MGMT', 'MNFC', 'MRKT', 
               'OTHR', 'PR', 'PRJM', 'PROD', 'PRSR', 'QA', 'REAL', 'RSCH', 'SALE', 'SCI', 
               'SPRT', 'SUPL', 'TECH', 'TRNS', 'WRT']

ALL_FEATURE_COLS = [f'R_{c}' for c in SKILL_CODES] + [f'J_{c}' for c in SKILL_CODES]

# --- 2. LOGIC FUNCTIONS ---
def clean_text(text):
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', '', text)

def extract_features(text, skill_mapper, prefix='R'):
    cleaned = clean_text(text)
    found_codes = set()
    for code, keywords in skill_mapper.items():
        if any(kw in cleaned for kw in keywords):
            found_codes.add(code)
    data = {f'{prefix}_{c}': [1 if c in found_codes else 0] for c in SKILL_CODES}
    return pd.DataFrame(data), found_codes

# --- 3. UI DESIGN ---
st.title("🚀 Intelligent Job Recommendation Engine")
st.markdown("Check your resume's suitability against any job description instantly.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    resume_text = st.text_area("Paste Resume Content Here", height=300)

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste Job Description Here", height=300)

if st.button("Analyze Match"):
    if resume_text and jd_text:
        with st.spinner("Calculating Suitability..."):
            # Use your SKILL_MAPPER from File 06 here
            skill_mapper = {
                'IT': ['python', 'java', 'javascript', 'sql', 'machine learning'],
                'PRJM': ['agile', 'scrum', 'project management'],
                'MGMT': ['leadership', 'management', 'teamwork'],
                # Add your full list here...
            }
            
            # Process
            res_df, res_skills = extract_features(resume_text, skill_mapper, 'R')
            jd_df, jd_skills = extract_features(jd_text, skill_mapper, 'J')
            
            input_x = pd.concat([res_df, jd_df], axis=1).reindex(columns=ALL_FEATURE_COLS, fill_value=0)
            
            # Predict
            prob = rf_model.predict_proba(input_x)[:, 1][0]
            score = round(prob * 100, 2)
            
            # --- Results Display ---
            st.divider()
            st.balloons()
            
            m_col1, m_col2 = st.columns([1, 2])
            with m_col1:
                st.metric(label="Suitability Score", value=f"{score}%")
            
            with m_col2:
                if score > 80:
                    st.success("Strong Match! Your skills align well with this role.")
                elif score > 50:
                    st.warning("Moderate Match. Consider adding missing keywords.")
                else:
                    st.error("Low Match. This role might require different core skills.")
            
            # Skill Breakdown
            st.subheader("Skill Breakdown")
            b1, b2 = st.columns(2)
            b1.write("**Skills Found in Resume:**")
            b1.info(", ".join(res_skills) if res_skills else "No matching skills found.")
            
            b2.write("**Required Skills for Job:**")
            b2.info(", ".join(jd_skills) if jd_skills else "No specific skills detected.")
            
    else:
        st.warning("Please provide both a Resume and a Job Description.")