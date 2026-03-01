import streamlit as st
import pandas as pd
import pickle
import re
import pdfplumber
from pathlib import Path

# --- 1. CONFIGURATION & ASSETS ---
st.set_page_config(page_title="AI Job Matcher", layout="wide")

# This list MUST be in the exact order of your training features
SKILL_CODES = ['ACCT', 'ADM', 'ADVR', 'ANLS', 'ART', 'BD', 'CNST', 'DSGN', 'EDCN', 'ENG', 
               'FASH', 'FIN', 'GENB', 'HCPR', 'HR', 'IT', 'LGL', 'MGMT', 'MNFC', 'MRKT', 
               'OTHR', 'PR', 'PRJM', 'PROD', 'PRSR', 'QA', 'REAL', 'RSCH', 'SALE', 'SCI', 
               'SPRT', 'SUPL', 'TECH', 'TRNS', 'WRT']

# Mapping of all 35 Categories to Keywords
SKILL_MAPPER = {
    'ACCT': ['accounting', 'audit', 'tax', 'ledger', 'reconciliation', 'cpa', 'billing'],
    'ADM': ['administration', 'office', 'clerical', 'data entry', 'receptionist', 'filing'],
    'ADVR': ['advertising', 'media planning', 'campaigns', 'copywriting', 'ad strategy'],
    'ANLS': ['data analysis', 'statistics', 'power bi', 'tableau', 'excel', 'modeling', 'insights', 'pandas', 'numpy', 'scikit-learn', 'r programming'],
    'ART': ['graphic design', 'illustration', 'creative direction', 'fine arts', 'visuals'],
    'BD': ['business development', 'partnerships', 'prospecting', 'growth', 'networking'],
    'CNST': ['construction', 'civil engineering', 'building', 'safety', 'site management'],
    'DSGN': ['ui ux', 'product design', 'figma', 'sketch', 'adobe xd', 'prototyping'],
    'EDCN': ['education', 'teaching', 'training', 'curriculum', 'mentoring', 'pedagogy'],
    'ENG': ['engineering', 'mechanical', 'electrical', 'structural', 'cad', 'blueprints'],
    'FASH': ['fashion', 'apparel', 'textiles', 'merchandising', 'styling', 'garment'],
    'FIN': ['finance', 'investment', 'banking', 'portfolio', 'equity', 'trading', 'valuation'],
    'GENB': ['general business', 'operations', 'entrepreneurship', 'commerce', 'business admin'],
    'HCPR': ['healthcare', 'medical', 'patient care', 'clinical', 'nursing', 'diagnosis'],
    'HR': ['human resources', 'recruitment', 'payroll', 'onboarding', 'employee relations'],
    'IT': ['python', 'java', 'sql', 'tensorflow', 'pytorch', 'aws', 'cloud', 'software', 'machine learning', 'deep learning', 'neural networks'],
    'LGL': ['legal', 'law', 'contract', 'paralegal', 'compliance', 'litigation', 'attorney'],
    'MGMT': ['leadership', 'management', 'agile', 'scrum', 'strategy', 'decision making'],
    'MNFC': ['manufacturing', 'production line', 'assembly', 'quality control', 'lean'],
    'MRKT': ['seo', 'sem', 'branding', 'marketing', 'social media', 'content strategy', 'advertising', 'market research'],
    'OTHR': ['general', 'miscellaneous', 'other'],
    'PR': ['public relations', 'press release', 'communications', 'media relations'],
    'PRJM': ['project management', 'pmp', 'milestone', 'resource planning', 'gantt'],
    'PROD': ['product management', 'roadmap', 'user stories', 'product lifecycle'],
    'PRSR': ['customer service', 'hospitality', 'support', 'client relations'],
    'QA': ['quality assurance', 'testing', 'automation', 'manual testing', 'bugs'],
    'REAL': ['real estate', 'property', 'leasing', 'mortgage', 'broker', 'housing'],
    'RSCH': ['research', 'market research', 'survey', 'investigation', 'methodology'],
    'SALE': ['sales', 'crm', 'account management', 'negotiation', 'closing', 'leads'],
    'SCI': ['science', 'laboratory', 'biology', 'chemistry', 'physics', 'biotech'],
    'SPRT': ['sports', 'fitness', 'coaching', 'athletics', 'physical education'],
    'SUPL': ['supply chain', 'logistics', 'inventory', 'procurement', 'shipping'],
    'TECH': ['technical support', 'troubleshooting', 'hardware', 'it infrastructure'],
    'TRNS': ['transportation', 'logistics', 'fleet', 'delivery', 'supply chain management'],
    'WRT': ['writing', 'editing', 'content creation', 'blogging', 'journalism']
}

# Ensure the feature names match the Model's expectations (70 total)
ALL_FEATURE_COLS = [f'R_{c}' for c in SKILL_CODES] + [f'J_{c}' for c in SKILL_CODES]

# --- 2. HELPER FUNCTIONS ---

@st.cache_resource
def load_model():
    # Adjusted path based on your folder structure
    model_path = Path(__file__).resolve().parent.parent / "data" / "Supervised Training" / "final_random_forest_model.pkl"
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() or "" for page in pdf.pages])

def clean_text(text):
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', '', text)

def extract_features(text, prefix):
    cleaned = clean_text(text)
    found_categories = []
    for category, keywords in SKILL_MAPPER.items():
        if any(kw in cleaned for kw in keywords):
            found_categories.append(category)
    
    vector = {f'{prefix}_{code}': [1 if code in found_categories else 0] for code in SKILL_CODES}
    return pd.DataFrame(vector), found_categories

# --- 3. UI DESIGN ---

st.title("🚀 Intelligent Job Recommendation Engine")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("📄 Candidate Resume")
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    resume_text = ""
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("Resume text extracted!")

with col2:
    st.header("💼 Job Description")
    jd_text = st.text_area("Paste the job requirements here...", height=250)

# --- 4. MATCHING LOGIC ---

if st.button("🔍 Analyze Match Probability"):
    if resume_text and jd_text:
        try:
            model = load_model()
            
            res_df, res_cats = extract_features(resume_text, 'R')
            jd_df, jd_cats = extract_features(jd_text, 'J')
            
            input_data = pd.concat([res_df, jd_df], axis=1)
            input_data = input_data.reindex(columns=ALL_FEATURE_COLS, fill_value=0)
            
            probability = model.predict_proba(input_data)[:, 1][0]
            final_score = probability * 100
            
            # --- GUARDRAIL FIX ---
            overlap = set(res_cats).intersection(set(jd_cats))
            
            if not overlap:
                final_score = final_score * 0.10 # Severe penalty for 0% category overlap
                st.error("🚨 Critical Domain Mismatch!")
            elif len(overlap) < 2:
                final_score = final_score * 0.50 # Moderate penalty
            
            st.divider()
            st.metric(label="Refined Match Score", value=f"{round(final_score, 1)}%")

            if final_score > 70:
                st.success("✅ Strong Candidate Match!")
            else:
                st.warning("⚖️ Partial or Weak Match.")

            with st.expander("Detailed Comparison"):
                st.write(f"**Resume Categories:** {', '.join(res_cats)}")
                st.write(f"**JD Categories:** {', '.join(jd_cats)}")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
    else:
        st.warning("Inputs missing!")