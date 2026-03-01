Intelligent Job Recommendation Engine

An end-to-end machine learning system that matches candidate resumes with relevant job postings using a hybrid approach of TF-IDF text similarity and supervised classification (Random Forest).

🚀 Project Overview

This project addresses the challenge of matching high volumes of resumes to specific job requirements. It processes raw resume and job data, extracts key skill features, and utilizes a supervised model to rank job opportunities for any given candidate.

📊 Datasets

The model is built by integrating two high-quality professional datasets:

1. Professional Resume Dataset

Source: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Content: A comprehensive collection of resumes categorized by job sector. This data provides the primary text for feature extraction and skill mapping.

Usage: Used to build candidate profiles and identify core competencies across various industries.

2. LinkedIn Job Postings Dataset

Source: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

Content: Detailed job listings from LinkedIn, including job descriptions, requirements, location, and company metadata.

Usage: Acts as the target database for the recommendation engine to find and rank matches based on resume analysis.

📂 Project Structure (Phases 1-3)

The development is divided into six logical steps:

01_data_exploration.ipynb: Initial EDA of both resume and LinkedIn job posting datasets. Performs basic cleaning and saves the base CSV files.

02_resume_parser.ipynb: Implements a Regex-based parser to extract structured information (contact, skills, education) from raw resume text/HTML.

03_feature_engineering.ipynb: Maps resumes and jobs into a high-dimensional binary feature space based on 35 standardized skill codes (e.g., IT, MGMT, ACCT).

04_hybrid_model_scoring.ipynb: Creates an initial "Unsupervised" ground truth by combining TF-IDF cosine similarity with skill-match matrices.

05_Supervised Models.ipynb: Trains and compares Logistic Regression and Random Forest models using the hybrid scores as labels. The Random Forest Classifier was identified as the best-performing model.

06_Job_recommendation_engine.ipynb: The final integration notebook that loads the saved model and generates a Top-10 ranked list of jobs for a target user.

🛠️ Technical Stack

Language: Python 3.x

Libraries: Pandas, NumPy, Scikit-learn, Scipy, Joblib, Pickle

Modeling: Random Forest Classifier (Supervised), TF-IDF / Cosine Similarity (Unsupervised)

Features: One-Hot Encoded Skill Matrices (35 unique skill categories)

🔧 Installation & Setup

Clone the Repository:

git clone [https://github.com/your-username/intelligent-job-recommendation.git](https://github.com/your-username/intelligent-job-recommendation.git)
cd intelligent-job-recommendation


Install Dependencies:

pip install pandas numpy scikit-learn scipy tabulate joblib


Data Requirements:
Ensure the following cleaned files are present in your data/ directory:

resume_skill_features.csv

job_skill_features.csv

postings_cleaned.csv

final_random_forest_model.pkl

🎯 Phase 4: Deployment (The Final Step)

The project is currently in the deployment phase. This involves moving the code from Jupyter Notebooks into a production-ready script or API.

Running the Recommendation Service

You can now generate recommendations via the command line using the provided recommendation_service.py:

# Generate recommendations for a specific Resume ID
python recommendation_service.py 2390


Future Roadmap

API Layer: Wrap the service in a FastAPI or Flask web server.

Real-time Parsing: Allow users to upload .pdf or .docx files directly for real-time skill extraction.

Frontend: Build a simple React or Streamlit dashboard to display job matches visually.

Developed as an end-to-end Machine Learning Pipeline.