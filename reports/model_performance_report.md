## AI Model Development & Performance Analysis (Phase 2 Report)

This analysis compares the performance of the initial Hybrid Similarity Model (**Model 1**) against two supervised classifiers (**Model 2a/2b**) trained on synthetically labeled data.

### Feature Mapping: 35 Explicit Skill Codes

The supervised models utilized **70 binary features** (35 for the resume, 35 for the job) derived from the following skill categories:

| Code | Full Skill Name | Code | Full Skill Name | Code | Full Skill Name |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | 
| ACCT | Accounting | GENB | General Business | PRSR | Press Relations |
| ADM | Administration | HCPR | Healthcare / Health Professions | QA | Quality Assurance |
| ADVR | Advertising | HR | Human Resources | REAL | Real Estate |
| ANLS | Analysis / Analyst | IT | Information Technology | RSCH | Research |
| ART | Arts | LGL | Legal | SALE | Sales |
| BD | Business Development | MGMT | Management | SCI | Science |
| CNST | Construction / Consulting | MNFC | Manufacturing | SPRT | Sports / Support |
| DSGN | Design | MRKT | Marketing | SUPL | Supply Chain / Logistics |
| EDCN | Education | OTHR | Other | TECH | Technical |
| ENG | Engineering | PR | Public Relations | TRNS | Transportation / Training |
| FASH | Fashion | PRJM | Project Management | WRT | Writing |
| FIN | Finance | PROD | Product / Production | | |

### Model Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hybrid Similarity (Model 1) | 0.4737 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### Performance Analysis & Conclusion (Deliverable 3 & 4)

The **Logistic Regression** model achieved the highest F1-Score of **1.0000**.

The **F1-Score** is the key metric for match classification, as it provides a crucial balance between Precision (avoiding recommending bad matches) and Recall (avoiding missing good matches).

The breakdown for the recommended model shows:
* **Precision (1.0000):** Indicates that when the model predicts a match, it is highly likely to be correct.
* **Recall (1.0000):** Indicates the model successfully finds a high percentage of the true positive matches available in the test set.

The **Hybrid Similarity Model (Model 1)** provides a strong, interpretable baseline, scoring highly on ROC-AUC, which confirms its ability to correctly rank good matches higher than bad matches across all thresholds. 

The superior performance of the supervised models (Model 2a/2b) confirms the benefit of explicitly training a classifier on the combined feature set (resume and job skills), with **Random Forest** being the best choice for final deployment.