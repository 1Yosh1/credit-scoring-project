# 🏦 AI Credit Risk Scoring System (MLOps Pipeline)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 📋 Executive Summary
This project is an end-to-end **Machine Learning Operations (MLOps) system** designed to automate credit risk assessment for financial institutions. 

Using historical loan application data, I built an XGBoost model to predict the probability of client default. The model is deployed via a **FastAPI microservice** and served through an interactive **Streamlit Dashboard**, all containerized using **Docker** for consistent deployment.

**Key Business Value:**
* **Reduced Risk:** Identifies high-risk applicants with >85% accuracy (AUC).
* **Automation:** Replaces manual underwriting with instant API predictions.
* **Scalability:** Microservices architecture allows independent scaling of the API and Dashboard.

---

## 🏗️ System Architecture
The application runs as a multi-container Docker application:

1.  **The "Brain" (API Container):** * Runs **FastAPI** to serve the XGBoost model.
    * Handles **Smart Imputation**: Automatically infers missing credit scores based on available data to prevent false positives.
    * Implements **Business Logic Rules**: Applies a dynamic risk threshold (20%) to flag risky loans.

2.  **The "Face" (Dashboard Container):** * Runs **Streamlit** to provide a user-friendly interface for loan officers.
    * Communicates with the API via a private Docker network.

---

## 🛠️ Tech Stack
| Component | Tools Used |
| :--- | :--- |
| **Modeling** | Python, XGBoost, Scikit-Learn, Pandas |
| **API Backend** | FastAPI, Pydantic (Data Validation) |
| **Frontend** | Streamlit (Interactive Dashboard) |
| **DevOps** | Docker, Docker Compose |
| **Version Control** | Git, GitHub |

---

## 🚀 How to Run Locally (Docker)
You can spin up the entire system with one command. Ensure you have Docker Desktop installed.

**1. Clone the Repository**
```bash
git clone [https://github.com/1Yosh1/credit-scoring-project.git](https://github.com/1Yosh1/credit-scoring-project.git)
cd credit-scoring-project