# 🦠 COVID-19 Epidemic Spread Prediction
**CodeCure AI Hackathon — Track C**

## 🌐 Live Demo
👉 **[Click here to open the dashboard](https://covid19-epidemic-prediction-gtw5j8mfyfenj97kafj5or.streamlit.app/)**

## 📌 Overview
A machine learning pipeline that predicts COVID-19 epidemic spread and visualizes outbreak risk across countries using multi-source epidemiological data.

## 🚀 Features
* Prophet 30-day case forecasting per country
* Per-country risk classification (High / Medium / Low)
* Interactive Folium risk map with heatmap layer
* Multi-source data pipeline (OWID + JHU + Vaccination + Testing)
* Streamlit dashboard with 5 interactive tabs
* Random Forest feature importance analysis

## 🗂️ Project Structure
├── prediction.py       # ML pipeline — run this first
├── dashboard.py        # Streamlit dashboard — run this second
├── ml_risk_map.html    # Pre-generated interactive risk map
├── requirements.txt    # All dependencies
└── README.md

## ⚙️ How to Run
1. Install dependencies
pip install -r requirements.txt

2. Run ML pipeline first
python prediction.py

3. Launch dashboard
python -m streamlit run dashboard.py

## 📥 Data Sources
* Our World in Data (OWID) — COVID-19 cases & deaths
* Johns Hopkins CSSE — Geographic coordinates
* Our World in Data — Vaccination coverage
* Our World in Data — Testing rates

> Note: All datasets are automatically fetched at runtime via Google Drive.
> No manual download required — just open the live demo link above.

## 🛠️ Tech Stack
Python · Streamlit · Prophet · Scikit-learn · Plotly · Folium · Pandas · NumPy · OpenPyXL

## 📊 Models Used
| Model | Purpose |
|---|---|
| Facebook Prophet | 30-day case forecasting |
| Linear Regression | 14-day per-country prediction |
| Random Forest | Feature importance analysis |

## 🔗 Repository
https://github.com/shadow12546-sketch/covid19-epidemic-prediction
