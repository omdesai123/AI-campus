# 🎓 AI Campus — Full ML-Powered Student Platform

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

## Project Structure

```
aicampus/
├── app.py                          # Flask backend — all 10 API endpoints
├── requirements.txt
├── models/
│   ├── performance.pkl             # PROVIDED — RandomForest pass/fail
│   ├── risk_model.pkl              # AUTO-TRAINED — RF risk classification
│   ├── stress_model.pkl            # AUTO-TRAINED — GBM stress prediction
│   ├── dropout_model.pkl           # AUTO-TRAINED — RF dropout prediction
│   ├── cluster_model.pkl           # AUTO-TRAINED — KMeans clustering
│   └── anomaly_model.pkl           # AUTO-TRAINED — Isolation Forest
├── static/
│   ├── css/main.css
│   └── js/main.js
└── templates/
    ├── base.html
    ├── index.html
    ├── login.html / signup.html
    ├── dashboard.html
    └── tool_*.html                 # 10 tool pages
```

## AI/ML Models

| # | Tool | Algorithm | Features | Output |
|---|------|-----------|----------|--------|
| 1 | Performance Predictor | RandomForest | attendance, study_hours, assignments, marks | Pass / Fail |
| 2 | Risk Classification | RandomForest | attendance, marks, behavior | High / Medium / Low |
| 3 | Attendance Analysis | Linear Regression | weekly history | Trend + 3-week forecast |
| 4 | Face Recognition | OpenCV Haar Cascade | webcam frame | Verified / Rejected |
| 5 | Anomaly Detection | Isolation Forest | attendance, marks, behavior, hours | Normal / Anomaly |
| 6 | Recommendations | Rule-based + ML scoring | all metrics | Personalised actions |
| 7 | Stress Predictor | Gradient Boosting | sleep, screen_time, activity | Low / Moderate / High |
| 8 | Chatbot | NLP Rule-based KB | free text | Answer + tips |
| 9 | Clustering | K-Means (k=3) | attendance, marks, assignments, hours | High / Average / At-Risk |
| 10 | Dropout Predictor | RandomForest | attendance, marks, assignments, hours, behavior | Risk + Intervention |

## API Endpoints

```
GET  /api/health                    → server status
POST /api/performance               → pass/fail prediction
POST /api/risk                      → risk classification
POST /api/attendance                → trend analysis
POST /api/faceid                    → face detection/verify
POST /api/anomaly                   → anomaly detection
POST /api/recommend                 → personalised recs
POST /api/stress                    → stress prediction
POST /api/chat                      → chatbot response
POST /api/cluster                   → student cluster
POST /api/dropout                   → dropout prediction
```

## Troubleshooting

- **"Network error"** → Ensure `python app.py` is running, visit `http://localhost:5000/api/health`
- **Model not found** → Ensure `models/` folder with all `.pkl` files is in the same directory as `app.py`
- **Webcam issues** → Allow camera permissions in browser; use HTTPS in production
- **sklearn version warning** → `pip install --upgrade scikit-learn`
