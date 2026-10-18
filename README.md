# SentinelAI — Hackathon Prototype

SentinelAI is a Streamlit prototype aligned with the concept note **“Artificial Intelligence-Based Real-Time Intrusion Detection Using Session and Network Behavioural Analysis.”**

## Features in this build
- Royal purple-and-gold dashboard inspired by the SentinelAI reference mockup, plus severity colour grading
- Demo events, overview metrics, severity distribution, and risk gauge
- Event Explorer with filters and CSV export
- Session grouping when a `session_id` field is available
- CSV ingestion and schema profiling
- Random Forest baseline for labelled numeric data
- Isolation Forest anomaly scoring
- Feature importance, hold-out metrics, and confusion matrix
- Risk scoring, severity mapping, human-readable explanations, alert queue
- CSV exports

## Windows setup
Open Command Prompt or the VS Code terminal in this folder.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If Streamlit is already installed, the first command updates/installs the project dependencies. It is okay to keep using `python -m streamlit` if the plain `streamlit` command is not recognised.

## Data
Start with the demo simulation, or upload a labelled CSV from **Data & Ingestion**. Public dataset examples mentioned in the concept note include UNSW-NB15, CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019, ToN-IoT, BoT-IoT, Edge-IIoTset, and CICIoT2023.

Dataset column names vary. The app tries to normalise a few common column aliases, but the training baseline expects numeric feature columns and a usable label column. Review your dataset schema before training.

## Important limitations
- Demo events are synthetic and are not real attacks or live traffic.
- Uploaded CSV analysis is batch analysis, not continuous real-time monitoring.
- Random Forest detects classes represented in the training data; it does not guarantee zero-day detection.
- Isolation Forest flags deviations from the fitted baseline; anomalies are not automatically malicious.
- Session analysis groups by an existing `session_id`. Reconstructing sessions from network events requires additional telemetry and logic.
- Risk scores are heuristic triage values, not calibrated probabilities.
- This prototype does not block IP addresses or send alerts to a real website owner.
- Validate on held-out data and, ideally, a separate attack family/dataset. Report precision, recall, F1 and confusion matrix, not accuracy alone.

## Suggested next implementation steps
1. Confirm the exact columns in the chosen public dataset and build a dataset-specific preprocessing adapter.
2. Implement session reconstruction and sequence-level correlation.
3. Add a controlled event simulator to show new events arriving over time.
4. Add an authenticated webhook/email notification integration only after consent, secure configuration, and testing.
5. Evaluate on an excluded attack family or independent dataset to measure generalisation.


## Royal purple dashboard refresh
The Overview page is redesigned with a midnight-purple background, lavender/purple highlights, gold headings, activity chart, recent alerts, an assistant panel, and system status. Existing Event Explorer, Session Analysis, AI Detection Lab, Alerts & Response, Data & Ingestion, and About workflows are retained. The dashboard assistant provides simple summaries from the current in-session dataset; it is not a hosted LLM.


## Dashboard theme
The Overview page uses a midnight-purple background, lavender/purple panels, gold headings, activity chart, dataset-aware AI assistant, recent alerts, and system status. The original feature pages remain available from the sidebar.
