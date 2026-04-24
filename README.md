# Sleep Health & Lifestyle Dashboard

**Live app:** https://sleep-health-dashboard-zinxayzj4zaw5appnkz6z2u.streamlit.app

I built this to get hands-on with pandas, scikit-learn and Streamlit while working toward data analyst roles in Munich. The question I wanted to answer: what actually predicts whether someone has a sleep disorder — and can I build something interactive around it?

---

## What it does

You can explore how stress, heart rate, occupation and daily habits relate to sleep disorders across 374 people. Filter by occupation or gender and watch the charts update live. Or go to the Predict page, enter your own numbers, and see what the model says about your sleep disorder risk.

---

## What I found

A few things genuinely surprised me during the analysis:

- Stress level correlates with sleep quality at r = -0.90. That's stronger than I expected — it basically drowns out everything else
- I assumed doctors and engineers would have the worst sleep disorders. Turns out engineers are the *least* stressed occupation in this dataset (avg 3.9/10) and have the healthiest sleep. Always check the data before writing conclusions
- Nurses have the highest Sleep Apnea rates despite only moderate stress (5.5/10) — probably shift work, not stress
- Physical activity has almost zero correlation with either stress or sleep. Didn't expect that

---

## How I built it

**Data cleaning**
The trickiest part was Sleep_Disorder — 219 out of 374 rows were NaN. That's 58%, which sounds alarming. But after checking the dataset description, NaN meant "no diagnosis", not missing data. Filled with "None". Blood pressure was stored as a string ("126/83") so I split that into two numeric columns.

**EDA**
Sleep duration has a bimodal distribution — two peaks at 6.5h and 7.8h rather than one clean bell curve. That suggested two distinct subgroups in the data before I even ran a model.

**Linear Regression**
Predicting sleep quality score from lifestyle features. Single train/test split gave R² = 0.91, which looked great. Then I ran 5-fold cross-validation and got 0.60 with std = 0.27 — the initial result was a lucky split. Also ran bootstrap stability analysis (1000 iterations) which showed high internal consistency but doesn't contradict the CV finding. Real-world performance is probably closer to 0.60.

**Logistic Regression**
Predicting sleep disorder type (None / Insomnia / Sleep Apnea). Had to handle a few things:
- Features on different scales caused a ConvergenceWarning — fixed with StandardScaler
- 3x class imbalance (219 None vs ~77 each disorder) — tested class_weight="balanced" and SMOTE, both gave identical results on this dataset, went with balanced weights
- Used a Pipeline for cross-validation to prevent data leakage — the scaler gets fitted on each fold's training data independently
- Final model: 5-fold stratified CV, F1 macro = 0.81 +/- 0.04

**Dashboard**
Built with Streamlit. Sidebar filters update all charts live. Predict page runs the trained model in real time and shows probabilities as a bar chart.

---

## Project structure

```
sleep-health-dashboard/
├── Notebook/
│   └── 01_eda.ipynb        # full analysis with markdown notes
├── app.py                  # Streamlit dashboard
├── Sleep_health_and_lifestyle_dataset.csv
├── sleep_cleaned.csv
└── requirements.txt
```

---

## Stack

pandas, numpy, scikit-learn, imbalanced-learn, statsmodels, matplotlib, seaborn, Streamlit

---

## Limitations worth mentioning

The dataset is synthetic and small (374 rows) — patterns are cleaner than real-world data would be. The linear regression model showed high cross-validation variance, which is a symptom of that. The logistic regression held up better. Predictions from the app are indicative only, not medical advice.

---

## Run it locally

```bash
git clone https://github.com/jayesh122003/sleep-health-dashboard.git
cd sleep-health-dashboard
pip install -r requirements.txt
streamlit run app.py
```
