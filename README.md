# 📊 SocialSense – Viral Trend Analysis on Social Media

Welcome to **SocialSense**, a data science project that explores what makes social media content go viral. Using real-world data from platforms like TikTok, Instagram, and YouTube, this app predicts engagement levels and explains model decisions using SHAP and LIME.

Built with 💻 **Python**, 📈 **Streamlit**, and 🧠 **Machine Learning**.

---

## 📌 Project Features

- 📱 Visualize post trends by platform, emotion, and time of day
- 🤖 Train and evaluate a machine learning model (LightGBM)
- 🔍 Interpret predictions using SHAP (global) and LIME (local)
- 💡 Get custom content recommendations based on mood, platform, and time

---

## 🧠 How It Works

### 1. Data Processing
- Simulates posting hour (to preserve privacy)
- Extracts emotions from hashtags
- Creates features: Engagement Rate, Energy Level, Temporal Compatibility

### 2. Model Training
- Uses LightGBM to classify engagement as Low, Medium, or High
- Evaluates with accuracy, confusion matrix, and F1 score

### 3. Interpretability
- SHAP explains global feature importance
- LIME explains specific predictions
- Supports model transparency and trust

### 4. Recommendation Engine
- Filters posts based on emotion, time, and platform
- Ranks them using a custom score combining engagement and compatibility

---

## ⚖️ Ethical Note

Inspired by the controversial [Facebook emotional contagion experiment (2014)](https://time.com/2951726/facebook-emotion-contagion-experiment/), this project was built with privacy and transparency in mind. No personal data was used, timestamps were simulated, and all features are anonymized.

---

## 🛠️ Technologies Used

- Python 3.12
- Streamlit
- Pandas, NumPy
- Scikit-learn, LightGBM, XGBoost
- SHAP, LIME
- Seaborn, Plotly

---
