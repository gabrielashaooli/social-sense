import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Viral Trend Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .recommendation-card {
        background-color: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .interpretation-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    df = pd.read_csv("Viral_Social_Media_Trends.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    if 'Post_ID' in df.columns:
        df = df.drop(['Post_ID'], axis=1)
    df['Engagement_Rate'] = (df['Likes'] + df['Shares'] + df['Comments']) / df['Views']
    df['Virality_Score'] = df['Shares'] / df['Views']
    df['Comment_Rate'] = df['Comments'] / df['Views']
    df['Like_Rate'] = df['Likes'] / df['Views']

    def detect_emotion(hashtag):
        hashtag = hashtag.lower()
        if any(word in hashtag for word in ['dance', 'music', 'party']): return 'Happy'
        elif any(word in hashtag for word in ['education', 'learn', 'tutorial', 'tech']): return 'Focused'
        elif any(word in hashtag for word in ['challenge', 'fitness', 'workout', 'motivation']): return 'Motivated'
        elif any(word in hashtag for word in ['comedy', 'funny', 'meme', 'lol']): return 'Funny'
        elif any(word in hashtag for word in ['travel', 'adventure', 'explore']): return 'Inspired'
        elif any(word in hashtag for word in ['food', 'cooking', 'recipe']): return 'Satisfied'
        elif any(word in hashtag for word in ['gaming', 'game', 'esports']): return 'Competitive'
        elif any(word in hashtag for word in ['beauty', 'fashion', 'style']): return 'Confident'
        else: return 'Neutral'
    df['Emotion'] = df['Hashtag'].apply(detect_emotion)

    np.random.seed(42)
    df['Hour_Posted'] = np.random.randint(0, 24, size=len(df))

    def classify_time_period(hour):
        if hour < 12: return 'Morning'
        elif hour < 18: return 'Afternoon'
        else: return 'Night'
    df['Time_Period'] = df['Hour_Posted'].apply(classify_time_period)

    emotional_energy = {
        'Motivated': 9, 'Competitive': 8, 'Happy': 8, 'Funny': 7,
        'Inspired': 7, 'Confident': 6, 'Focused': 6, 'Satisfied': 5,
        'Neutral': 4
    }
    df['Energy_Level'] = df['Emotion'].map(emotional_energy)

    def temporal_compatibility(time_period, emotion):
        matrix = {
            'Morning': {'Motivated': 10, 'Focused': 9, 'Inspired': 8, 'Happy': 6},
            'Afternoon': {'Happy': 10, 'Funny': 9, 'Satisfied': 8, 'Confident': 7},
            'Night': {'Neutral': 10, 'Funny': 8, 'Happy': 7, 'Competitive': 6}
        }
        return matrix.get(time_period, {}).get(emotion, 5)
    df['Temporal_Compatibility'] = df.apply(
        lambda row: temporal_compatibility(row['Time_Period'], row['Emotion']), axis=1
    )

    min_count = df['Engagement_Level'].value_counts().min()
    df_balanced = df.groupby('Engagement_Level').apply(
        lambda x: x.sample(min(min_count, len(x)), random_state=42)
    ).reset_index(drop=True)

    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

@st.cache_data
def train_model(df):
    y = df['Engagement_Level']
    X = df[['Platform', 'Energy_Level', 'Emotion']].copy()
    label_encoders = {}
    for col in ['Platform', 'Emotion']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    model = LGBMClassifier(objective='multiclass', num_class=3, random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le_target.classes_)

    return model, label_encoders, le_target, accuracy, matrix, report, X_train, X_test, y_train, y_test

@st.cache_data
def prepare_model_comparison_data(df):
    """Prepare data for model comparison"""
    y = df['Engagement_Level']
    X = df[['Platform', 'Energy_Level', 'Emotion', 'Temporal_Compatibility']].copy()
    
    label_encoders = {}
    for col in ['Platform', 'Emotion']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, label_encoders, le_target

def get_model_dict():
    """Dictionary of models for comparison"""
    return {
        'LGBMClassifier': LGBMClassifier(random_state=42, verbose=-1),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=0),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state=42),
        'RidgeClassifier': RidgeClassifier(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis()
    }

def interpret_model_shap(model, model_name, X_train, X_test, le_target):
    """Generate SHAP interpretations"""
    try:
        # Tree-based models
        if model_name in ["RandomForestClassifier", "ExtraTreesClassifier", "XGBClassifier", "LGBMClassifier"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_train)
        
        shap_values = explainer(X_test)
        
        # Summary plot
        fig_summary = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP Summary - {model_name}")
        plt.tight_layout()
        
        # Waterfall plot for first instance
        fig_waterfall = plt.figure(figsize=(10, 6))
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
            # Multi-class case
            shap.plots.waterfall(shap_values[0][:, 0], show=False)
        else:
            shap.plots.waterfall(shap_values[0], show=False)
        plt.title(f"SHAP Waterfall - {model_name} (First Instance)")
        plt.tight_layout()
        
        return fig_summary, fig_waterfall, True
    except Exception as e:
        st.error(f"SHAP Error for {model_name}: {str(e)}")
        return None, None, False

def interpret_model_lime(model, model_name, X_train, X_test, le_target):
    """Generate LIME interpretations"""
    try:
        if not hasattr(model, "predict_proba"):
            return None, False
            
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=le_target.classes_.tolist(),
            mode='classification'
        )
        
        explanation = explainer.explain_instance(
            data_row=X_test.iloc[0].values,
            predict_fn=model.predict_proba,
            num_features=len(X_train.columns)
        )
        
        # Convert LIME explanation to matplotlib figure
        fig = explanation.as_pyplot_figure()
        fig.suptitle(f"LIME Explanation - {model_name}")
        plt.tight_layout()
        
        return fig, True
    except Exception as e:
        st.error(f"LIME Error for {model_name}: {str(e)}")
        return None, False

# Recommendation function
def recommend_content(df, emotion, time_period, platform, num_recommendations):
    filtered_df = df[
        (df['Emotion'] == emotion) &
        (df['Time_Period'] == time_period) &
        (df['Platform'] == platform)
    ].copy()
    if filtered_df.empty:
        return pd.DataFrame()
    filtered_df['Recommendation_Score'] = (
        0.5 * filtered_df['Engagement_Rate'] +
        0.3 * filtered_df['Temporal_Compatibility'] / 10 +
        0.2 * filtered_df['Energy_Level'] / 10
    )
    return filtered_df.sort_values(by='Recommendation_Score', ascending=False).head(num_recommendations)

# Sidebar
st.sidebar.header("🎯 Configuration")
page = st.sidebar.selectbox(
    "Select a page:",
    ["📚 Introduction", "📈 Main Dashboard", "🤖 Model Training", "🔍 Model Interpretability", "💡 Content Recommendations", "✅ Conclusions"]
)

# Load data and model if needed
if page in ["📈 Main Dashboard", "💡 Content Recommendations", "🔍 Model Interpretability", "🤖 Model Training"]:
    df = load_and_process_data()

if page in ["🤖 Model Training"]:
    model, label_encoders, le_target, model_accuracy, cm, report, X_train, X_test, y_train, y_test = train_model(df)

# Pages
if page == "📈 Main Dashboard":
    st.header("Main Dashboard")
    model, label_encoders, le_target, model_accuracy, cm, report, X_train, X_test, y_train, y_test = train_model(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📊 Total Posts", f"{len(df):,}")
    with col2: st.metric("📱 Average Engagement", f"{df['Engagement_Rate'].mean():.3f}")
    with col3: st.metric("🎯 Model Accuracy", f"{model_accuracy:.2%}")
    with col4: st.metric("🏆 Top Platform", df['Platform'].value_counts().idxmax())

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📱 Platform Distribution")
        fig = px.pie(values=df['Platform'].value_counts(), names=df['Platform'].value_counts().index)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("😊 Emotion Distribution")
        fig = px.bar(x=df['Emotion'].value_counts().values, y=df['Emotion'].value_counts().index, orientation='h')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Engagement Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x='Likes', y='Views', color='Engagement_Level', size='Shares')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, y='Shares', color='Engagement_Level')
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.header("LightGBM (LGBMClassifier)")
    st.metric("Accuracy", f"{model_accuracy:.2%}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=le_target.classes_, 
                   yticklabels=le_target.classes_, ax=ax)
        plt.title("Confusion Matrix")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Classification Report")
        st.text(report)

elif page == "🔍 Model Interpretability":
    st.header("🔍 Model Interpretability")
    st.write("Understanding how different models make predictions using SHAP and LIME explanations.")
    
    # Prepare data for model comparison
    X_train_comp, X_test_comp, y_train_comp, y_test_comp, label_encoders_comp, le_target_comp = prepare_model_comparison_data(df)
    
    # Model selection
    model_dict = get_model_dict()
    selected_models = st.multiselect(
        "Select models to interpret:",
        options=list(model_dict.keys()),
        default=["LGBMClassifier", "RandomForestClassifier"]
    )
    
    interpretation_method = st.selectbox(
        "Select interpretation method:",
        ["SHAP", "LIME", "Both"]
    )
    
    if st.button("🚀 Run Model Interpretability Analysis", type="primary"):
        if not selected_models:
            st.warning("Please select at least one model.")
        else:
            for model_name in selected_models:
                with st.expander(f"📊 {model_name} Interpretation", expanded=True):
                    st.markdown(f"""
                    <div class="interpretation-card">
                        <h3>Model: {model_name}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Train the model
                    model = model_dict[model_name]
                    model.fit(X_train_comp, y_train_comp)
                    
                    # Calculate accuracy
                    y_pred = model.predict(X_test_comp)
                    accuracy = accuracy_score(y_test_comp, y_pred)
                    st.metric("Model Accuracy", f"{accuracy:.2%}")
                    
                    # SHAP Interpretation
                    if interpretation_method in ["SHAP", "Both"]:
                        st.subheader("🎯 SHAP Analysis")
                        fig_summary, fig_waterfall, shap_success = interpret_model_shap(
                            model, model_name, X_train_comp, X_test_comp, le_target_comp
                        )
                        
                        if shap_success:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Feature Importance Summary**")
                                st.pyplot(fig_summary)
                            with col2:
                                st.write("**Individual Prediction Explanation**")
                                st.pyplot(fig_waterfall)
                        else:
                            st.warning(f"SHAP analysis not available for {model_name}")
                    
                    # LIME Interpretation
                    if interpretation_method in ["LIME", "Both"]:
                        st.subheader("🔍 LIME Analysis")
                        fig_lime, lime_success = interpret_model_lime(
                            model, model_name, X_train_comp, X_test_comp, le_target_comp
                        )
                        
                        if lime_success:
                            st.write("**Local Explanation for First Test Instance**")
                            st.pyplot(fig_lime)
                        else:
                            st.warning(f"LIME analysis not available for {model_name}")
                    
                    st.markdown("---")

elif page == "📚 Introduction":
    st.header("📚 DATA SCIENCE PROJECT ")
    st.subheader("Gabriela Shaooli, Luis Atristain and  Daniel Leonardo Saldaña")
    st.markdown("""
    Social media has a huge impact on how people interact, share ideas, and follow trends. This project explores how viral content works by analyzing real data from platforms like TikTok, Instagram, and YouTube.

    We focus on how emotions, timing, and platform choice influence engagement levels. The goal is to build a recommendation system that suggests content based on the user's mood, preferred platform, and time of day.

    The dataset includes various features:
    - Hashtags
    - Platform (TikTok, Instagram, YouTube)
    - Views, Likes, Shares, Comments
    - Engagement Level (Low, Medium, High)
    - Detected Emotion (e.g., Happy, Motivated, Funny)
    - Time of Day (Morning, Afternoon, Night)
    - Energy Level (1–10)
    - Temporal Compatibility (how well the emotion fits the time of day)

    Since the original dataset did not include a timestamp, we generated the posting hour randomly for simulation purposes and categorized it into morning, afternoon, and night.

    With this information, we built an interactive web app where you can:
    - Explore trends by platform, emotion, and engagement level
    - See visual patterns through charts and filters
    - Train a machine learning model to predict engagement
    - Interpret model predictions using SHAP and LIME explanations
    - Get personalized content recommendations based on mood, platform, and time

    The goal is to better understand what kind of content goes viral and how data can help us make smarter content decisions.
    """)

elif page == "✅ Conclusions":
    st.header("✅ Conclusions")
    st.markdown("""
    This project allowed us to explore how data science can be applied to understand social media behavior. We chose the topic of viral trends because social media has become a powerful space for communication, influence, and even business.

    By analyzing posts from different platforms, we discovered how factors like emotion, timing, and platform type can impact user engagement. Creating features such as emotional tone and time compatibility helped us go beyond basic numbers and think about how people actually react to content.

    One of the most valuable parts was building an interactive app that combines data processing, visual analysis, machine learning, **model interpretability**, and recommendations. The addition of SHAP and LIME explanations helps us understand not just what the models predict, but why they make those predictions. This transparency is crucial for building trust in AI systems.

    This project also made us reflect on ethical questions around data and emotions. We discussed the controversial Facebook "emotional contagion" experiment, where the platform manipulated users' feeds to observe emotional reactions. This case reminded us how essential transparency and consent are in any project that involves human behavior, especially when emotions are involved. Even in data science, responsibility and ethics should never be an afterthought.
   
    In the end, this project helped us improve both technical skills and our understanding of how digital content works. It also reminded us how important it is to think critically about the content we consume and create online.
    """)

elif page == "💡 Content Recommendations":
    st.header("Content Recommendation System")
    col1, col2, col3 = st.columns(3)
    with col1:
        user_platform = st.selectbox("📱 Your Platform:", options=df['Platform'].unique())
    with col2:
        user_emotion = st.selectbox("😊 Your Mood:", options=df['Emotion'].unique())
    with col3:
        user_time = st.selectbox("🕐 Time of Day:", options=df['Time_Period'].unique())
    num_recommendations = st.slider("📊 Number of Recommendations:", min_value=3, max_value=10, value=5)
    if st.button("🔍 Get Recommendations", type="primary"):
        results = recommend_content(df, user_emotion, user_time, user_platform, num_recommendations)
        if not results.empty:
            st.success(f"🎉 {len(results)} recommendations found")
            for _, row in results.iterrows():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>🏷️ {row['Hashtag']}</h4>
                    <p>👁️ {row['Views']:,} | ❤️ {row['Likes']:,} | 🔄 {row['Shares']:,}</p>
                    <p>📊 Engagement: {row['Engagement_Rate']:.3f} | ⭐ Score: {row['Recommendation_Score']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No recommendations found for those criteria.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#888;'>📊 Final Project - Data Science</div>", unsafe_allow_html=True)