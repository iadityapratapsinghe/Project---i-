"""
Music Genre Classifier and Explorer - Streamlit Web Application
Student Name: Aditya Pratap Singh | Roll No.: 22515500003
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(
    page_title="Music Genre Classifier & Explorer",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Genre Classifier & Explorer")
st.markdown("**B.Tech Computer Science & Engineering (AI)** | *Aditya Pratap Singh (22515500003)*")
st.markdown("---")

# 2. Load Saved Model Artifacts Safely with Error Handling
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("artifacts/model.pkl")
        scaler = joblib.load("artifacts/scaler.pkl")
        encoder = joblib.load("artifacts/encoder.pkl")
        features = joblib.load("artifacts/features.pkl")
        return model, scaler, encoder, features, True
    except Exception as e:
        return None, None, None, None, False

model, scaler, encoder, feature_cols, artifacts_loaded = load_model_artifacts()

# Auto-load dataset into session_state if present in folder
if 'dataset' not in st.session_state and os.path.exists("features_30_sec.csv"):
    st.session_state['dataset'] = pd.read_csv("features_30_sec.csv")

# 3. Sidebar Navigation Menu (Expanded for all features)
menu = st.sidebar.selectbox("Navigation Menu", [
    "1. Dataset Upload & Overview", 
    "2. Genre Prediction & Output", 
    "3. Top-3 Probabilities (Uncertainty)",
    "4.1 Music DNA Profile (Radar)",
    "4.2 Explainable Feature Importance",
    "4.3 Similar Music Finder",
    "4.4 Mood & Energy Explorer"
])

# PAGE 1: DATASET UPLOAD & OVERVIEW 
if menu == "1. Dataset Upload & Overview":
    st.header("Dataset Ingestion & Preview")
    st.markdown("Upload your structured CSV audio-feature dataset (e.g., `features_30_sec.csv`) to begin.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['dataset'] = df
            st.success(f"Dataset successfully loaded! Total rows: {df.shape[0]}, Columns: {df.shape[1]}")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    elif 'dataset' in st.session_state:
        st.success("Loaded local 'features_30_sec.csv' automatically from your project folder.")
    
    if 'dataset' in st.session_state:
        st.dataframe(st.session_state['dataset'].head())
    else:
        st.warning("Please upload a CSV dataset file to proceed.")

# PAGE 2: GENRE PREDICTION & OUTPUT 
elif menu == "2. Genre Prediction & Output":
    st.header("Single Track Classification Inference")
    
    if not artifacts_loaded:
        st.error("Error: Model artifacts not found in the 'artifacts/' folder.")
    elif 'dataset' in st.session_state:
        df = st.session_state['dataset']
        row_idx = st.number_input("Select Track Index", min_value=0, max_value=len(df)-1, value=0)
        selected_row = df.iloc[row_idx]
        
        actual_genre = selected_row['label'] if 'label' in selected_row else "Unknown"
        st.write(f"**Actual Genre in Dataset:** `{actual_genre}`")
        
        if st.button("Predict Genre"):
            try:
                feature_values = pd.to_numeric(selected_row[feature_cols], errors='coerce')
                input_vector = feature_values.values.reshape(1, -1).astype(float)
                
                if np.isnan(input_vector).any():
                    st.error("Error: Selected row contains missing or NaN numerical values.")
                else:
                    scaled_input = scaler.transform(input_vector)
                    pred_idx = model.predict(scaled_input)[0]
                    predicted_genre = encoder.inverse_transform([pred_idx])[0]
                    
                    probabilities = model.predict_proba(scaled_input)[0]
                    confidence = probabilities[pred_idx] * 100
                    
                    st.success(f"### Predicted Genre: **{predicted_genre.upper()}**")
                    st.info(f"Model Confidence: **{confidence:.2f}%**")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please load your dataset in Page 1 first.")

# PAGE 3: TOP-3 PROBABILITIES 
elif menu == "3. Top-3 Probabilities (Uncertainty)":
    st.header("Top-3 Genre Probabilities (Uncertainty Analysis)")
    st.markdown("Shows the three most likely genre classes with model-estimated probabilities.")
    
    if not artifacts_loaded:
        st.error("Error: Model artifacts not found.")
    elif 'dataset' in st.session_state:
        df = st.session_state['dataset']
        row_idx = st.number_input("Select Track Index", min_value=0, max_value=len(df)-1, value=0, key="p3")
        selected_row = df.iloc[row_idx]
        
        if st.button("Calculate Top-3 Probabilities"):
            try:
                feature_values = pd.to_numeric(selected_row[feature_cols], errors='coerce')
                input_vector = feature_values.values.reshape(1, -1).astype(float)
                scaled_input = scaler.transform(input_vector)
                
                probabilities = model.predict_proba(scaled_input)[0]
                top_3_indices = np.argsort(probabilities)[::-1][:3]
                top_3_genres = encoder.inverse_transform(top_3_indices)
                top_3_scores = probabilities[top_3_indices] * 100
                
                prob_df = pd.DataFrame({"Genre": top_3_genres, "Probability (%)": top_3_scores})
                fig = px.bar(prob_df, x="Genre", y="Probability (%)", text="Probability (%)", title="Top-3 Most Likely Genres")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating probabilities: {e}")
    else:
        st.warning("Please load your dataset in Page 1 first.")

# PAGE 4.1: MUSIC DNA PROFILE (RADAR CHART) 
elif menu == "4.1 Music DNA Profile (Radar)":
    st.header("Music DNA Profile (Radar Chart)")
    st.markdown("Visualizes a selected track's musical profile across normalized audio attributes (tempo, energy, spectral features).")
    
    if 'dataset' in st.session_state:
        df = st.session_state['dataset']
        row_idx = st.number_input("Select Track Index", min_value=0, max_value=len(df)-1, value=0, key="p41")
        selected_row = df.iloc[row_idx]
        
        # Pick a few key representative numerical features if available
        available_cols = [c for c in ['tempo', 'rms_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean', 'rolloff_mean', 'zero_crossing_rate_mean'] if c in df.columns]
        if not available_cols:
            available_cols = feature_cols[:6] # fallback to first 6 features
            
        if st.button("Generate DNA Radar Chart"):
            try:
                vals = [float(selected_row[c]) for c in available_cols]
                # Min-max scale values for radar representation (0 to 1 range approx)
                max_vals = [float(df[c].max()) if df[c].max() != 0 else 1.0 for c in available_cols]
                normalized_vals = [v / m for v, m in zip(vals, max_vals)]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=normalized_vals,
                    theta=available_cols,
                    fill='toself',
                    name=f"Track {row_idx}"
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title=f"Audio Feature DNA Profile for Track Index {row_idx}"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating radar chart: {e}")
    else:
        st.warning("Please load your dataset in Page 1 first.")

# PAGE 4.2: EXPLAINABLE FEATURE IMPORTANCE 
elif menu == "4.2 Explainable Feature Importance":
    st.header("Explainable Genre Prediction & Feature Influence")
    st.markdown("Displays the audio features that most influenced the model's decision structure.")
    
    if not artifacts_loaded:
        st.error("Model artifacts not found.")
    else:
        try:
            importances = None
            
            # Case 1: Tree-based models (Random Forest, Gradient Boosting)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                method_name = "Tree-based Feature Importance"
                
            # Case 2: Linear models (Logistic Regression, Linear SVM)
            elif hasattr(model, 'coef_'):
                # Take the mean absolute value across classes for multi-class coefficients
                importances = np.mean(np.abs(model.coef_), axis=0)
                method_name = "Linear Model Coefficients (Absolute Magnitude)"
                
            if importances is not None:
                feat_imp_df = pd.DataFrame({
                    "Feature": feature_cols, 
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False).head(10)
                
                fig = px.bar(
                    feat_imp_df, 
                    x="Importance", 
                    y="Feature", 
                    orientation='h', 
                    title=f"Top 10 Influential Features ({method_name})"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"Method used: **{method_name}**. Higher values indicate features that carry stronger weight in the model's classification decisions.")
            
            # Case 3: Distance/Kernel models (KNN, Kernel SVM, etc.) where direct weights aren't exposed
            else:
                st.warning("This model type (e.g., KNN or non-linear SVM) relies on instance distances or complex kernels rather than direct feature weights. However, you can analyze feature variance across your dataset below:")
                
                if 'dataset' in st.session_state:
                    df = st.session_state['dataset']
                    numeric_df = pd.to_numeric(df[feature_cols].stack(), errors='coerce').unstack()
                    variances = numeric_df.var().sort_values(ascending=False).head(10)
                    
                    var_df = pd.DataFrame({"Feature": variances.index, "Variance": variances.values})
                    fig = px.bar(var_df, x="Variance", y="Feature", orientation='h', title="Top 10 High-Variance Features (Dataset Spread)")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please load your dataset in Page 1 to view feature variance analysis.")
                    
        except Exception as e:
            st.error(f"Error calculating feature importance: {e}")

# PAGE 4.3: SIMILAR MUSIC FINDER
elif menu == "4.3 Similar Music Finder":
    st.header("Similar Music Finder")
    st.markdown("Recommends songs from the dataset with similar feature patterns using cosine similarity.")
    
    if not artifacts_loaded:
        st.error("Model artifacts not found.")
    elif 'dataset' in st.session_state:
        df = st.session_state['dataset']
        row_idx = st.number_input("Select Reference Track Index", min_value=0, max_value=len(df)-1, value=0, key="p43")
        
        if st.button("Find Similar Songs"):
            try:
                # Extract all feature vectors
                X_all = pd.to_numeric(df[feature_cols].stack(), errors='coerce').unstack().fillna(0)
                X_scaled = scaler.transform(X_all)
                
                ref_vector = X_scaled[row_idx].reshape(1, -1)
                similarities = cosine_similarity(ref_vector, X_scaled)[0]
                
                # Get top 5 similar indices (excluding itself)
                top_indices = np.argsort(similarities)[::-1][1:6]
                
                rec_df = df.iloc[top_indices].copy()
                rec_df['Similarity Score'] = similarities[top_indices]
                
                display_cols = ['filename', 'label', 'Similarity Score'] if 'filename' in rec_df.columns else ['label', 'Similarity Score']
                st.dataframe(rec_df[[c for c in display_cols if c in rec_df.columns]])
            except Exception as e:
                st.error(f"Error finding similar tracks: {e}")
    else:
        st.warning("Please load your dataset in Page 1 first.")

# PAGE 4.4: MOOD & ENERGY EXPLORER
elif menu == "4.4 Mood & Energy Explorer":
    st.header("Mood and Energy Explorer")
    st.markdown("Explore analytical feature-based groups (Approximate categories: High-Energy vs. Low-Energy).")
    
    if 'dataset' in st.session_state:
        df = st.session_state['dataset']
        energy_col = 'rms_mean' if 'rms_mean' in df.columns else (feature_cols[0] if feature_cols else None)
        
        if energy_col:
            median_val = df[energy_col].median()
            df['Analytical Category'] = df[energy_col].apply(lambda x: 'High-Energy / Intense' if x >= median_val else 'Low-Energy / Calm')
            
            category_filter = st.selectbox("Select Analytical Group", ["High-Energy / Intense", "Low-Energy / Calm"])
            filtered_df = df[df['Analytical Category'] == category_filter]
            
            st.write(f"Showing tracks classified under **{category_filter}** (Total: {len(filtered_df)} tracks):")
            st.dataframe(filtered_df.head(20))
        else:
            st.error("Energy-related feature column not found in dataset.")
    else:
        st.warning("Please load your dataset in Page 1 first.")

st.markdown("---")
st.markdown("© 2026 Aditya Pratap Singh | GLA University, Greater Noida")