import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from groq import Groq

# --- PAGE SETUP ---
st.set_page_config(page_title="AI-NIDS PortScan Detector", layout="wide")

st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**: Detects **PortScan attacks** with Random Forest + **Groq AI Analyst**
**Dataset**: 286K samples | BENIGN + PortScan
""")

# --- CONFIGURATION ---
DATA_FILE = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("⚙️ Settings")
groq_api_key = st.sidebar.text_input("Groq API Key (starts with gsk_)", type="password")
st.sidebar.caption("[Get free key](https://console.groq.com/keys)")

st.sidebar.header("🎯 Model Controls")
test_size = st.sidebar.slider("Test Split %", 5, 30, 15)
trees = st.sidebar.slider("Trees", 50, 200, 100)

@st.cache_data
def load_portscan_data():
    try:
        # Load raw data
        df = pd.read_csv(DATA_FILE, low_memory=False, nrows=50000)  # Limit for speed
        
        # STRIP ALL COLUMN NAMES FIRST
        df.columns = df.columns.str.strip()
        
        # AUTO-DETECT LABEL COLUMN (handles both 'Label' and ' Label')
        label_col = None
        possible_labels = ['Label', ' label', 'Label_encoded']
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                break
        
        
        # Define features (with stripped names)
        feature_cols = [
            'Destination Port', 'Flow Duration', 'Total Fwd Packets', 
            'Total Backward Packets', 'Total Length of Fwd Packets', 
            'Total Length of Bwd Packets', 'Fwd Packet Length Mean', 
            'Flow Packets/s', 'Fwd Packets/s', 'Packet Length Mean'
        ]
        
        # Find available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) < 5:
            st.error(f"❌ Need more features. Available: {available_features}")
            st.write("All columns:", df.columns.tolist())
            return None, None, None
        
        
        # Clean data
        df_clean = df[available_features + [label_col]].copy()
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        
       
        initial_rows = len(df_clean)
        df_clean.dropna(subset=[label_col], inplace=True)
        
        # Fill remaining NaNs
        df_clean[available_features] = df_clean[available_features].fillna(0)
        
        # Encode labels
        unique_labels = df_clean[label_col].unique()

        
        le = LabelEncoder()
        df_clean['Label_encoded'] = le.fit_transform(df_clean[label_col])
        
        return df_clean, le, available_features
        
    except FileNotFoundError:
        st.error(f"❌ File '{DATA_FILE}' not found!")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Load error: {str(e)}")
        return None, None, None

# --- MAIN APP ---
df, label_encoder, feature_cols = load_portscan_data()

if df is None:
    st.stop()

# --- TRAINING ---
st.divider()
st.header("🚀 1. Train Detector")

X = df[feature_cols]
y = df['Label_encoded']

if st.button("**TRAIN PORTSCAN DETECTOR**", type="primary", use_container_width=True):
    with st.spinner("Training..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=trees, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.session_state.update({
            'model': model, 'X_test': X_test, 'y_test': y_test,
            'features': feature_cols, 'label_encoder': label_encoder,
            'accuracy': acc
        })
        st.success(f"✅ **Trained! Accuracy: {acc:.2%}**")
        st.balloons()

# --- RESULTS & AI ANALYSIS ---
if 'model' in st.session_state:
    st.header("📊 2. Live Threat Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎲 Random Attack Simulation")
        if st.button("🎲 **Capture Packet**", use_container_width=True):
            idx = np.random.randint(0, len(st.session_state['X_test']))
            packet_data = st.session_state['X_test'].iloc[idx].values
            actual_label = int(st.session_state['y_test'].iloc[idx])
            
            st.session_state['current_packet'] = packet_data
            st.session_state['actual_label'] = actual_label
    
    if 'current_packet' in st.session_state:
        packet = st.session_state['current_packet']
        
        with col1:
            st.write("**📡 Packet Data:**")
            packet_df = pd.DataFrame({
                'Feature': [f.strip() for f in st.session_state['features']],
                'Value': packet
            })
            st.dataframe(packet_df, use_container_width=True)
        
        with col2:
            # Prediction
            pred = st.session_state['model'].predict([packet])[0]
            pred_label = st.session_state['label_encoder'].inverse_transform([pred])[0]
            actual_label = st.session_state['label_encoder'].inverse_transform([st.session_state['actual_label']])[0]
            probs = st.session_state['model'].predict_proba([packet])[0]
            
            if pred == 1:
                st.error(f"### 🚨 **PORTSCAN DETECTED**")
                st.metric("Attack Confidence", f"{probs[1]:.1%}")
            else:
                st.success("### ✅ **BENIGN**")
                st.metric("Safe Confidence", f"{probs[0]:.1%}")
            
            st.caption(f"Ground Truth: **{actual_label}**")
            
            # Groq AI
            if st.button("🧠 **AI Explanation**"):
                if groq_api_key:
                    try:
                        client = Groq(api_key=groq_api_key)
                        prompt = f"""
                        Cybersecurity analysis for student:
                        Packet: {pred_label} (conf: {probs[1]:.0%})
                        Truth: {actual_label}
                        Features: {packet_df.to_dict('records')}
                        
                        Explain why PortScan/BENIGN in 3 bullet points.
                        """
                        with st.spinner("AI analyzing..."):
                            resp = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                            )
                            st.info(resp.choices[0].message.content)
                    except Exception as e:
                        st.error(f"API Error: {e}")
                else:
                    st.warning("👈 Enter Groq API key")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    col1.metric("✅ Accuracy", f"{st.session_state['accuracy']:.2%}")
    col2.metric("🚨 Attacks Caught", np.sum(st.session_state['model'].predict(st.session_state['X_test'])))
    
    # Confusion Matrix
    cm = confusion_matrix(st.session_state['y_test'], st.session_state['model'].predict(st.session_state['X_test']))
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['BENIGN','PORTSCAN'], yticklabels=['BENIGN','PORTSCAN'])
    st.pyplot(fig)
    
    # Feature importance
    importances = pd.DataFrame({
        'Feature': [f.strip() for f in st.session_state['features']],
        'Importance': st.session_state['model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importances.head(8), x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig)

else:
    st.info("👆 Click **TRAIN** to start!")