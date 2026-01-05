import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="AI-Based Network Intrusion Detection - PortScan ", layout="wide")

st.title("🛡️ AI-Based Network Intrusion Detection ")
st.markdown("286K samples | BENIGN + PortScan | 99% accuracy expected")

# --- EXACT COLUMN LOADER (CSV confirmed) ---
@st.cache_data
def load_portscan_data():
    csv_file = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    df = pd.read_csv(csv_file, low_memory=False)
    
    # EXACT COLUMNS 
    label_col = ' Label'  # Has leading space
    all_cols = df.columns.tolist()
    
    st.success(f"✅ Loaded {len(df):,} samples | {len(all_cols)} columns")
    st.caption(f"Label column: '{label_col}' ✓")
    
    # TOP 10 MOST IMPORTANT FEATURES (safe, exist in CSV)
    feature_cols = [
        ' Destination Port', ' Flow Duration', ' Total Fwd Packets', 
        ' Total Backward Packets', 'Total Length of Fwd Packets', 
        ' Total Length of Bwd Packets', ' Fwd Packet Length Mean', 
        ' Flow Packets/s', 'Fwd Packets/s', ' Packet Length Mean'
    ]
    
    # Verify ALL features exist
    safe_features = [col for col in feature_cols if col in df.columns]
    st.info(f"Using {len(safe_features)} verified features")
    
    # Clean ONLY safe features
    df_clean = df[safe_features + [label_col]].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean.dropna(subset=[label_col], inplace=True)
    df_clean[safe_features] = df_clean[safe_features].fillna(0)
    
    # Encode labels
    le = LabelEncoder()
    df_clean['Label_encoded'] = le.fit_transform(df_clean[label_col])
    
    return df_clean, le, safe_features

# LOAD DATA
try:
    df, label_encoder, feature_cols = load_portscan_data()
    st.balloons()  # Success animation!
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

# DASHBOARD
col1, col2, col3 = st.columns(3)
col1.metric("Samples", f"{len(df):,}")
col2.metric("Features", len(feature_cols))
col3.metric("PortScan %", f"{len(df[df['Label_encoded']==1])/len(df)*100:.1f}%")

# Preview
st.subheader("📊 Preview")
st.dataframe(df[feature_cols + [' Label', 'Label_encoded']].head())

# Class distribution
fig = px.pie(values=df['Label_encoded'].value_counts().values, 
             names=['BENIGN', 'PortScan'], title="Class Balance")
st.plotly_chart(fig)

# --- PREPROCESS ---
X = df[feature_cols]
y = df['Label_encoded']

# --- CONTROLS ---
st.sidebar.header("⚙️ Controls")
test_size = st.sidebar.slider("Test Split %", 5, 30, 15)
trees = st.sidebar.slider("Trees", 50, 200, 100)

# --- TRAINING ---
st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🚀 Train Model")
    if st.button("**TRAIN PORTSCAN DETECTOR**", type="primary"):
        with st.spinner(f"Training on {len(df):,} samples..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42, stratify=y)
            
            model = RandomForestClassifier(n_estimators=trees, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            st.session_state.update({
                'model': model, 'X_test': X_test, 'y_test': y_test,
                'features': feature_cols
            })
            st.success("✅ **99%+ accuracy expected!**")

# --- RESULTS ---
with col2:
    st.subheader("📈 Performance")
    if 'model' in st.session_state:
        y_pred = st.session_state['model'].predict(st.session_state['X_test'])
        acc = accuracy_score(st.session_state['y_test'], y_pred)
        
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("PortScans Caught", f"{np.sum(y_pred)}")
        
        # Confusion Matrix
        cm = confusion_matrix(st.session_state['y_test'], y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
        st.pyplot(fig)

# --- LIVE DETECTOR ---
st.divider()
st.subheader("🎯 **LIVE PortScan Detector**")
st.info("💡 **PortScan =** High `Flow Packets/s` + Many `Destination Ports`")


c1, c2, c3, c4, c5 = st.columns(5)
dest_port = c1.number_input("Destination Port", 1, 65535, 80)
flow_duration = c2.number_input("Flow Duration (ms)", 0, 1000000, 5000)
fwd_packets = c3.number_input("Total Fwd Packets", 0, 500, 5)
bwd_packets = c4.number_input("Total Bwd Packets", 0, 500, 3)
flow_pps = c5.number_input("Flow Packets/s **(Key!)**", 0.0, 20000.0, 10.0)

if st.button("**🔍 DETECT NOW**", type="primary", use_container_width=True):
    if 'model' in st.session_state:
        test_packet = np.array([[
            dest_port, flow_duration, fwd_packets, bwd_packets,
            fwd_packets*400, bwd_packets*300, 500,  
            flow_pps*0.8, flow_pps, 500  
        ]])
        
        pred = st.session_state['model'].predict(test_packet)[0]
        probs = st.session_state['model'].predict_proba(test_packet)[0]
        
        if pred == 1:
            st.error("### 🚨 **PORTSCAN ATTACK!**")
            st.caption("💥 High scan rate + packet volume")
        else:
            st.success("### ✅ **SAFE Traffic**")
        
        st.metric("Attack Probability", f"{probs[1]:.1%}")
    else:
        st.warning("Train first! 👆")

# Feature importance
if 'model' in st.session_state:
    st.subheader("📊 Top Detection Features")
    importances = pd.DataFrame({
        'Feature': [f.strip() for f in st.session_state['features']],
        'Importance': st.session_state['model'].feature_importances_
    }).sort_values('Importance', ascending=False).head(8)
    
    fig = px.bar(importances, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig)