"""
Streamlit Frontend Application for SentimentIQ

Interactive Web App supporting Single Text Analysis, Real-Time Analytics Dashboard,
Batch CSV Processing, and Architectural System Documentation.
"""

import sys
import os
import time
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to sys.path to allow local fallback imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from backend.predictor import predict_text
    from backend.csv_logger import log_prediction, get_prediction_history, get_analytics_summary, clear_prediction_history
    LOCAL_PREDICTOR_AVAILABLE = True
except Exception as e:
    LOCAL_PREDICTOR_AVAILABLE = False

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Page Configuration
st.set_page_config(
    page_title="SentimentIQ — Affective Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark/Glassmorphic Aesthetic
st.markdown("""
<style>
    /* Global Styles & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Layout Styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Metric Cards Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    /* Custom Badges */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 4px;
    }
    
    .badge-positive { background-color: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
    .badge-negative { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-neutral { background-color: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid #94a3b8; }
    
    .badge-joy { background-color: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid #eab308; }
    .badge-sadness { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
    .badge-anger { background-color: rgba(225, 29, 72, 0.2); color: #fda4af; border: 1px solid #e11d48; }
    .badge-fear { background-color: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid #a855f7; }
    .badge-love { background-color: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid #ec4899; }
    .badge-surprise { background-color: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #f97316; }
    
    .badge-sarcastic { background-color: rgba(245, 158, 11, 0.25); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-normal { background-color: rgba(20, 184, 166, 0.2); color: #2dd4bf; border: 1px solid #14b8a6; }
    
    /* Header Styling */
    .title-text {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)


def render_mermaid_diagram(code: str, height: int = 580):
    """Render a live interactive Mermaid JS diagram via HTML component."""
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                securityLevel: 'loose',
                themeVariables: {{
                    fontFamily: 'Inter, sans-serif',
                    darkMode: true,
                    background: '#0e1117',
                    primaryColor: '#6366f1',
                    primaryTextColor: '#ffffff',
                    primaryBorderColor: '#818cf8',
                    lineColor: '#94a3b8'
                }}
            }});
        </script>
        <style>
            body {{
                background-color: #0e1117;
                margin: 0;
                padding: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .mermaid {{
                width: 100%;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
        {code}
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


def check_api_health() -> bool:
    """Check if FastAPI server is responsive."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_predict_api(text: str) -> dict:
    """Call backend /predict endpoint with fallback to direct module invocation."""
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    if LOCAL_PREDICTOR_AVAILABLE:
        pred = predict_text(text)
        return log_prediction(pred)
    else:
        st.error("API server unavailable and local predictor fallback failed.")
        st.stop()


# Sidebar Navigation
st.sidebar.markdown("<h2 class='title-text'>🧠 SentimentIQ</h2>", unsafe_allow_html=True)
st.sidebar.markdown("*Multi-Tiered Affective Intelligence*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Analyze Text", "📊 Analytics Dashboard", "📁 Batch Processing", "ℹ️ Architecture & About"],
    index=0
)

st.sidebar.markdown("---")
# Health Status
api_online = check_api_health()
if api_online:
    st.sidebar.success("🟢 API Server: Online (Port 8000)")
else:
    st.sidebar.warning("🟠 API Server: Offline (Using Local Engine)")

st.sidebar.caption("Powered by Scikit-Learn, Transformers & FastAPI")


# ==============================================================================
# PAGE 1: ANALYZE TEXT
# ==============================================================================
if page == "🔍 Analyze Text":
    st.markdown("<h1 class='title-text'>Real-Time Text Analysis</h1>", unsafe_allow_html=True)
    st.markdown("Analyze sentiment, underlying emotion, fine-grained feeling, and sarcasm detection in real-time.")

    # Initialize session state key for text input persistence
    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    # Sample Quick Load Buttons
    st.markdown("##### 💡 Try Sample Inputs:")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    if col_s1.button("🎉 Joyful News"):
        st.session_state.user_text = "I am so thrilled and grateful for this incredible promotion at work!"
    if col_s2.button("🤬 Frustrated Rant"):
        st.session_state.user_text = "This customer support is an absolute nightmare, no one ever responds!"
    if col_s3.button("😏 Sarcastic Remark"):
        st.session_state.user_text = "Oh great, another unexpected flight delay, just what I needed today /s"
    if col_s4.button("😰 Anxious Thought"):
        st.session_state.user_text = "I feel so nervous and terrified about the upcoming medical results."

    user_input = st.text_area(
        "Enter text to analyze:",
        key="user_text",
        height=120,
        placeholder="Type or paste text here (e.g., 'I absolutely love this new product, it completely made my day!')..."
    )

    c1, c2 = st.columns([1, 5])
    analyze_btn = c1.button("⚡ Analyze Text", type="primary", use_container_width=True)
    clear_btn = c2.button("🗑️ Clear", use_container_width=False)

    if clear_btn:
        st.session_state.user_text = ""
        st.rerun()

    if analyze_btn and st.session_state.user_text.strip():
        with st.spinner("Processing text through SentimentIQ pipeline..."):
            result = call_predict_api(st.session_state.user_text.strip())

        st.markdown("---")
        st.subheader("Analysis Results")

        sentiment = result.get("sentiment", "Neutral")
        emotion = result.get("emotion", "Neutral")
        feeling = result.get("feeling", "Peaceful")
        sarcasm = result.get("sarcasm", False)
        confidence = float(result.get("confidence", 0.0))

        # CSS Class mappings for badges
        s_class = f"badge-{sentiment.lower()}"
        e_class = f"badge-{emotion.lower()}"
        sarcasm_class = "badge-sarcastic" if sarcasm else "badge-normal"
        sarcasm_label = "⚠️ Sarcastic / Ironic" if sarcasm else "✅ Normal Tone"

        # Metric Cards Grid
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment</div>
                <div class="metric-value"><span class="badge {s_class}">{sentiment}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Emotion</div>
                <div class="metric-value"><span class="badge {e_class}">{emotion}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mapped Feeling</div>
                <div class="metric-value"><span class="badge badge-positive">{feeling}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sarcasm Flag</div>
                <div class="metric-value"><span class="badge {sarcasm_class}">{sarcasm_label}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence Bar
        st.markdown(f"##### 🎯 Model Confidence Score: `{confidence * 100:.1f}%`")
        st.progress(confidence)

        # Raw JSON Expander
        with st.expander("🛠️ View Raw JSON Payload"):
            st.json(result)


# ==============================================================================
# PAGE 2: ANALYTICS DASHBOARD
# ==============================================================================
elif page == "📊 Analytics Dashboard":
    st.markdown("<h1 class='title-text'>Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Real-time aggregated statistics and trend analysis from logged predictions.")

    # Fetch Analytics Summary
    summary = {}
    history = []

    try:
        r = requests.get(f"{API_URL}/analytics", timeout=3)
        if r.status_code == 200:
            summary = r.json()
        r_hist = requests.get(f"{API_URL}/history?limit=500", timeout=3)
        if r_hist.status_code == 200:
            history = r_hist.json()
    except Exception:
        if LOCAL_PREDICTOR_AVAILABLE:
            summary = get_analytics_summary()
            history = get_prediction_history(limit=500)

    total_count = summary.get("total_count", 0)
    avg_conf = summary.get("avg_confidence", 0.0)
    sarcasm_cnt = summary.get("sarcasm_count", 0)
    sarcasm_rate = (sarcasm_cnt / total_count * 100) if total_count > 0 else 0.0

    # Top KPI Metrics
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Predictions", f"{total_count:,}")
    k2.metric("Average Confidence", f"{avg_conf * 100:.1f}%")
    k3.metric("Sarcasm Detection Rate", f"{sarcasm_rate:.1f}%")

    st.markdown("---")

    if total_count == 0:
        st.info("No prediction data logged yet. Analyze some text or run batch processing to see analytics!")
    else:
        # Plotly Charts Grid
        ch1, ch2 = st.columns(2)

        # 1. Sentiment Distribution (Pie Chart)
        sent_dist = summary.get("sentiment_distribution", {})
        if sent_dist:
            df_sent = pd.DataFrame(list(sent_dist.items()), columns=["Sentiment", "Count"])
            fig_sent = px.pie(
                df_sent,
                values="Count",
                names="Sentiment",
                title="Sentiment Distribution",
                color="Sentiment",
                color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8"},
                hole=0.4
            )
            fig_sent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            ch1.plotly_chart(fig_sent, use_container_width=True)

        # 2. Emotion Distribution (Bar Chart)
        emo_dist = summary.get("emotion_distribution", {})
        if emo_dist:
            df_emo = pd.DataFrame(list(emo_dist.items()), columns=["Emotion", "Count"])
            fig_emo = px.bar(
                df_emo,
                x="Emotion",
                y="Count",
                title="Emotion Distribution",
                color="Emotion",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_emo.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            ch2.plotly_chart(fig_emo, use_container_width=True)

        ch3, ch4 = st.columns(2)

        # 3. Feeling Distribution
        feel_dist = summary.get("feeling_distribution", {})
        if feel_dist:
            df_feel = pd.DataFrame(list(feel_dist.items()), columns=["Feeling", "Count"])
            fig_feel = px.bar(
                df_feel,
                x="Count",
                y="Feeling",
                orientation="h",
                title="Fine-Grained Feeling Distribution",
                color="Feeling",
                color_discrete_sequence=px.colors.sequential.Sunset
            )
            fig_feel.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            ch3.plotly_chart(fig_feel, use_container_width=True)

        # 4. Confidence Score Distribution Histogram
        if history:
            df_hist = pd.DataFrame(history)
            fig_conf = px.histogram(
                df_hist,
                x="confidence",
                nbins=20,
                title="Confidence Score Histogram",
                color_discrete_sequence=["#818cf8"]
            )
            fig_conf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ed")
            ch4.plotly_chart(fig_conf, use_container_width=True)

        # Recent Prediction History Table
        st.markdown("---")
        st.subheader("📜 Recent Prediction Logs")

        if history:
            df_table = pd.DataFrame(history)
            
            # Search & Filter
            search_query = st.text_input("🔍 Search prediction history by text keyword:", "")
            if search_query.strip():
                df_table = df_table[df_table["text"].str.contains(search_query, case=False, na=False)]

            st.dataframe(df_table, use_container_width=True, height=300)

            btn_col1, btn_col2 = st.columns([1, 5])
            
            csv_bytes = df_table.to_csv(index=False).encode("utf-8")
            btn_col1.download_button(
                "📥 Download CSV Log",
                data=csv_bytes,
                file_name="sentimentiq_history.csv",
                mime="text/csv"
            )

            if btn_col2.button("🗑️ Clear History Log"):
                try:
                    requests.delete(f"{API_URL}/history", timeout=3)
                except Exception:
                    if LOCAL_PREDICTOR_AVAILABLE:
                        clear_prediction_history()
                st.success("Prediction history cleared!")
                st.rerun()


# ==============================================================================
# PAGE 3: BATCH PROCESSING
# ==============================================================================
elif page == "📁 Batch Processing":
    st.markdown("<h1 class='title-text'>Batch CSV Upload</h1>", unsafe_allow_html=True)
    st.markdown("Upload a CSV dataset containing text inputs to perform batch affect intelligence processing.")

    uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"📁 **Uploaded CSV Preview** ({len(df_upload)} rows):")
            st.dataframe(df_upload.head(5), use_container_width=True)

            text_cols = list(df_upload.columns)
            default_idx = 0
            for idx, col in enumerate(text_cols):
                if "text" in col.lower() or "comment" in col.lower() or "review" in col.lower():
                    default_idx = idx
                    break

            selected_col = st.selectbox("Select column containing text to analyze:", text_cols, index=default_idx)

            if st.button("🚀 Process Batch Predictions", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                results = []
                total_rows = len(df_upload)

                for i, row in df_upload.iterrows():
                    text_val = str(row[selected_col])
                    if text_val and text_val.strip() and text_val.lower() != "nan":
                        res = call_predict_api(text_val.strip())
                        results.append({
                            "original_text": text_val,
                            "sentiment": res.get("sentiment"),
                            "emotion": res.get("emotion"),
                            "feeling": res.get("feeling"),
                            "sarcasm": res.get("sarcasm"),
                            "confidence": res.get("confidence")
                        })
                    else:
                        results.append({
                            "original_text": text_val,
                            "sentiment": None,
                            "emotion": None,
                            "feeling": None,
                            "sarcasm": None,
                            "confidence": None
                        })

                    progress = (i + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {i + 1}/{total_rows} rows...")

                status_text.success("🎉 Batch processing complete!")

                df_results = pd.DataFrame(results)
                st.markdown("##### Batch Results Preview:")
                st.dataframe(df_results, use_container_width=True, height=350)

                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Processed Batch CSV",
                    data=csv_data,
                    file_name="sentimentiq_batch_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")


# ==============================================================================
# PAGE 4: ARCHITECTURE & ABOUT
# ==============================================================================
elif page == "ℹ️ Architecture & About":
    st.markdown("<h1 class='title-text'>System Architecture & Documentation</h1>", unsafe_allow_html=True)
    st.markdown("Detailed breakdown of model training pipelines, dataset specs, architectural design, and system trade-offs.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏗️ End-to-End Pipeline Dataflow Diagram")
    
    # Live Interactive Mermaid JS Rendering
    mermaid_code = """
    graph TD
        A["📥 User Input / CSV Batch"] --> B["🧹 backend/preprocess.py (clean_text)"]
        B --> C["🔤 TF-IDF Vectorizer (50k N-Grams)"]
        B --> D["🤖 HuggingFace Sarcasm Model"]
        C --> E["📊 Sentiment Classifier (Logistic Regression)"]
        C --> F["🎭 Emotion Classifier (Logistic Regression)"]
        E --> G["🎯 Confidence Calculator"]
        F --> G
        D --> H["⚠️ Sarcasm Flag (True / False)"]
        G --> I["🧩 backend/feeling_mapper.py (20+ Rule Engine)"]
        H --> I
        I --> J["💾 backend/csv_logger.py"]
        J --> K[("📂 data/predictions.csv")]
        J --> L["🚀 FastAPI REST Server / Streamlit UI"]

        classDef inputStyle fill:#1e293b,stroke:#6366f1,stroke-width:2px,color:#fff;
        classDef processStyle fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#fff;
        classDef modelStyle fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#fff;
        classDef outputStyle fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff;

        class A inputStyle;
        class B,G,I,J processStyle;
        class C,D,E,F,H modelStyle;
        class K,L outputStyle;
    """
    render_mermaid_diagram(mermaid_code, height=560)

    st.markdown("---")
    st.subheader("⚡ 4-Stage Architectural Pipeline Steps")
    
    # Visual HTML Step Cards
    st.markdown("""
    <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 25px;">
        <div style="flex: 1; min-width: 220px; background: rgba(30, 41, 59, 0.7); border-top: 4px solid #6366f1; border-radius: 10px; padding: 18px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <h4 style="margin: 0 0 10px 0; color: #a5b4fc;">1. Preprocessing & Clean</h4>
            <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.5;">Raw text or CSV batch parsed by <code>clean_text()</code>. Performs URL removal, punctuation stripping, and lowercasing.</p>
        </div>
        <div style="flex: 1; min-width: 220px; background: rgba(30, 41, 59, 0.7); border-top: 4px solid #3b82f6; border-radius: 10px; padding: 18px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <h4 style="margin: 0 0 10px 0; color: #93c5fd;">2. Feature & Deep Learning</h4>
            <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.5;">TF-IDF transforms text into 50,000 N-Gram vector space while HuggingFace Transformer detects sarcasm.</p>
        </div>
        <div style="flex: 1; min-width: 220px; background: rgba(30, 41, 59, 0.7); border-top: 4px solid #a855f7; border-radius: 10px; padding: 18px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <h4 style="margin: 0 0 10px 0; color: #d8b4fe;">3. Affect Synthesis Engine</h4>
            <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.5;">Balanced Logistic Regressions compute Sentiment & Emotion. 20+ Rule Engine synthesizes 7 human Feelings.</p>
        </div>
        <div style="flex: 1; min-width: 220px; background: rgba(30, 41, 59, 0.7); border-top: 4px solid #10b981; border-radius: 10px; padding: 18px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <h4 style="margin: 0 0 10px 0; color: #6ee7b7;">4. Persistence & API UI</h4>
            <p style="font-size: 0.88rem; color: #cbd5e1; margin: 0; line-height: 1.5;">Persisted to <code>predictions.csv</code> with auto-increment ID & timestamp, served via FastAPI & Streamlit.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.subheader("📚 Datasets & Model Specifications")
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);">
            <h4 style="color: #6366f1; margin-top: 0;">📊 Sentiment Classifier</h4>
            <ul style="color: #cbd5e1; font-size: 0.9rem;">
                <li><b>Dataset</b>: <code>cardiffnlp/tweet_eval</code> (Sentiment split)</li>
                <li><b>Model</b>: TF-IDF (1,2 n-grams, max 50k features) + Logistic Regression (<code>class_weight='balanced'</code>)</li>
                <li><b>Accuracy</b>: ~65% validation / ~60% test accuracy on tweet benchmarks</li>
            </ul>
            <h4 style="color: #a855f7; margin-top: 15px;">🎭 Emotion Classifier</h4>
            <ul style="color: #cbd5e1; font-size: 0.9rem;">
                <li><b>Dataset</b>: <code>dair-ai/emotion</code> (sadness, joy, love, anger, fear, surprise)</li>
                <li><b>Model</b>: Shared Day 2 TF-IDF vector space + Logistic Regression</li>
                <li><b>Performance</b>: ~68% weighted F1 score</li>
            </ul>
            <h4 style="color: #ec4899; margin-top: 15px;">🤖 Sarcasm & Irony Detector</h4>
            <ul style="color: #cbd5e1; font-size: 0.9rem;">
                <li><b>Model</b>: <code>helinivan/english-sarcasm-detector</code> HuggingFace Transformer</li>
                <li><b>Fallback</b>: Heuristic rule detector for offline execution resilience</li>
            </ul>
            <h4 style="color: #10b981; margin-top: 15px;">🧩 Feeling Mapper Engine</h4>
            <ul style="color: #cbd5e1; font-size: 0.9rem;">
                <li><b>Architecture</b>: 20+ Word-boundary rule engine synthesizing Sentiment + Emotion into 7 Feelings (*Excited, Grateful, Frustrated, Bitter, Anxious, Shocked, Peaceful*)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.subheader("⚖️ Architectural Trade-Offs & Decisions")
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);">
            <h4 style="color: #38bdf8; margin-top: 0;">🚫 Data Leakage Prevention</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 15px;">
                The TF-IDF vectorizer was <code>.fit_transform()</code>ed <b>exclusively on the training split</b> of the sentiment dataset. All validation/test splits and emotion models only call <code>.transform()</code>, preventing vocabulary leakage.
            </p>
            <h4 style="color: #fb923c; margin-top: 15px;">⚠️ The Disgust Gap</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 15px;">
                The <code>dair-ai/emotion</code> dataset omits a <code>Disgust</code> label. To address this gap, the Feeling Mapper routes strong negative sentiment without anger into <code>Frustrated</code> or <code>Bitter</code> depending on keyword triggers.
            </p>
            <h4 style="color: #f43f5e; margin-top: 15px;">⚡ Resilient Offline Inference</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0;">
                HuggingFace transformer pipeline downloads lazily and defaults gracefully to a fast keyword heuristic if offline or low-bandwidth environments are encountered.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("💡 **SentimentIQ** portfolio project built with Python 3.11, FastAPI, Streamlit, Scikit-Learn, and `uv` package manager.")
