"""
streamlit_app.py  (project root)

SentinelML Control Panel — 5 pages + sidebar service health

Pages
─────
  🤖 Chat          — natural-language agent via POST /agent/query (SSE)
  🎯 Predict       — live inference form with all 14 UCI Adult features
  🚦 Live Demo     — traffic burst generator + real-time metrics
  📋 Logs          — prediction log viewer with filters
  📊 Dashboard     — Grafana iframe + key metric cards

Sidebar
───────
  Service health panel — green/red for all 8 running services

Run
───
  streamlit run streamlit_app.py

Dependencies
────────────
  pip install streamlit requests sseclient-py plotly pandas
"""

import json
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
GRAFANA_BASE = "http://localhost:3000"
GRAFANA_DASHBOARD_UID = "ml-monitoring-mai"

SERVICES = [
    {"name": "ML API", "url": f"{API_BASE}/health", "port": 8000},
    {"name": "Prometheus", "url": "http://localhost:9090/-/healthy", "port": 9090},
    {"name": "Grafana", "url": "http://localhost:3000/api/health", "port": 3000},
    {"name": "Airflow", "url": "http://localhost:8080/health", "port": 8080},
    {"name": "cAdvisor", "url": "http://localhost:8081/healthz", "port": 8081},
    {"name": "Node Exporter", "url": "http://localhost:9101/metrics", "port": 9101},
    {"name": "Exporter", "url": "http://localhost:9100/metrics", "port": 9100},
    {"name": "PostgreSQL", "url": f"{API_BASE}/health", "port": 5432},
]

# UCI Adult feature definitions
WORKCLASS_OPTS = [
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
]
EDUCATION_OPTS = [
    "Bachelors",
    "Some-college",
    "11th",
    "HS-grad",
    "Prof-school",
    "Assoc-acdm",
    "Assoc-voc",
    "9th",
    "7th-8th",
    "12th",
    "Masters",
    "1st-4th",
    "10th",
    "Doctorate",
    "5th-6th",
    "Preschool",
]
MARITAL_OPTS = [
    "Married-civ-spouse",
    "Divorced",
    "Never-married",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]
OCCUPATION_OPTS = [
    "Tech-support",
    "Craft-repair",
    "Other-service",
    "Sales",
    "Exec-managerial",
    "Prof-specialty",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Priv-house-serv",
    "Protective-serv",
    "Armed-Forces",
]
RELATIONSHIP_OPTS = [
    "Wife",
    "Own-child",
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]
RACE_OPTS = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
SEX_OPTS = ["Male", "Female"]
COUNTRY_OPTS = [
    "United-States",
    "Cuba",
    "Jamaica",
    "India",
    "Mexico",
    "South",
    "Japan",
    "Greece",
    "China",
    "Ecuador",
    "Italy",
    "Poland",
    "Columbia",
    "Cambodia",
    "Thailand",
    "Laos",
    "Taiwan",
    "Haiti",
    "Portugal",
    "Dominican-Republic",
    "El-Salvador",
    "France",
    "Guatemala",
    "China",
    "Nicaragua",
    "Scotland",
    "Thailand",
    "Yugoslavia",
    "Puerto-Rico",
    "Outlying-US(Guam-USVI-etc)",
    "Hungary",
    "Honduras",
    "Hong",
    "Ireland",
    "Trinadad&Tobago",
    "Peru",
    "Vietnam",
    "Iran",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _get(path: str, timeout: int = 5) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post(path: str, payload: dict, timeout: int = 10) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _check_service(url: str | None) -> bool:
    if url is None:
        return False
    try:
        r = requests.get(url, timeout=2)
        return r.status_code < 500
    except Exception:
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────


def _render_sidebar():
    st.sidebar.title("🛰 SentinelML")
    st.sidebar.caption("ML Pipeline Monitor")
    st.sidebar.divider()
    st.sidebar.subheader("Service Health")

    for svc in SERVICES:
        alive = _check_service(svc["url"])
        icon = "🟢" if alive else "🔴"
        label = f"{icon} **{svc['name']}**"
        if svc["url"]:
            link = f"http://localhost:{svc['port']}"
            st.sidebar.markdown(f"{label} — [:{svc['port']}]({link})")
        else:
            st.sidebar.markdown(f"{label} — port {svc['port']}")

    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh Health"):
        st.rerun()


# ── Page: Chat ────────────────────────────────────────────────────────────────


def page_chat():
    st.title("🤖 Chat with SentinelML Agent")
    st.caption("Ask anything about your live ML pipeline in plain English.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("e.g. Is there any drift right now?")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_chunks = []

        try:
            # SSE streaming
            with requests.post(
                f"{API_BASE}/agent/query",
                json={"question": question, "stream": True},
                stream=True,
                timeout=60,
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data = decoded[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "text" in chunk:
                                answer_chunks.append(chunk["text"])
                                placeholder.markdown("".join(answer_chunks) + "▌")
                            elif "error" in chunk:
                                answer_chunks.append(f"\n\n⚠️ {chunk['error']}")
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            answer_chunks = [f"⚠️ Could not reach agent: {e}"]

        final = "".join(answer_chunks) or "⚠️ No response."
        placeholder.markdown(final)

    st.session_state.messages.append({"role": "assistant", "content": final})

    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── Page: Predict ─────────────────────────────────────────────────────────────


def page_predict():
    st.title("🎯 Live Prediction")
    st.caption("Fill in the 14 UCI Adult features and run the model.")

    with st.form("predict_form"):
        st.subheader("Numerical Features")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=17, max_value=90, value=35)
        fnlwgt = c2.number_input("fnlwgt", min_value=1, max_value=999999, value=200000)
        education_num = c3.number_input(
            "Education Num", min_value=1, max_value=16, value=13
        )
        capital_gain = c1.number_input(
            "Capital Gain", min_value=0, max_value=99999, value=0
        )
        capital_loss = c2.number_input(
            "Capital Loss", min_value=0, max_value=99999, value=0
        )
        hours_per_week = c3.number_input(
            "Hours / Week", min_value=1, max_value=99, value=40
        )

        st.subheader("Categorical Features")
        c4, c5 = st.columns(2)
        workclass = c4.selectbox("Workclass", WORKCLASS_OPTS)
        education = c5.selectbox("Education", EDUCATION_OPTS, index=0)
        marital_status = c4.selectbox("Marital Status", MARITAL_OPTS)
        occupation = c5.selectbox("Occupation", OCCUPATION_OPTS)
        relationship = c4.selectbox("Relationship", RELATIONSHIP_OPTS)
        race = c5.selectbox("Race", RACE_OPTS)
        sex = c4.selectbox("Sex", SEX_OPTS)
        native_country = c5.selectbox("Native Country", COUNTRY_OPTS)

        submitted = st.form_submit_button("⚡ Predict", use_container_width=True)

    if not submitted:
        return

    payload = {
        "age": age,
        "fnlwgt": fnlwgt,
        "education_num": education_num,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "workclass": workclass,
        "education": education,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "native_country": native_country,
    }

    with st.spinner("Running inference…"):
        result = _post("/predict", payload)

    if not result or "error" in result:
        st.error(f"Prediction failed: {result}")
        return

    pred = result.get("prediction", result.get("predicted_class"))
    conf = result.get("confidence", 0)
    prob1 = result.get("probability_class_1", 0)
    label = ">50K 💰" if pred == 1 else "≤50K"
    color = "green" if pred == 1 else "orange"

    st.markdown(f"### Prediction: :{color}[**{label}**]")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{conf:.1%}")
    m2.metric("P(>50K)", f"{prob1:.1%}")
    m3.metric("Latency", f"{result.get('latency_ms', 0):.1f} ms")
    m4.metric("Model Version", result.get("model_version", "—"))

    if result.get("warnings"):
        st.warning("Warnings: " + " | ".join(result["warnings"]))

    # Gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob1 * 100, 1),
            title={"text": "P(income > 50K) %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#f0f0f0"},
                    {"range": [40, 60], "color": "#ffe082"},
                    {"range": [60, 100], "color": "#c8e6c9"},
                ],
                "threshold": {"line": {"color": "red", "width": 2}, "value": 50},
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ── Page: Live Demo ───────────────────────────────────────────────────────────


def page_live_demo():
    st.title("🚦 Live Demo — Traffic Generator")
    st.caption("Send bursts of predictions to generate Grafana traffic.")

    SAMPLE_PAYLOAD = {
        "age": 35,
        "workclass": "Private",
        "fnlwgt": 200000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    col1, col2 = st.columns(2)
    n_requests = col1.slider("Number of requests", 1, 100, 20)
    delay_ms = col2.slider("Delay between requests (ms)", 0, 500, 100)

    if st.button("🚀 Send Traffic Burst", use_container_width=True):
        progress = st.progress(0)
        results = []
        status = st.empty()

        for i in range(n_requests):
            r = _post("/predict", SAMPLE_PAYLOAD)
            results.append(r)
            progress.progress((i + 1) / n_requests)
            status.caption(f"Sent {i+1}/{n_requests}")
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)

        status.empty()
        errors = [r for r in results if r and "error" in r]
        success = [r for r in results if r and "error" not in r]
        confs = [r.get("confidence", 0) for r in success if r]
        class1 = sum(1 for r in success if r and r.get("prediction") == 1)

        st.success(f"✅ {len(success)} succeeded  |  ❌ {len(errors)} failed")
        m1, m2, m3 = st.columns(3)
        m1.metric("Class >50K", f"{class1}/{len(success)}")
        m2.metric("Mean Confidence", f"{sum(confs)/len(confs):.1%}" if confs else "—")
        m3.metric("Errors", len(errors))

    st.divider()
    st.subheader("Live API Metrics")
    snapshot = _get("/metrics/summary")
    if snapshot:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Requests", snapshot.get("requests_total", "—"))
        m2.metric("Error Rate", f"{snapshot.get('error_rate_per_sec', 0):.4f}/s")
        m3.metric(
            "Mean Confidence",
            (
                f"{snapshot.get('mean_confidence', 0):.1%}"
                if snapshot.get("mean_confidence")
                else "—"
            ),
        )
        m4.metric("Request Rate", f"{snapshot.get('request_rate_per_sec', 0):.2f}/s")
    else:
        st.warning("Could not reach /metrics/summary")

    if st.button("🔄 Refresh Metrics"):
        st.rerun()


# ── Page: Logs ────────────────────────────────────────────────────────────────


def page_logs():
    st.title("📋 Prediction Logs")

    col1, col2 = st.columns(2)
    hours = col1.selectbox("Look-back window", [1, 3, 6, 12, 24, 48], index=2)
    limit = col2.selectbox("Max records", [50, 100, 250, 500], index=1)

    if st.button("🔍 Fetch Logs", use_container_width=True):
        with st.spinner("Loading…"):
            data = _get(f"/logs?hours={hours}&limit={limit}")

        if not data:
            st.error("Could not reach /logs endpoint.")
            return

        records = data if isinstance(data, list) else data.get("logs", [])
        if not records:
            st.info("No logs found in this window.")
            return

        df = pd.DataFrame(records)

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", len(df))
        if "prediction" in df.columns:
            m2.metric("Class >50K", int((df["prediction"] == 1).sum()))
            m3.metric("Class ≤50K", int((df["prediction"] == 0).sum()))
        if "confidence" in df.columns:
            m4.metric("Mean Confidence", f"{df['confidence'].mean():.1%}")

        # Confidence histogram
        if "confidence" in df.columns:
            fig = go.Figure(
                go.Histogram(
                    x=df["confidence"],
                    nbinsx=20,
                    marker_color="#4CAF50",
                    opacity=0.75,
                )
            )
            fig.update_layout(
                title="Confidence Distribution",
                xaxis_title="Confidence",
                yaxis_title="Count",
                height=250,
                margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Table — show key columns only
        display_cols = [
            c
            for c in [
                "request_id",
                "timestamp",
                "prediction",
                "confidence",
                "model_version",
                "latency_ms",
                "ground_truth",
            ]
            if c in df.columns
        ]
        st.dataframe(df[display_cols], use_container_width=True, height=400)

        # Download
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False).encode(),
            file_name=f"logs_{hours}h.csv",
            mime="text/csv",
        )


# ── Page: Dashboard ───────────────────────────────────────────────────────────


def page_dashboard():
    st.title("📊 System Dashboard")

    # Key metric cards from API
    st.subheader("Live API Metrics")
    snapshot = _get("/metrics/summary")
    model = _get("/model/info")

    if snapshot:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Requests", snapshot.get("requests_total", "—"))
        m2.metric("Total Errors", snapshot.get("errors_total", "—"))
        m3.metric("Request Rate", f"{snapshot.get('request_rate_per_sec', 0):.2f}/s")
        m4.metric("Class 0 (≤50K)", snapshot.get("predictions_class_0_total", "—"))
        m5.metric("Class 1 (>50K)", snapshot.get("predictions_class_1_total", "—"))
    else:
        st.warning("API metrics unavailable — is the API running?")

    if model:
        st.subheader("Production Model")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version", model.get("version") or model.get("version_tag", "—"))
        c2.metric(
            "Val Accuracy",
            f"{model.get('val_accuracy', 0):.4f}" if model.get("val_accuracy") else "—",
        )
        c3.metric("Alias", model.get("alias", "—"))
        c4.metric("Registered At", (model.get("registered_at") or "—")[:19])

    st.divider()

    # Grafana iframe
    st.subheader("Grafana Dashboard")
    grafana_url = (
        f"{GRAFANA_BASE}/d/{GRAFANA_DASHBOARD_UID}" "?orgId=1&refresh=10s&kiosk=tv"
    )
    st.markdown(
        f'<iframe src="{grafana_url}" width="100%" height="700px" '
        f'frameborder="0"></iframe>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Can't see the dashboard? "
        f"[Open Grafana directly]({GRAFANA_BASE}) "
        f"(admin / admin)"
    )

    st.divider()

    # Service links
    st.subheader("Service Links")
    cols = st.columns(4)
    links = [
        ("Grafana", "http://localhost:3000"),
        ("Prometheus", "http://localhost:9090"),
        ("Airflow", "http://localhost:8080"),
        ("API Docs", "http://localhost:8000/docs"),
        ("cAdvisor", "http://localhost:8081"),
        ("ML Exporter", "http://localhost:9100/metrics"),
        ("Node Exporter", "http://localhost:9101/metrics"),
        ("API Health", "http://localhost:8000/health"),
    ]
    for i, (name, url) in enumerate(links):
        cols[i % 4].link_button(name, url, use_container_width=True)

    if st.button("🔄 Refresh"):
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="SentinelML",
        page_icon="🛰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _render_sidebar()

    pages = {
        "🤖 Chat": page_chat,
        "🎯 Predict": page_predict,
        "🚦 Live Demo": page_live_demo,
        "📋 Logs": page_logs,
        "📊 Dashboard": page_dashboard,
    }

    page = st.sidebar.radio(
        "Navigate", list(pages.keys()), label_visibility="collapsed"
    )
    st.sidebar.divider()
    st.sidebar.caption(f"SentinelML · {datetime.now().strftime('%H:%M:%S')}")

    pages[page]()


if __name__ == "__main__":
    main()
