import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Moleculear MVP", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("molecule-ar_proprietary_failures_v0.csv")

df = load_data()

NUM_COLS = ["logP", "solubility", "clearance", "hERG_risk", "pd_signal"]

def build_model(dataframe):
    X = dataframe[NUM_COLS].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(Xs)
    return scaler, nn, Xs

scaler, nn, Xs = build_model(df)

st.title("Molecule-ar — Failure Intelligence MVP")
st.caption("Enter a candidate profile → retrieve similar historical failures → predict likely failure mode → suggest next best experiment.")

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1) New Molecule Intake")

    logP = st.slider("logP", 0.0, 6.0, 3.0, 0.1)
    solubility = st.number_input("Solubility (mg/mL)", min_value=0.0001, max_value=10.0, value=0.05, step=0.01, format="%.4f")
    clearance = st.slider("Clearance proxy", 5.0, 70.0, 30.0, 0.5)
    herg = st.slider("hERG risk (0–1)", 0.0, 1.0, 0.35, 0.01)
    pd_signal = st.slider("PD signal (0–1)", 0.0, 1.0, 0.50, 0.01)

    k = st.slider("How many similar historical molecules?", 3, 10, 5)
    analyze = st.button("Analyze Failure Intelligence", type="primary", use_container_width=True)

with right:
    st.subheader("2) Failure Intelligence Output")

    if analyze:
        # rebuild neighbors model with chosen k
        nn2 = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn2.fit(Xs)

        x = np.array([[logP, solubility, clearance, herg, pd_signal]])
        xs = scaler.transform(x)

        dist, idx = nn2.kneighbors(xs)
        neighbors = df.iloc[idx[0]].copy()
        neighbors["distance"] = dist[0]

        # Vote on failure mode weighted by inverse distance
        eps = 1e-6
        neighbors["w"] = 1.0 / (neighbors["distance"] + eps)
        scores = neighbors.groupby("failure_mode")["w"].sum().sort_values(ascending=False)

        top_mode = scores.index[0]
        confidence = float(scores.iloc[0] / scores.sum()) if scores.sum() > 0 else 0.0

        # pick most common suggested experiment among top-mode neighbors
        next_exp = neighbors[neighbors["failure_mode"] == top_mode]["next_best_experiment"].mode()
        next_exp = next_exp.iloc[0] if len(next_exp) else "Run follow-up assay to confirm risk."

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Failure Mode", top_mode)
        c2.metric("Confidence", f"{confidence*100:.1f}%")
        c3.metric("Dataset Size", f"{len(df)} molecules")

        st.markdown("### Suggested next best experiment")
        st.info(next_exp)

        st.markdown("### Similar historical molecules (your failure-memory dataset)")
        st.dataframe(
            neighbors[["mol_id", "failure_mode", "next_best_experiment", "logP", "solubility", "clearance", "hERG_risk", "pd_signal", "distance"]],
            use_container_width=True,
            height=260
        )

        st.markdown("### Branch Trace (mini decision tree)")
        if top_mode in ["LOW_SOLUBILITY", "HIGH_CLEARANCE"]:
            st.code(
                "Node: Developability risk detected\n"
                " ├─ Option A: Optimize properties → Re-test → (Advance or Kill)\n"
                " ├─ Option B: Formulation route → Exposure test → (Advance or Kill)\n"
                " └─ Option C: Stop series → Record failure signature\n"
            )
        elif top_mode == "HERG_QT":
            st.code(
                "Node: Cardiac risk detected\n"
                " ├─ Option A: Ion-channel selectivity → Confirm mechanism\n"
                " ├─ Option B: Structural change → Re-test hERG\n"
                " └─ Option C: Terminate → Capture liability pattern\n"
            )
        else:
            st.code(
                "Node: Weak PD signal\n"
                " ├─ Option A: Improve target engagement assay\n"
                " ├─ Option B: Adjust exposure → PK-PD linkage study\n"
                " └─ Option C: Stop → Capture 'no-PD' pattern\n"
            )

    else:
        st.write("Click **Analyze Failure Intelligence** to see results.")

st.divider()
with st.expander("Why this counts as a product (not a feature)"):
    st.write(
        "This MVP demonstrates the core Moleculear idea: a reusable failure-memory dataset + decision logic.\n\n"
        "In real pharma deployment, the dataset is populated from ELN/LIMS/warehouse history, and the same logic produces "
        "auditable, comparable decision traces across programs."
    )

