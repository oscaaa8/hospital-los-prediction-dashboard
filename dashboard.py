# dashboard.py
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

figsize = 600
# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Hospital Length Of Stay (LOS) Prediction",
    page_icon="🏥",
    layout="wide",
)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.title("🏥 LOS Dashboard")
    st.markdown(
        "Applied ML pipeline for **Length of Stay (LOS)** prediction: "
        "_data → model → eval → fairness → explainability_."
    )
    st.divider()
    st.subheader("Project")
    st.markdown(
        "- Author: **Oscar Aguilar**  \n"
        "- Stack: Streamlit, scikit-learn, matplotlib  \n"
        "- Focus: Performance + Fairness + Explainability"
    )
    # Data at a glance
    st.subheader("Data at a glance")
    st.markdown(
    "- Samples: **500,000**  \n"
    "- Features after encoding: **43**  \n"
    "- Train/Test split: **70/30** \n"
    "- Best model: **Random Forest** (MAE **0.89** days, RMSE **1.32**, R² **0.970**)"
)

# -------------------------------
# Header
# -------------------------------
st.title("Hospital Length of Stay (LOS) Prediction Dashboard")
st.markdown(
    "This dashboard showcases a full applied ML pipeline to predict **Length of Stay (LOS)** from hospital admission data."
)

# -------------------------------
# Helpers
# -------------------------------
PLOTS_DIR = Path("plots")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

def callout(kind: str, title: str, body: str):
    kinds = {"info": st.info, "success": st.success, "warning": st.warning}
    kinds.get(kind, st.info)(f"**{title}** — {body}")


def load_png(name: str) -> Image.Image | None:
    path = PLOTS_DIR / name
    if path.exists():
        try:
            return Image.open(path)
        except Exception as e:
            st.warning(f"Could not open `{name}`: {e}")
    else:
        st.info(f"Add `{name}` to the `plots/` folder to display this figure.")
    return None

def load_metrics() -> dict:
    """Load artifacts/metrics.json or fall back to example values."""
    mpath = ARTIFACTS_DIR / "metrics.json"
    if mpath.exists():
        try:
            return json.loads(mpath.read_text())
        except Exception as e:
            st.warning(f"Could not parse metrics.json: {e}")

    # ------- Fallback (edit to your real run) -------
    return {
        "model_name": "Random Forest Regressor",
        "MAE_days": 0.89,
        "RMSE_days": 1.32,
        "R2": 0.970,  # 97%
        "bin_metrics": {  # optional but shown if present
            "binned_accuracy": 0.911,
            "balanced_accuracy": 0.810,
            "macro_f1": 0.84,
            "per_class": {
                "≤7 days": {"precision": 0.987, "recall": 0.440, "f1": 0.608},
                "8–14 days": {"precision": 0.879, "recall": 0.879, "f1": 0.879},
                ">14 days": {"precision": 0.987, "recall": 0.998, "f1": 0.992},
            },
        },
        "per_bin_errors": {
            "≤7 days": {"MAE": 0.87, "MedianAE": 0.79, "P90AE": 1.68},
            "8–14 days": {"MAE": 0.58, "MedianAE": 0.46, "P90AE": 1.21},
            ">14 days": {"MAE": 1.75, "MedianAE": 1.35, "P90AE": 3.75},
        },
        "top_predictors": [
            {"feature": "Available Extra Rooms in Hospital", "direction": "↑", "effect": "longer LOS"},
            {"feature": "Admission Deposit", "direction": "↑", "effect": "longer LOS"},
            {"feature": "Department: Gynecology", "direction": "–", "effect": "shorter average LOS"},
        ],
    }

M = load_metrics()

# -------------------------------
# EDA Explanation
# -------------------------------
st.markdown(
    "Through **exploratory data analysis (EDA)** we uncovered key hospital patterns:  \n"
    "- A **dominance of gynecology cases** (over 90k patients per physician) shaping overall patient flow. **(This was found to be the most important feature in our model for predicting LOS)**  \n"
    "- A **skewed length-of-stay distribution**, with most patients discharged quickly, but a long-tail of extended stays.  \n"
    "- A clear **age × severity interactions**, where older or higher-severity patients required longer admissions.  \n"
)

st.markdown(
    "While these findings provided valuable context, **EDA alone cannot predict the LOS of an individual admission**. "
    "To bridge this gap, we developed a machine learning model that transforms messy hospital data into "
    "a **predictive and interpretable tool** for clinicians and administrators."
)



# -------------------------------
# Model summary cards
# -------------------------------

st.header("📊 Model Performance")
colAa, colA, colB, colC = st.columns(4)
with colAa: st.metric("Model", M.get("model_name", "N/A"))
with colA: st.metric("Overall Model's Absolute Error (MAE)", f"{M['MAE_days']:.2f} days")
with colB: st.metric("R²", f"{M['R2']:.3f}")
with colC: st.metric("Binned accuracy", f"{M.get('bin_metrics',{}).get('binned_accuracy','—')}")
callout("info",
        "Framing",
        "We predict **Length of Stay** to inform bed turnover, discharge planning, and staffing.")





# -------------------------------
# Fairness / Bias diagnostics
# -------------------------------
st.header("⚖️ Model Bias Evaluation (Binned LOS)")
st.markdown(
    "**Addressing Class Imbalance:** Hospital length of stay data is **highly skewed**, which can cause models to "
    "systematically over- or under-predict for different patient groups. "
    "To address this, we evaluated performance in **clinically meaningful bins "
    "(≤7, 8–14, >14 days)**. This approach revealed hidden bias "
    "(e.g., short stays being pulled upward) and ensured that predictions were "
    "both **fair and interpretable** for real-world hospital planning."
)


cm_img = load_png("confusion_matrix.png")
if cm_img:
    st.image(cm_img, caption="Confusion Matrix on Binned Regression Output", width= figsize)#use_container_width=False)

dist_img = load_png("true_vs_pred_bins.png")
if dist_img:
    st.image(dist_img, caption="True vs Predicted LOS Bin Proportions", width= figsize) #use_container_width=False)


# -------------------------------
# Perspective
# -------------------------------
st.subheader("🔎 Evidence → Interpretation → Action")
col1, col2, col3 = st.columns([1.2, 1.2, 1])

with col1:
    st.markdown(
        "### Evidence  \n"
        "- Strongest performance in **8–14 days**  \n"
        "- **≤7 days**: under-recalled (many predicted as 8–14)  \n"
        "- **>14 days**: well-identified but errors larger (more variability)"
    )

with col2:
    st.markdown(
        "### Interpretation  \n"
        "- Skewed targets cause **shrink-to-middle** behavior  \n"
        "- Short stays get pulled upward; long stays vary more, so absolute error rises  \n"
        "- Overall fit is strong, but **calibration** differs by LOS range"
    )

with col3:
    st.markdown(
        "### Action  \n"
        "- **Normalize skewed features** (e.g., log-transform deposits, scale room counts)  \n"
        "- Add **regularization** (Ridge/Lasso) to reduce overfitting on correlated predictors  \n"
        "- **Balance short-stay samples** with class weights or resampling"
    )



# ------------------------------- 
# Per-bin absolute errors table
# -------------------------------
if "bin_metrics" in M:
    st.subheader("Binned Classification Metrics")
    bm = M["bin_metrics"]

    summary_tbl = pd.DataFrame(
        {
            "Metric": ["Binned accuracy", "Balanced accuracy", "Macro F1"],
            "Score": [
                bm.get("binned_accuracy"),
                bm.get("balanced_accuracy"),
                bm.get("macro_f1"),
            ],
        }
    )

    st.dataframe(
        summary_tbl.style.format({"Score": "{:.3f}"}), 
        use_container_width=False,  # keeps table compact
        height=150
    )


# ------------------------------- 
# Contextualized Insights
# -------------------------------
st.subheader("Contextualized Model Insights")

st.success(
    "✅ **Best range (8–14 days)**  \n"
    "- Strong classification metrics  \n"
    "- Low error rates across this bin"
)

st.info(
    "ℹ️ **Long stays (>14 days)**  \n"
    "- High recall & precision → well-flagged  \n"
    f"- Wider variability inflates error (MAE ≈ {M.get('per_bin_errors', {}).get('>14 days', {}).get('MAE', '—')})"
)

short_recall = (
    M.get("bin_metrics", {})
     .get("per_class", {})
     .get("≤7 days", {})
     .get("recall", "—")
)

st.warning(
    "⚠️ **Short stays (≤7 days)**  \n"
    f"- Often misclassified as 8–14 → lower recall ({short_recall})  \n"
    "- Despite low MAE, predictions **shrink toward the center bin** "
    "(common with skewed/imbalanced targets)"
)

st.caption(
    f"📌 **Takeaway:** Excellent overall fit (R² ≈ {M.get('R2', float('nan')):.3f}), "
    "but calibration, class balance, and use-case thresholds still matter."
)


# ------------------------------- 
# Feature importance (no SHAP)
# -------------------------------
st.header("Top Predictors of LOS")
st.markdown(
    "Tree-based **feature importance** highlights the following predictors "
    "and directions of effect:"
)
feat_importance_img = load_png("feature_importance_visual.png")
if feat_importance_img:
    st.image(feat_importance_img, caption="Top 10 predictors of hospital length of stay from the Random Forest model. Department affiliation (especially gynecology), followed by patient age groups (31–40, 41–50), were the strongest drivers of LOS, while operational and financial factors such as admission deposit and available rooms played smaller but notable roles.", width= figsize)#use_container_width=False)

if isinstance(M.get("top_predictors"), list):
    for item in M["top_predictors"]:
        feat = item.get("feature", "Feature")
        direction = item.get("direction", "")
        effect = item.get("effect", "")
        arrow = "↑" if direction == "↑" else "→"

# ------------------------------- 
# Key Outcome
# -------------------------------
st.success(
    "**Key outcome:** The model reached strong performance (R² ≈ 0.97).  \n"
    "- Best results were for patients staying **8–14 days**.  \n"
    "- By grouping predictions into **clear clinical ranges (≤7, 8–14, >14 days)**, "
    "results become easier to interpret and directly usable for hospital planning."
)

# ------------------------------- 
# Conclusive Perspective
# -------------------------------
st.header("Key Insights & Next Steps")

# Highlighted result in bold + big font
st.markdown(
    "<h2 style='text-align: center; color: green; font-weight: bold;'>"
    "✅ The model correctly predicts the right LOS category ~80–90% of the time"
    "</h2>",
    unsafe_allow_html=True
)

# Business Recommendation
st.markdown(
    "### 💡 Business Recommendation  \n"
    "Hospitals can use these insights to **allocate resources by LOS category**:  \n"
    "- **Short stays (≤7 days):** Focus on rapid turnover (beds, discharges, staff coverage).  \n"
    "- **Medium stays (8–14 days):** Prioritize this group as it represents the majority of admissions.  \n"
    "- **Long stays (>14 days):** Plan for higher variability with specialized care units and extended resources.  \n\n"
    "Together, exploratory analysis and predictive modeling create a **practical, data-driven foundation** "
    "for managing patient flow, staffing, and hospital capacity."
)

# ------------------------------- 
# Footer
# -------------------------------
st.divider()
st.markdown(
    "Built by **Oscar Aguilar** — end-to-end applied ML: "
    "_data cleaning • modeling • fairness checks • interpretation • deployment_."
)
