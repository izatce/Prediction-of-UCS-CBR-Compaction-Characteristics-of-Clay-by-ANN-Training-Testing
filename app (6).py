# See README.md for workflow. This Streamlit app trains on one uploaded
# Excel file, then tests the trained model on a separately uploaded unseen file.

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from google import genai
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Soil Engineering Properties Prediction", page_icon="🌍", layout="wide")

TARGET_DEFAULT = "UCS (kPa)"
CANDIDATE_ARCHITECTURES = [(8,), (16,), (32,), (8,4), (16,8), (32,16), (64,32)]

def load_excel(file):
    df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def metrics(y, p):
    return {
        "R²": r2_score(y, p),
        "MAE": mean_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "MAPE (%)": mean_absolute_percentage_error(y, p) * 100,
    }

def build_model(architecture, max_iter, seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ann", MLPRegressor(
            hidden_layer_sizes=architecture,
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=seed,
        )),
    ])

def evaluate_architecture(X, y, architecture, max_iter, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        model = build_model(architecture, max_iter, seed + fold)
        model.fit(X.iloc[tr], y.iloc[tr])
        rows.append(metrics(y.iloc[va], model.predict(X.iloc[va])))
    d = pd.DataFrame(rows)
    return {
        "Architecture": str(architecture),
        "Hidden Layers": len(architecture),
        "Total Neurons": sum(architecture),
        "Mean R²": d["R²"].mean(),
        "Mean MAE": d["MAE"].mean(),
        "Mean RMSE": d["RMSE"].mean(),
        "Mean MAPE (%)": d["MAPE (%)"].mean(),
    }

def get_gemini_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")

st.title("🌍 Soil Engineering Properties Prediction Using ANN")
st.caption("Automatic ANN Optimization • 5-Fold CV • Independent Unseen Testing • SHAP")

if "trained" not in st.session_state:
    st.session_state.trained = None
if "optimization" not in st.session_state:
    st.session_state.optimization = None
if "test_result" not in st.session_state:
    st.session_state.test_result = None

st.sidebar.header("⚙️ ANN Configuration")
st.sidebar.success("🤖 Fully Automatic ANN Architecture Selection")
st.sidebar.write("Hidden layers cannot be manually changed.")
st.sidebar.code("\n".join(map(str, CANDIDATE_ARCHITECTURES)))
max_iter = st.sidebar.number_input("Maximum Training Iterations", 500, 5000, 2000, 100)
seed = st.sidebar.number_input("Random State", 0, 9999, 42, 1)

t1, t2, t3, t4, t5, t6 = st.tabs([
    "📘 1. Training Data",
    "🧠 2. Train ANN",
    "📊 3. CV Results",
    "📕 4. Unseen Testing",
    "🔍 5. Explainable AI",
    "🤖 6. Gemini Interpretation",
])

with t1:
    st.header("Stage 1: Upload Training / Development Data")
    train_file = st.file_uploader("📘 Upload Training Dataset", type=["xlsx", "xls"], key="train_file")
    if train_file:
        try:
            train_df = load_excel(train_file)
            st.session_state.train_df = train_df
            st.success(f"Loaded {len(train_df)} training samples.")
            st.dataframe(train_df.head(10), use_container_width=True)
            cols = list(train_df.columns)
            default_features = [c for c in cols if c not in [TARGET_DEFAULT, "S.No."]]
            features = st.multiselect("Select Input Features", cols, default=default_features, key="features")
            target = st.selectbox(
                "Select Output / Target Property",
                cols,
                index=cols.index(TARGET_DEFAULT) if TARGET_DEFAULT in cols else len(cols)-1,
                key="target",
            )
            # IMPORTANT:
            # The widgets already manage these values through:
            # st.session_state["features"] and st.session_state["target"].
            # Do not assign to those keys again after widget creation.
        except Exception as e:
            st.error(f"Could not read training file: {e}")

ready = (
    hasattr(st.session_state, "train_df")
    and hasattr(st.session_state, "features")
    and len(st.session_state.features) >= 2
    and hasattr(st.session_state, "target")
    and st.session_state.target not in st.session_state.features
)

if ready:
    raw = st.session_state.train_df[st.session_state.features + [st.session_state.target]].copy()
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=[st.session_state.target]).reset_index(drop=True)
    X = raw[st.session_state.features]
    y = raw[st.session_state.target]

with t2:
    st.header("Stage 2: Automatic ANN Training")
    if not ready:
        st.warning("Upload training data and select valid features first.")
    else:
        st.write(f"Training samples: **{len(X)}**")
        st.write("The app automatically tests multiple ANN architectures using 5-fold CV.")
        if st.button("🚀 Automatically Optimize and Train ANN", type="primary"):
            rows = []
            bar = st.progress(0)
            status = st.empty()
            try:
                for i, arch in enumerate(CANDIDATE_ARCHITECTURES):
                    status.write(f"Testing {arch} ...")
                    rows.append(evaluate_architecture(X, y, arch, int(max_iter), int(seed)))
                    bar.progress(int((i+1)/len(CANDIDATE_ARCHITECTURES)*100))
                opt = pd.DataFrame(rows).sort_values(
                    ["Mean RMSE", "Mean R²", "Total Neurons"],
                    ascending=[True, False, True],
                ).reset_index(drop=True)
                opt.insert(0, "Rank", range(1, len(opt)+1))
                lookup = {str(a): a for a in CANDIDATE_ARCHITECTURES}
                best_arch = lookup[opt.loc[0, "Architecture"]]
                status.write(f"Training final model with {best_arch} ...")
                model = build_model(best_arch, int(max_iter), int(seed))
                model.fit(X, y)
                st.session_state.optimization = opt
                st.session_state.trained = {
                    "model": model,
                    "architecture": best_arch,
                    "features": list(st.session_state.features),
                    "target": st.session_state.target,
                    "X": X.copy(),
                    "training_samples": len(X),
                }
                st.session_state.test_result = None
                bar.empty()
                status.empty()
                st.success("Model trained successfully. It is ready for independent unseen testing.")
            except Exception as e:
                st.error(f"Training error: {e}")
        if st.session_state.trained:
            r = st.session_state.trained
            st.metric("Automatically Selected Architecture", str(r["architecture"]))
            st.info("Now go to Tab 4 and upload the separate unseen testing file.")

with t3:
    st.header("Stage 3: 5-Fold Cross-Validation Results")
    if st.session_state.optimization is None:
        st.info("Train the model first.")
    else:
        opt = st.session_state.optimization
        st.dataframe(opt.style.format({
            "Mean R²": "{:.4f}",
            "Mean MAE": "{:.4f}",
            "Mean RMSE": "{:.4f}",
            "Mean MAPE (%)": "{:.2f}",
        }), use_container_width=True)
        best = opt.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Mean R²", f"{best['Mean R²']:.4f}")
        c2.metric("Mean RMSE", f"{best['Mean RMSE']:.4f}")
        c3.metric("Mean MAE", f"{best['Mean MAE']:.4f}")
        c4.metric("Mean MAPE", f"{best['Mean MAPE (%)']:.2f}%")

with t4:
    st.header("Stage 4: Independent Unseen Data Testing")
    if st.session_state.trained is None:
        st.warning("Train the ANN model first. Testing upload is used only after training.")
    else:
        st.success("Trained ANN model is ready. Upload unseen data below.")
        test_file = st.file_uploader("📕 Upload Unseen Testing Dataset", type=["xlsx", "xls"], key="test_file")
        if test_file:
            try:
                test_df = load_excel(test_file)
                trained = st.session_state.trained
                required = trained["features"] + [trained["target"]]
                missing = [c for c in required if c not in test_df.columns]
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                else:
                    st.write(f"Unseen testing samples uploaded: **{len(test_df)}**")
                    st.dataframe(test_df.head(10), use_container_width=True)
                    if st.button("🔬 Test Trained ANN Model", type="primary"):
                        # Preserve original rows and only remove rows lacking actual UCS.
                        work = test_df[required].copy()
                        for c in required:
                            work[c] = pd.to_numeric(work[c], errors="coerce")
                        valid = work[trained["target"]].notna()
                        work = work.loc[valid].copy()
                        Xtest = work[trained["features"]]
                        ytest = work[trained["target"]]
                        pred = trained["model"].predict(Xtest)
                        m = metrics(ytest, pred)
                        table = test_df.loc[work.index].copy()
                        table[f"Actual {trained['target']}"] = ytest.values
                        table[f"Predicted {trained['target']}"] = pred
                        table["Residual (Actual - Predicted)"] = ytest.values - pred
                        st.session_state.test_result = {
                            "metrics": m,
                            "actual": ytest.to_numpy(),
                            "predicted": pred,
                            "table": table,
                        }
                        st.success("Independent unseen testing completed.")
            except Exception as e:
                st.error(f"Testing error: {e}")

        if st.session_state.test_result:
            result = st.session_state.test_result
            m = result["metrics"]
            st.subheader("🎯 Final Unseen Test Performance")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Testing R²", f"{m['R²']:.4f}")
            c2.metric("Testing RMSE", f"{m['RMSE']:.4f}")
            c3.metric("Testing MAE", f"{m['MAE']:.4f}")
            c4.metric("Testing MAPE", f"{m['MAPE (%)']:.2f}%")

            actual, pred = result["actual"], result["predicted"]
            fig, ax = plt.subplots(figsize=(7,5))
            ax.scatter(actual, pred, alpha=0.75)
            lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
            ax.plot([lo,hi],[lo,hi], linestyle="--")
            ax.set_xlabel(f"Actual {trained['target']}")
            ax.set_ylabel(f"Predicted {trained['target']}")
            ax.set_title("Independent Unseen Test: Actual vs Predicted")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.dataframe(result["table"], use_container_width=True)
            st.download_button(
                "⬇️ Download Test Predictions",
                result["table"].to_csv(index=False).encode("utf-8"),
                "ucs_unseen_test_predictions.csv",
                "text/csv",
            )

with t5:
    st.header("Stage 5: Explainable AI (SHAP)")
    if st.session_state.trained is None:
        st.info("Train the ANN model first.")
    else:
        trained = st.session_state.trained
        st.caption(f"Explaining ANN architecture: {trained['architecture']}")
        if st.button("🔍 Generate SHAP Feature Importance"):
            try:
                with st.spinner("Generating SHAP analysis..."):
                    Xexp = trained["X"]
                    background = shap.sample(Xexp, min(50, len(Xexp)), random_state=int(seed))
                    explain_data = Xexp.sample(min(100, len(Xexp)), random_state=int(seed))
                    explainer = shap.Explainer(trained["model"].predict, background)
                    values = explainer(explain_data)
                    imp = pd.DataFrame({
                        "Feature": trained["features"],
                        "Mean |SHAP Value|": np.abs(values.values).mean(axis=0),
                    }).sort_values("Mean |SHAP Value|", ascending=False)
                    st.session_state.shap_importance = imp
            except Exception as e:
                st.error(f"SHAP error: {e}")
        if hasattr(st.session_state, "shap_importance"):
            imp = st.session_state.shap_importance
            st.dataframe(imp, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8,5))
            plot = imp.sort_values("Mean |SHAP Value|")
            ax.barh(plot["Feature"], plot["Mean |SHAP Value|"])
            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.set_title("Global Feature Importance")
            st.pyplot(fig)
        st.warning("Interpret related variables carefully. For example, PI = LL − PL.")

with t6:
    st.header("Stage 6: Gemini AI Interpretation")
    if st.session_state.test_result is None:
        st.info("Complete independent unseen testing first.")
    else:
        m = st.session_state.test_result["metrics"]
        st.write(f"Testing R²: **{m['R²']:.4f}**")
        st.write(f"Testing RMSE: **{m['RMSE']:.4f}**")
        st.write(f"Testing MAE: **{m['MAE']:.4f}**")
        st.write(f"Testing MAPE: **{m['MAPE (%)']:.2f}%**")
        if st.button("🤖 Generate Gemini Interpretation"):
            key = get_gemini_key()
            if not key:
                st.error("Gemini API key not found. Add GEMINI_API_KEY to Streamlit Secrets.")
            else:
                try:
                    features = ", ".join(st.session_state.trained["features"])
                    prompt = f"""You are assisting with geotechnical engineering research.
An ANN model predicted soil UCS from these features: {features}.
The model was optimized using 5-fold cross-validation on training data
and then independently tested on unseen data.
Unseen testing results are R²={m['R²']:.4f}, RMSE={m['RMSE']:.4f},
MAE={m['MAE']:.4f}, MAPE={m['MAPE (%)']:.2f}%.
Write a concise academic interpretation. State clearly that ANN produced
the numerical predictions and Gemini only provides interpretation."""
                    client = genai.Client(api_key=key)
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )
                    st.session_state.gemini_text = response.text
                except Exception as e:
                    st.error(f"Gemini error: {e}")
        if hasattr(st.session_state, "gemini_text"):
            st.markdown(st.session_state.gemini_text)

st.markdown("---")
st.caption("Soil UCS Prediction • Automatic ANN Optimization • 5-Fold CV • Independent Unseen Testing • SHAP")
