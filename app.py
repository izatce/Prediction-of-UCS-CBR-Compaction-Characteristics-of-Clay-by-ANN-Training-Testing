
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from google import genai
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Soil Engineering Properties Prediction",
    page_icon="🌍",
    layout="wide",
)

DEFAULT_TARGET = "UCS (kPa)"
ARCHITECTURES = [(8,), (16,), (32,), (8, 4), (16, 8), (32, 16), (64, 32)]


# ============================================================
# FUNCTIONS
# ============================================================
def load_excel(file):
    df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def calculate_metrics(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def build_model(architecture, max_iter, random_state):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "ann",
                MLPRegressor(
                    hidden_layer_sizes=architecture,
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_architecture(X, y, architecture, max_iter, random_state):
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        model = build_model(
            architecture,
            max_iter,
            random_state + fold,
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])

        fold_rows.append(
            calculate_metrics(y.iloc[val_idx], pred)
        )

    result = pd.DataFrame(fold_rows)

    return {
        "Architecture": str(architecture),
        "Mean R²": result["R²"].mean(),
        "Mean MAE": result["MAE"].mean(),
        "Mean RMSE": result["RMSE"].mean(),
        "Mean MAPE (%)": result["MAPE (%)"].mean(),
        "Total Neurons": sum(architecture),
    }


def clear_model_results():
    for key in [
        "trained_model",
        "optimization_results",
        "test_result",
        "shap_importance",
        "gemini_text",
    ]:
        st.session_state.pop(key, None)


def remove_target_from_features():
    """Callback runs before widgets are recreated."""
    target = st.session_state.get("target")
    features = st.session_state.get("features", [])

    if target in features:
        st.session_state["features"] = [
            f for f in features if f != target
        ]

    clear_model_results()


def get_gemini_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


# ============================================================
# INITIAL SESSION STATE
# ============================================================
for key, value in {
    "trained_model": None,
    "optimization_results": None,
    "test_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================
st.title("🌍 Soil Engineering Properties Prediction Using ANN")
st.caption(
    "Automatic ANN Optimization • 5-Fold Cross-Validation • "
    "Independent Unseen Testing • SHAP Explainable AI"
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Model Settings")
st.sidebar.success("🤖 ANN hidden layers are selected automatically.")
st.sidebar.caption(
    "The app compares several ANN architectures using 5-fold CV "
    "and selects the best one based primarily on Mean RMSE."
)

max_iter = st.sidebar.number_input(
    "Maximum Training Iterations",
    min_value=500,
    max_value=5000,
    value=2000,
    step=100,
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📘 1. Training Data",
        "🧠 2. Train ANN",
        "📊 3. CV Results",
        "📕 4. Unseen Testing",
        "🔍 5. Explainable AI",
        "🤖 6. Gemini Interpretation",
    ]
)

# ============================================================
# TAB 1
# ============================================================
with tab1:
    st.header("Stage 1: Upload and Configure Training Data")

    train_file = st.file_uploader(
        "📘 Upload Training Dataset",
        type=["xlsx", "xls"],
        key="train_file",
    )

    if train_file is not None:
        try:
            file_id = f"{train_file.name}_{train_file.size}"

            # Reset widgets/results only when a new file is uploaded.
            if st.session_state.get("train_file_id") != file_id:
                st.session_state["train_file_id"] = file_id
                st.session_state.pop("features", None)
                st.session_state.pop("target", None)
                clear_model_results()

            train_df = load_excel(train_file)
            st.session_state["train_df"] = train_df

            st.success(
                f"Training dataset loaded successfully: "
                f"{len(train_df)} samples"
            )

            st.dataframe(train_df.head(10), use_container_width=True)

            columns = list(train_df.columns)

            if len(columns) < 3:
                st.error(
                    "Dataset must contain at least two input columns "
                    "and one target column."
                )
            else:
                # IMPORTANT: SELECT TARGET FIRST
                # This fixes the previous issue where the selected
                # target could remain inside the input feature list.
                default_target_index = (
                    columns.index(DEFAULT_TARGET)
                    if DEFAULT_TARGET in columns
                    else len(columns) - 1
                )

                st.subheader("1. Select Output Property")

                target = st.selectbox(
                    "Select Output / Target Property",
                    options=columns,
                    index=default_target_index,
                    key="target",
                    on_change=remove_target_from_features,
                )

                st.subheader("2. Select Input Features")

                available_features = [
                    c for c in columns if c != target
                ]

                if "features" not in st.session_state:
                    excluded_ids = {
                        "S.No.",
                        "S.No",
                        "Sample No.",
                        "Sample ID",
                        "ID",
                    }

                    st.session_state["features"] = [
                        c for c in available_features
                        if c not in excluded_ids
                    ]

                # Make sure target is never in selected features.
                st.session_state["features"] = [
                    c for c in st.session_state["features"]
                    if c in available_features
                ]

                features = st.multiselect(
                    "Select Input Features",
                    options=available_features,
                    key="features",
                )

                if len(features) < 2:
                    st.warning(
                        "Please select at least two input features."
                    )
                else:
                    st.success(
                        f"Target: {target} | "
                        f"Selected inputs: {len(features)}"
                    )

                st.info(
                    "Important: The selected target property is "
                    "automatically excluded from the input features."
                )

        except Exception as e:
            st.error(f"Could not read training file: {e}")


# ============================================================
# PREPARE TRAINING DATA
# ============================================================
ready = (
    "train_df" in st.session_state
    and "target" in st.session_state
    and "features" in st.session_state
    and len(st.session_state["features"]) >= 2
    and st.session_state["target"] not in st.session_state["features"]
)

if ready:
    selected_target = st.session_state["target"]
    selected_features = list(st.session_state["features"])

    model_df = st.session_state["train_df"][
        selected_features + [selected_target]
    ].copy()

    for col in model_df.columns:
        model_df[col] = pd.to_numeric(
            model_df[col],
            errors="coerce",
        )

    # Target cannot be missing. Input missing values are imputed.
    model_df = model_df.dropna(
        subset=[selected_target]
    ).reset_index(drop=True)

    X_train = model_df[selected_features]
    y_train = model_df[selected_target]


# ============================================================
# TAB 2 - TRAIN
# ============================================================
with tab2:
    st.header("Stage 2: Automatic ANN Training")

    if not ready:
        st.warning(
            "First upload training data, select a target, "
            "and select at least two valid input features."
        )
    else:
        st.write(f"**Target Property:** {selected_target}")
        st.write(f"**Training Samples:** {len(X_train)}")
        st.write(
            f"**Input Features ({len(selected_features)}):** "
            + ", ".join(selected_features)
        )

        if len(X_train) < 10:
            st.error(
                "Too few samples for reliable 5-fold cross-validation."
            )
        else:
            if st.button(
                f"🚀 Optimize and Train ANN for {selected_target}",
                type="primary",
            ):
                progress = st.progress(0)
                status = st.empty()
                rows = []

                try:
                    for i, architecture in enumerate(ARCHITECTURES):
                        status.write(
                            f"Evaluating ANN architecture "
                            f"{architecture}..."
                        )

                        row = evaluate_architecture(
                            X_train,
                            y_train,
                            architecture,
                            int(max_iter),
                            int(random_state),
                        )

                        rows.append(row)

                        progress.progress(
                            int(
                                (i + 1)
                                / len(ARCHITECTURES)
                                * 100
                            )
                        )

                    optimization = (
                        pd.DataFrame(rows)
                        .sort_values(
                            [
                                "Mean RMSE",
                                "Mean R²",
                                "Total Neurons",
                            ],
                            ascending=[True, False, True],
                        )
                        .reset_index(drop=True)
                    )

                    optimization.insert(
                        0,
                        "Rank",
                        range(1, len(optimization) + 1),
                    )

                    best_arch_text = optimization.loc[
                        0, "Architecture"
                    ]

                    architecture_lookup = {
                        str(a): a for a in ARCHITECTURES
                    }

                    best_architecture = architecture_lookup[
                        best_arch_text
                    ]

                    status.write(
                        f"Training final ANN with "
                        f"{best_architecture}..."
                    )

                    final_model = build_model(
                        best_architecture,
                        int(max_iter),
                        int(random_state),
                    )

                    final_model.fit(X_train, y_train)

                    st.session_state["optimization_results"] = (
                        optimization
                    )

                    st.session_state["trained_model"] = {
                        "model": final_model,
                        "architecture": best_architecture,
                        "features": selected_features,
                        "target": selected_target,
                        "X_train": X_train.copy(),
                        "training_samples": len(X_train),
                    }

                    st.session_state["test_result"] = None
                    st.session_state.pop(
                        "shap_importance", None
                    )
                    st.session_state.pop(
                        "gemini_text", None
                    )

                    progress.empty()
                    status.empty()

                    st.success(
                        "✅ Model training completed successfully."
                    )

                except Exception as e:
                    progress.empty()
                    status.empty()
                    st.error(f"Training error: {e}")

        if st.session_state["trained_model"] is not None:
            trained = st.session_state["trained_model"]

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Target",
                trained["target"],
            )
            c2.metric(
                "Selected ANN",
                str(trained["architecture"]),
            )
            c3.metric(
                "Training Samples",
                trained["training_samples"],
            )

            st.success(
                "The model is ready for independent unseen testing."
            )


# ============================================================
# TAB 3 - CV RESULTS
# ============================================================
with tab3:
    st.header("Stage 3: 5-Fold Cross-Validation Results")

    if st.session_state["optimization_results"] is None:
        st.info("Train the model first.")
    else:
        optimization = st.session_state[
            "optimization_results"
        ]

        best = optimization.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean R²", f"{best['Mean R²']:.4f}")
        c2.metric(
            "Mean RMSE",
            f"{best['Mean RMSE']:.4f}",
        )
        c3.metric(
            "Mean MAE",
            f"{best['Mean MAE']:.4f}",
        )
        c4.metric(
            "Mean MAPE",
            f"{best['Mean MAPE (%)']:.2f}%",
        )

        st.dataframe(
            optimization.style.format(
                {
                    "Mean R²": "{:.4f}",
                    "Mean MAE": "{:.4f}",
                    "Mean RMSE": "{:.4f}",
                    "Mean MAPE (%)": "{:.2f}",
                }
            ),
            use_container_width=True,
        )


# ============================================================
# TAB 4 - UNSEEN TESTING
# ============================================================
with tab4:
    st.header("Stage 4: Independent Unseen Data Testing")

    if st.session_state["trained_model"] is None:
        st.warning(
            "Train the ANN model first."
        )
    else:
        trained = st.session_state["trained_model"]

        target = trained["target"]
        features = trained["features"]

        st.success(
            f"Trained model ready for: {target}"
        )

        test_file = st.file_uploader(
            "📕 Upload Independent Unseen Testing Dataset",
            type=["xlsx", "xls"],
            key="test_file",
        )

        if test_file is not None:
            try:
                test_df = load_excel(test_file)

                required_columns = features + [target]

                missing_columns = [
                    c for c in required_columns
                    if c not in test_df.columns
                ]

                if missing_columns:
                    st.error(
                        "Testing dataset is missing required columns: "
                        + ", ".join(missing_columns)
                    )
                else:
                    st.success(
                        f"Testing file loaded: {len(test_df)} samples"
                    )

                    st.dataframe(
                        test_df.head(10),
                        use_container_width=True,
                    )

                    if st.button(
                        f"🔬 Run Unseen Test for {target}",
                        type="primary",
                    ):
                        work = test_df[
                            required_columns
                        ].copy()

                        for col in required_columns:
                            work[col] = pd.to_numeric(
                                work[col],
                                errors="coerce",
                            )

                        work = work.dropna(
                            subset=[target]
                        )

                        if len(work) == 0:
                            st.error(
                                "No valid testing rows remain. "
                                "Check the target column."
                            )
                        else:
                            X_test = work[features]
                            y_test = work[target]

                            predictions = trained["model"].predict(
                                X_test
                            )

                            test_metrics = calculate_metrics(
                                y_test,
                                predictions,
                            )

                            output_table = test_df.loc[
                                work.index
                            ].copy()

                            output_table[
                                f"Actual {target}"
                            ] = y_test.values

                            output_table[
                                f"Predicted {target}"
                            ] = predictions

                            output_table[
                                "Residual (Actual - Predicted)"
                            ] = (
                                y_test.values - predictions
                            )

                            st.session_state["test_result"] = {
                                "metrics": test_metrics,
                                "actual": y_test.to_numpy(),
                                "predicted": predictions,
                                "table": output_table,
                                "target": target,
                            }

                            st.success(
                                "✅ Unseen testing completed successfully."
                            )

            except Exception as e:
                st.error(f"Testing error: {e}")

        # IMPORTANT:
        # Display results OUTSIDE the button and uploader block.
        # This ensures results remain visible after Streamlit reruns.
        if st.session_state["test_result"] is not None:
            result = st.session_state["test_result"]
            m = result["metrics"]
            current_target = result["target"]

            st.subheader(
                f"🎯 Final Output Results: {current_target}"
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Testing R²", f"{m['R²']:.4f}")
            c2.metric(
                "Testing RMSE",
                f"{m['RMSE']:.4f}",
            )
            c3.metric(
                "Testing MAE",
                f"{m['MAE']:.4f}",
            )
            c4.metric(
                "Testing MAPE",
                f"{m['MAPE (%)']:.2f}%",
            )

            actual = result["actual"]
            predicted = result["predicted"]

            st.subheader(
                f"Actual vs Predicted {current_target}"
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.scatter(
                actual,
                predicted,
                alpha=0.75,
            )

            low = min(
                float(np.min(actual)),
                float(np.min(predicted)),
            )

            high = max(
                float(np.max(actual)),
                float(np.max(predicted)),
            )

            ax.plot(
                [low, high],
                [low, high],
                linestyle="--",
            )

            ax.set_xlabel(
                f"Actual {current_target}"
            )
            ax.set_ylabel(
                f"Predicted {current_target}"
            )
            ax.set_title(
                f"Actual vs Predicted {current_target}"
            )
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            st.subheader("Prediction Table")

            st.dataframe(
                result["table"],
                use_container_width=True,
            )

            csv_data = (
                result["table"]
                .to_csv(index=False)
                .encode("utf-8")
            )

            safe_name = (
                current_target
                .replace(" ", "_")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")
            )

            st.download_button(
                "⬇️ Download Prediction Results",
                data=csv_data,
                file_name=f"{safe_name}_predictions.csv",
                mime="text/csv",
            )


# ============================================================
# TAB 5 - SHAP
# ============================================================
with tab5:
    st.header("Stage 5: Explainable AI (SHAP)")

    if st.session_state["trained_model"] is None:
        st.info("Train the ANN model first.")
    else:
        trained = st.session_state["trained_model"]

        st.info(
            f"Generating SHAP explanations for: "
            f"{trained['target']}"
        )

        if st.button(
            "🔍 Generate SHAP Feature Importance"
        ):
            try:
                with st.spinner(
                    "Generating SHAP analysis..."
                ):
                    X_explain = trained["X_train"]

                    background = shap.sample(
                        X_explain,
                        min(50, len(X_explain)),
                        random_state=int(random_state),
                    )

                    explain_data = X_explain.sample(
                        min(100, len(X_explain)),
                        random_state=int(random_state),
                    )

                    explainer = shap.Explainer(
                        trained["model"].predict,
                        background,
                    )

                    shap_values = explainer(
                        explain_data
                    )

                    importance = pd.DataFrame(
                        {
                            "Feature": trained["features"],
                            "Mean |SHAP Value|": np.abs(
                                shap_values.values
                            ).mean(axis=0),
                        }
                    ).sort_values(
                        "Mean |SHAP Value|",
                        ascending=False,
                    )

                    st.session_state[
                        "shap_importance"
                    ] = importance

                    st.success(
                        "SHAP analysis completed."
                    )

            except Exception as e:
                st.error(f"SHAP error: {e}")

        if "shap_importance" in st.session_state:
            importance = st.session_state[
                "shap_importance"
            ]

            st.dataframe(
                importance,
                use_container_width=True,
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            plot_data = importance.sort_values(
                "Mean |SHAP Value|",
                ascending=True,
            )

            ax.barh(
                plot_data["Feature"],
                plot_data["Mean |SHAP Value|"],
            )

            ax.set_xlabel(
                "Mean Absolute SHAP Value"
            )
            ax.set_title(
                f"Global Feature Importance: "
                f"{trained['target']}"
            )

            st.pyplot(fig)


# ============================================================
# TAB 6 - GEMINI
# ============================================================
with tab6:
    st.header("Stage 6: Gemini AI Interpretation")

    if st.session_state["test_result"] is None:
        st.info(
            "Complete independent unseen testing first."
        )
    else:
        result = st.session_state["test_result"]
        m = result["metrics"]
        target = result["target"]

        st.write(
            f"**Target Property:** {target}"
        )
        st.write(f"**R²:** {m['R²']:.4f}")
        st.write(f"**RMSE:** {m['RMSE']:.4f}")
        st.write(f"**MAE:** {m['MAE']:.4f}")
        st.write(
            f"**MAPE:** {m['MAPE (%)']:.2f}%"
        )

        if st.button(
            f"🤖 Interpret {target} with Gemini"
        ):
            api_key = get_gemini_key()

            if not api_key:
                st.error(
                    "Gemini API key not found. "
                    "Add GEMINI_API_KEY to Streamlit Secrets."
                )
            else:
                try:
                    trained = st.session_state[
                        "trained_model"
                    ]

                    prompt = f"""
You are assisting with geotechnical engineering research.

An ANN model was developed to predict:
{target}

Input features:
{", ".join(trained["features"])}

The ANN architecture was selected using 5-fold
cross-validation on the training dataset.

Independent unseen testing results:
R² = {m["R²"]:.4f}
RMSE = {m["RMSE"]:.4f}
MAE = {m["MAE"]:.4f}
MAPE = {m["MAPE (%)"]:.2f}%

Write a concise academic interpretation.
Do not claim causality.
State that ANN generated the numerical predictions and
Gemini only provides interpretation.
"""

                    client = genai.Client(
                        api_key=api_key
                    )

                    response = (
                        client.models.generate_content(
                            model="gemini-3.5-flash",
                            contents=prompt,
                        )
                    )

                    st.session_state[
                        "gemini_text"
                    ] = response.text

                except Exception as e:
                    st.error(f"Gemini error: {e}")

        if "gemini_text" in st.session_state:
            st.markdown(
                st.session_state["gemini_text"]
            )


st.markdown("---")
st.caption(
    "Soil Engineering Properties Prediction • "
    "Automatic ANN Optimization • 5-Fold CV • "
    "Independent Unseen Testing • SHAP"
)
