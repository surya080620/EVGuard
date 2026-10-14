import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, r2_score,
    mean_absolute_error, mean_squared_error, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="EV ChargeLens | ML Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).resolve().parent / "EV_Charging_Station.xlsx"
RANDOM_STATE = 42
MAX_TRAIN_ROWS_DEFAULT = 20000
MAX_PLOT_ROWS = 5000

st.markdown("""
<style>
.stApp {background: #081225; color: #f3f7ff;}
[data-testid="stSidebar"] {background: #101b30;}
.block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
.hero {padding: 26px 30px; border: 1px solid #24486e; border-radius: 20px;
background: linear-gradient(120deg,#102a4c,#123a59); margin-bottom: 18px;}
.hero h1 {color:#fff; margin-bottom:8px;}
.hero p {color:#c4d8ef; margin:0;}
.small-note {color:#a7bad3; font-size:0.9rem;}
div[data-testid="stMetric"] {background:#111f34; border:1px solid #293b55;
padding:14px 16px; border-radius:14px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Data loading / utility functions
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path_string: str, modified_time: float):
    """Read the fixed local Excel file once per file version."""
    data = pd.read_excel(path_string, engine="openpyxl")
    data.columns = [str(c).strip() for c in data.columns]

    # Convert dates robustly, including Excel serial dates.
    if "Open Date" in data.columns:
        raw = data["Open Date"]
        parsed = pd.to_datetime(raw, errors="coerce")
        numeric = pd.to_numeric(raw, errors="coerce")
        excel_dates = pd.to_datetime(
            numeric, unit="D", origin="1899-12-30", errors="coerce"
        )
        data["Open Date"] = parsed.fillna(excel_dates)
        data["Open Year"] = data["Open Date"].dt.year
        data["Open Month"] = data["Open Date"].dt.month
        data = data.drop(columns=["Open Date"])

    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].replace(r"^\s*$", np.nan, regex=True)

    # Drop completely empty columns, but preserve all other columns.
    data = data.dropna(axis=1, how="all")
    return data


def load_fixed_data():
    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found: `{DATA_PATH.name}`. Place the Excel file in the "
            "same folder as `app.py`."
        )
        st.stop()
    try:
        return load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not read `{DATA_PATH.name}`: {exc}")
        st.info("Check that the workbook is valid and that `openpyxl` is installed.")
        st.stop()


df = load_fixed_data()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
classification_targets = [
    c for c in ["Status Code", "Profit_Potential", "Competition_Level"] if c in df.columns
]
regression_targets = [
    c for c in [
        "Opportunity_Score", "Nearest_Station_Distance_km",
        "EV Level2 EVSE Num", "EV DC Fast Count"
    ] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
]
if not classification_targets:
    classification_targets = [
        c for c in df.columns if df[c].nunique(dropna=True) <= 30 and df[c].nunique(dropna=True) >= 2
    ][:8]
if not regression_targets:
    regression_targets = [c for c in numeric_cols if df[c].nunique(dropna=True) > 10][:8]


def plotly_chart(fig, key=None):
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def sample_frame(data, max_rows=MAX_PLOT_ROWS, random_state=RANDOM_STATE):
    if len(data) <= max_rows:
        return data
    return data.sample(n=max_rows, random_state=random_state)


def safe_text(value):
    if pd.isna(value):
        return "Missing"
    return str(value)


def feature_pipeline(X, scale=False):
    nums = X.select_dtypes(include=np.number).columns.tolist()
    cats = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if nums:
        transformers.append(("num", Pipeline(numeric_steps), nums))
    if cats:
        # Keep high-cardinality categorical features manageable.
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ])
        transformers.append(("cat", cat_pipe, cats))

    return ColumnTransformer(transformers, remainder="drop")


def limit_training_data(X, y, max_rows, classification=False):
    """Reproducible, disclosed sample for responsiveness on large workbooks."""
    if len(X) <= max_rows:
        return X, y, False
    if classification:
        combined = X.copy()
        combined["__target__"] = y.to_numpy()
        # Sample by class where possible, then cap.
        pieces = []
        counts = y.value_counts(dropna=False)
        per_class = max(1, max_rows // max(1, len(counts)))
        for label in counts.index:
            group = combined[combined["__target__"] == label]
            pieces.append(group.sample(n=min(len(group), per_class), random_state=RANDOM_STATE))
        sampled = pd.concat(pieces, axis=0)
        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=RANDOM_STATE)
        ys = sampled.pop("__target__")
        return sampled, ys, True
    indices = X.sample(n=max_rows, random_state=RANDOM_STATE).index
    return X.loc[indices], y.loc[indices], True


def choose_features(default_names, target, exclude_labels=True):
    excluded = {target}
    if exclude_labels:
        excluded.update(
            {"Status Code", "Profit_Potential", "Competition_Level", "Opportunity_Score"} - {target}
        )
    return [c for c in df.columns if c not in excluded]


def feature_plot(data, feature, chart_type, top_n=20):
    series = data[feature]
    if pd.api.types.is_numeric_dtype(series):
        valid = pd.to_numeric(series, errors="coerce")
        temp = pd.DataFrame({feature: valid}).dropna()
        if chart_type in ("Histogram", "Distribution"):
            fig = px.histogram(temp, x=feature, nbins=40, marginal="box",
                               title=f"Distribution of {feature}")
        elif chart_type == "Box plot":
            fig = px.box(temp, y=feature, points="outliers", title=f"Box plot: {feature}")
        elif chart_type == "Violin plot":
            fig = px.violin(temp, y=feature, box=True, points="outliers",
                            title=f"Violin plot: {feature}")
        elif chart_type == "Density plot":
            fig = px.histogram(temp, x=feature, histnorm="probability density",
                               nbins=40, title=f"Density of {feature}")
        elif chart_type == "Trend by row":
            temp = temp.reset_index(drop=True).reset_index(names="Record")
            temp = sample_frame(temp, MAX_PLOT_ROWS)
            fig = px.line(temp, x="Record", y=feature, title=f"{feature} by record order")
        elif chart_type == "Missing values":
            counts = pd.DataFrame({
                "Value": ["Present", "Missing"],
                "Count": [int(series.notna().sum()), int(series.isna().sum())]
            })
            fig = px.bar(counts, x="Value", y="Count", text_auto=True,
                         title=f"Missing values: {feature}")
        else:
            fig = px.histogram(temp, x=feature, nbins=40, title=f"{feature}")
    else:
        counts = (
            series.fillna("Missing").astype(str).value_counts()
            .head(top_n).rename_axis(feature).reset_index(name="Count")
        )
        if chart_type == "Pie / donut":
            fig = px.pie(counts, names=feature, values="Count", hole=0.42,
                         title=f"Category proportions: {feature}")
        elif chart_type == "Missing values":
            missing = pd.DataFrame({
                "Value": ["Present", "Missing"],
                "Count": [int(series.notna().sum()), int(series.isna().sum())]
            })
            fig = px.bar(missing, x="Value", y="Count", text_auto=True,
                         title=f"Missing values: {feature}")
        else:
            fig = px.bar(counts, x=feature, y="Count", text_auto=True,
                         title=f"Top {min(top_n, len(counts))} categories: {feature}")
            fig.update_layout(xaxis_tickangle=-35)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_metric_comparison(results, metric, higher_is_better=True):
    valid = results.dropna(subset=[metric]).copy()
    if valid.empty:
        st.warning(f"No valid results are available to plot for {metric}.")
        return
    valid = valid.sort_values(metric, ascending=not higher_is_better)
    fig = px.bar(
        valid, x="Algorithm", y=metric, color=metric, text=metric,
        title=f"Algorithm comparison — {metric}"
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_tickangle=-25, yaxis=dict(automargin=True), margin=dict(b=90))
    plotly_chart(fig)


def add_data_quality_warning(y):
    counts = y.value_counts(dropna=False)
    if len(counts) < 2:
        st.error("The selected target has fewer than two classes. Classification is not possible.")
        return False
    if counts.min() < 2:
        st.warning("At least one class has fewer than two rows; a stratified split may not be possible.")
    if counts.max() / counts.sum() >= 0.95:
        st.warning("Severe class imbalance detected. Accuracy alone may be misleading; inspect macro F1 and the confusion matrix.")
    return True


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ EV ChargeLens")
    st.caption("EV Charging Station ML Analytics")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Feature Explorer", "Predict Status", "Classification",
         "Regression", "Clustering", "About Project"],
    )
    st.markdown("---")
    st.success(f"Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
    st.caption("Fixed local dataset • no file upload")
    st.caption(f"File: {DATA_PATH.name}")


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <p style="letter-spacing:2px;font-weight:700;color:#60e0ee;">EV CHARGING • MACHINE LEARNING</p>
      <h1>Explore the EV Charging Station Dataset</h1>
      <p>Analyze station availability, charging infrastructure, data quality, and investment opportunities with interactive Plotly charts and ML models.</p>
    </div>
    """, unsafe_allow_html=True)

    missing_cells = int(df.isna().sum().sum())
    state_col = next((c for c in ["State", "state", "Province", "Region"] if c in df.columns), None)
    state_count = df[state_col].nunique(dropna=True) if state_col else None
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", f"{len(df):,}")
    m2.metric("Features", f"{len(df.columns):,}")
    m3.metric("Missing cells", f"{missing_cells:,}")
    m4.metric("States / regions", f"{state_count:,}" if state_count is not None else "N/A")

    n1, n2, n3 = st.columns(3)
    n1.metric("Numeric features", len(numeric_cols))
    n2.metric("Categorical features", len(categorical_cols))
    n3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Dataset preview")
    st.dataframe(df.head(15), use_container_width=True, height=330)

    left, right = st.columns(2)
    with left:
        status_col = next((c for c in ["Status Code", "Station Status", "Status"] if c in df.columns), None)
        if status_col:
            vc = df[status_col].fillna("Missing").astype(str).value_counts().head(20)
            plotly_chart(px.bar(vc.rename_axis("Status").reset_index(name="Stations"),
                                x="Status", y="Stations", text_auto=True,
                                title="Station status distribution"))
        elif categorical_cols:
            c = categorical_cols[0]
            vc = df[c].fillna("Missing").astype(str).value_counts().head(15)
            plotly_chart(px.bar(vc.rename_axis(c).reset_index(name="Count"),
                                x=c, y="Count", text_auto=True,
                                title=f"Top categories — {c}"))
    with right:
        facility_col = next((c for c in ["Facility Type", "Facility_Type"] if c in df.columns), None)
        if facility_col:
            vc = df[facility_col].fillna("Missing").astype(str).value_counts().head(12)
            plotly_chart(px.bar(vc.rename_axis("Facility Type").reset_index(name="Stations"),
                                x="Facility Type", y="Stations", text_auto=True,
                                title="Top facility types"))
        elif numeric_cols:
            c = numeric_cols[0]
            plotly_chart(px.histogram(sample_frame(df[[c]].dropna()), x=c, nbins=35,
                                      title=f"Distribution — {c}"))

    st.subheader("Data quality")
    missing_summary = pd.DataFrame({
        "Feature": df.columns,
        "Missing values": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2),
        "Unique values": [df[c].nunique(dropna=True) for c in df.columns],
        "Data type": [str(df[c].dtype) for c in df.columns],
    }).sort_values("Missing values", ascending=False)
    plotly_chart(px.bar(missing_summary.head(20), x="Feature", y="Missing values",
                        title="Features with the most missing values", text="Missing values"))
    with st.expander("Full feature quality table"):
        st.dataframe(missing_summary, use_container_width=True, hide_index=True)

    if len(numeric_cols) >= 2:
        st.subheader("Numeric correlation heatmap")
        corr = df[numeric_cols].corr(numeric_only=True)
        plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto",
                               color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                               title="Correlation between numeric features"))


# ---------------------------------------------------------------------
# Feature Explorer
# ---------------------------------------------------------------------
elif page == "Feature Explorer":
    st.markdown("## Feature Explorer")
    st.write("Explore a single feature or select **All Features** for a compact dataset-wide report.")
    feature_options = ["All Features"] + df.columns.tolist()
    feature = st.selectbox("Select feature", feature_options)

    if feature == "All Features":
        st.subheader("Dataset-wide feature analysis")
        st.caption("Charts use aggregated counts or a reproducible sample where appropriate. The original dataset remains unchanged.")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Numeric distributions", "Categorical distributions",
            "Missing values", "Correlations", "Feature quality"
        ])

        with tab1:
            if numeric_cols:
                cols_to_show = st.multiselect(
                    "Numeric features to plot", numeric_cols,
                    default=numeric_cols[:min(6, len(numeric_cols))]
                )
                if cols_to_show:
                    chart_cols = st.columns(2)
                    for i, col in enumerate(cols_to_show):
                        with chart_cols[i % 2]:
                            values = pd.to_numeric(df[col], errors="coerce").dropna()
                            if len(values):
                                plotly_chart(px.histogram(
                                    pd.DataFrame({col: values.sample(
                                        n=min(len(values), 30000), random_state=RANDOM_STATE
                                    )}), x=col, nbins=35, marginal="box",
                                    title=f"Distribution — {col}"
                                ), key=f"all_numeric_{col}")
                    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
                else:
                    st.info("Select at least one numeric feature.")
            else:
                st.info("No numeric features are available.")

        with tab2:
            if categorical_cols:
                top_n = st.slider("Top categories per feature", 5, 30, 12, key="all_cat_top_n")
                selected_cats = st.multiselect(
                    "Categorical features to plot", categorical_cols,
                    default=categorical_cols[:min(4, len(categorical_cols))]
                )
                if selected_cats:
                    chart_cols = st.columns(2)
                    for i, col in enumerate(selected_cats):
                        with chart_cols[i % 2]:
                            counts = df[col].fillna("Missing").astype(str).value_counts().head(top_n)
                            plotly_chart(px.bar(
                                counts.rename_axis("Category").reset_index(name="Count"),
                                x="Category", y="Count", text_auto=True,
                                title=f"Category frequency — {col}"
                            ), key=f"all_cat_{col}")
                    cat_summary = pd.DataFrame({
                        "Feature": categorical_cols,
                        "Unique values": [df[c].nunique(dropna=True) for c in categorical_cols],
                        "Missing values": [int(df[c].isna().sum()) for c in categorical_cols],
                        "Most common value": [
                            str(df[c].mode(dropna=True).iloc[0]) if not df[c].mode(dropna=True).empty else "N/A"
                            for c in categorical_cols
                        ],
                    })
                    st.dataframe(cat_summary, use_container_width=True, hide_index=True)
                else:
                    st.info("Select at least one categorical feature.")
            else:
                st.info("No categorical features are available.")

        with tab3:
            missing = pd.DataFrame({
                "Feature": df.columns,
                "Missing values": df.isna().sum().values,
                "Missing %": (df.isna().mean().values * 100).round(2),
            }).sort_values("Missing %", ascending=False)
            plotly_chart(px.bar(missing, x="Feature", y="Missing %",
                                title="Missing percentage by feature", text="Missing %"))
            plotly_chart(px.bar(missing, x="Feature", y="Missing values",
                                title="Missing count by feature", text="Missing values"))
            st.dataframe(missing, use_container_width=True, hide_index=True)

        with tab4:
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr(numeric_only=True)
                plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto",
                                       color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                       title="Numeric feature correlation heatmap"))
                pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
                pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
                pairs["Absolute correlation"] = pairs["Correlation"].abs()
                st.subheader("Most correlated feature pairs")
                st.dataframe(pairs.sort_values("Absolute correlation", ascending=False).head(20),
                             use_container_width=True, hide_index=True)
            else:
                st.info("At least two numeric features are required for correlation analysis.")

        with tab5:
            quality = pd.DataFrame({
                "Feature": df.columns,
                "Data type": [str(df[c].dtype) for c in df.columns],
                "Non-missing": [int(df[c].notna().sum()) for c in df.columns],
                "Missing": [int(df[c].isna().sum()) for c in df.columns],
                "Missing %": (df.isna().mean().values * 100).round(2),
                "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
            })
            st.dataframe(quality, use_container_width=True, hide_index=True)
            st.subheader("Potential numeric outliers (IQR rule)")
            outlier_rows = []
            for col in numeric_cols:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) < 4:
                    continue
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                outlier_rows.append({
                    "Feature": col, "Outlier rows (IQR)": count,
                    "Outlier %": round(100 * count / len(s), 2)
                })
            if outlier_rows:
                st.dataframe(pd.DataFrame(outlier_rows).sort_values(
                    "Outlier rows (IQR)", ascending=False
                ), use_container_width=True, hide_index=True)

    else:
        is_numeric = pd.api.types.is_numeric_dtype(df[feature])
        chart_options = (
            ["Histogram", "Distribution", "Box plot", "Violin plot",
             "Density plot", "Trend by row", "Missing values"]
            if is_numeric else
            ["Category frequency", "Pie / donut", "Missing values"]
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            chart_type = st.selectbox("Plot type", chart_options)
        with c2:
            top_n = st.slider("Top categories", 5, 50, 20) if not is_numeric else 20
        plotly_chart(feature_plot(df, feature, chart_type, top_n=top_n))
        st.subheader("Feature summary")
        if is_numeric:
            st.dataframe(pd.to_numeric(df[feature], errors="coerce").describe().to_frame("Value"),
                         use_container_width=True)
        else:
            summary = df[feature].fillna("Missing").astype(str).value_counts().head(30)
            st.dataframe(summary.rename_axis("Value").reset_index(name="Count"),
                         use_container_width=True, hide_index=True)

        st.subheader("Compare features")
        available_numeric = [c for c in numeric_cols if c != feature]
        compare_type = st.selectbox(
            "Comparison type",
            ["None", "Two numeric features", "Numeric feature grouped by category"]
        )
        if compare_type == "Two numeric features":
            x_col = st.selectbox("X feature", numeric_cols)
            y_candidates = [c for c in numeric_cols if c != x_col]
            if y_candidates:
                y_col = st.selectbox("Y feature", y_candidates)
                temp = sample_frame(df[[x_col, y_col]].dropna(), MAX_PLOT_ROWS)
                plotly_chart(px.scatter(temp, x=x_col, y=y_col,
                                         title=f"{y_col} vs {x_col}",
                                         trendline=None))
        elif compare_type == "Numeric feature grouped by category":
            if numeric_cols and categorical_cols:
                num_col = st.selectbox("Numeric feature", numeric_cols)
                cat_col = st.selectbox("Group by category", categorical_cols)
                top_groups = df[cat_col].fillna("Missing").astype(str).value_counts().head(12).index
                temp = df[[num_col, cat_col]].copy()
                temp[cat_col] = temp[cat_col].fillna("Missing").astype(str)
                temp = temp[temp[cat_col].isin(top_groups)].dropna(subset=[num_col])
                temp = sample_frame(temp, MAX_PLOT_ROWS)
                plotly_chart(px.box(temp, x=cat_col, y=num_col, points="outliers",
                                    title=f"{num_col} by {cat_col}"))
            else:
                st.info("This comparison needs at least one numeric and one categorical feature.")


# ---------------------------------------------------------------------
# Predict Status
# ---------------------------------------------------------------------
elif page == "Predict Status":
    st.markdown("## Predict station status")
    st.write("Choose a prediction type, enter station details, and run an algorithm from the relevant ML family.")

    prediction_type = st.radio(
        "Prediction type", ["Classification", "Regression", "Clustering"],
        horizontal=True, key="prediction_type"
    )

    if prediction_type == "Classification":
        targets = classification_targets
        if not targets:
            st.warning("No suitable classification target columns were found.")
        else:
            target = st.selectbox("Target to predict", targets, key="pred_clf_target")
            candidates = [c for c in df.columns if c != target and c not in ["Profit_Potential", "Competition_Level"]]
            default = [c for c in ["City", "State", "EV Level1 EVSE Num", "EV Level2 EVSE Num",
                                   "EV DC Fast Count", "EV Network", "Latitude", "Longitude",
                                   "Facility Type", "Access Code"] if c in candidates]
            features = st.multiselect("Input features", candidates, default=default[:8], key="pred_clf_features")
            algo = st.selectbox("Classification algorithm", [
                "Logistic Regression", "K-Nearest Neighbors", "Gaussian Naive Bayes",
                "Decision Tree Classifier", "Random Forest Classifier", "AdaBoost Classifier",
                "Gradient Boosting Classifier", "Bagging Classifier", "MLP Classifier",
                "Support Vector Classifier (SVC)"
            ], key="pred_clf_algo")
            max_rows = st.select_slider("Maximum training rows", [5000, 10000, 20000, 50000], value=20000, key="pred_clf_rows")
            if features:
                inputs = {}
                cols = st.columns(3)
                for i, col in enumerate(features):
                    with cols[i % 3]:
                        if col in numeric_cols:
                            vals = pd.to_numeric(df[col], errors="coerce").dropna()
                            inputs[col] = st.number_input(col, value=float(vals.median()) if len(vals) else 0.0, key=f"pred_clf_num_{col}")
                        else:
                            vals = df[col].dropna().astype(str).value_counts().index.tolist()
                            inputs[col] = st.selectbox(col, vals[:100] or ["Unknown"], key=f"pred_clf_cat_{col}")
            if st.button("Train and predict", type="primary", key="predict_clf"):
                if not features:
                    st.error("Select at least one input feature.")
                else:
                    work = df[features + [target]].dropna(subset=[target]).copy()
                    X, y = work[features], work[target].astype(str)
                    X, y, sampled = limit_training_data(X, y, max_rows, classification=True)
                    counts = y.value_counts()
                    keep = y.isin(counts[counts >= 2].index)
                    X, y = X.loc[keep], y.loc[keep]
                    if y.nunique() < 2:
                        st.error("At least two classes with two or more records are required.")
                    else:
                        strat = y if y.value_counts().min() >= 2 else None
                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat)
                        models = {
                            "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
                            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
                            "Gaussian Naive Bayes": GaussianNB(),
                            "Decision Tree Classifier": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE, class_weight="balanced"),
                            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=18, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
                            "AdaBoost Classifier": AdaBoostClassifier(n_estimators=60, random_state=RANDOM_STATE),
                            "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=80, random_state=RANDOM_STATE),
                            "Bagging Classifier": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=10), n_estimators=30, random_state=RANDOM_STATE, n_jobs=-1),
                            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, early_stopping=True, random_state=RANDOM_STATE),
                            "Support Vector Classifier (SVC)": SVC(C=1.0, kernel="rbf")
                        }
                        scale = algo in ["Logistic Regression", "K-Nearest Neighbors", "Gaussian Naive Bayes", "MLP Classifier", "Support Vector Classifier (SVC)"]
                        pipe = Pipeline([("prep", feature_pipeline(Xtr, scale=scale)), ("model", models[algo])])
                        try:
                            with st.spinner(f"Training {algo}..."):
                                pipe.fit(Xtr, ytr)
                            test_pred = pipe.predict(Xte)
                            one_pred = pipe.predict(pd.DataFrame([inputs], columns=features))[0]
                            st.success(f"Predicted {target}: **{one_pred}**")
                            st.caption(f"Algorithm used: {algo}" + (f" · trained on a reproducible sample of {len(X):,} rows" if sampled else ""))
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Accuracy", f"{accuracy_score(yte, test_pred):.3f}")
                            c2.metric("Weighted precision", f"{precision_score(yte, test_pred, average='weighted', zero_division=0):.3f}")
                            c3.metric("Weighted recall", f"{recall_score(yte, test_pred, average='weighted', zero_division=0):.3f}")
                            c4.metric("Weighted F1", f"{f1_score(yte, test_pred, average='weighted', zero_division=0):.3f}")
                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")

    elif prediction_type == "Regression":
        targets = regression_targets
        if not targets:
            st.warning("No suitable numeric regression target columns were found.")
        else:
            target = st.selectbox("Numeric target", targets, key="pred_reg_target")
            candidates = [c for c in df.columns if c != target and c not in ["Profit_Potential", "Competition_Level", "Status Code"]]
            default = [c for c in ["EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count", "Latitude", "Longitude", "EV Workplace Charging", "Open Year", "Open Month"] if c in candidates]
            features = st.multiselect("Input features", candidates, default=default[:8], key="pred_reg_features")
            algo = st.selectbox("Regression algorithm", [
                "Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
                "Polynomial Regression (degree 2)", "Decision Tree Regressor", "Random Forest Regressor",
                "Gradient Boosting Regressor", "Support Vector Regressor (SVR)", "K-Nearest Neighbors Regressor"
            ], key="pred_reg_algo")
            max_rows = st.select_slider("Maximum training rows", [5000, 10000, 20000, 50000], value=20000, key="pred_reg_rows")
            if features:
                inputs = {}
                cols = st.columns(3)
                for i, col in enumerate(features):
                    with cols[i % 3]:
                        if col in numeric_cols:
                            vals = pd.to_numeric(df[col], errors="coerce").dropna()
                            inputs[col] = st.number_input(col, value=float(vals.median()) if len(vals) else 0.0, key=f"pred_reg_num_{col}")
                        else:
                            vals = df[col].dropna().astype(str).value_counts().index.tolist()
                            inputs[col] = st.selectbox(col, vals[:100] or ["Unknown"], key=f"pred_reg_cat_{col}")
            if st.button("Train and predict", type="primary", key="predict_reg"):
                if not features:
                    st.error("Select at least one input feature.")
                else:
                    work = df[features].join(pd.to_numeric(df[target], errors="coerce").rename(target)).dropna(subset=[target])
                    X, y = work[features], work[target]
                    X, y, sampled = limit_training_data(X, y, max_rows, classification=False)
                    if y.nunique() < 2:
                        st.error("The selected target has insufficient variation.")
                    else:
                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
                        models = {
                            "Linear Regression": LinearRegression(),
                            "Ridge Regression": Ridge(alpha=1.0),
                            "Lasso Regression": Lasso(alpha=0.001, max_iter=2000),
                            "ElasticNet Regression": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000),
                            "Polynomial Regression (degree 2)": Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("reg", Ridge(alpha=1.0))]),
                            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
                            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=18, random_state=RANDOM_STATE, n_jobs=-1),
                            "Gradient Boosting Regressor": HistGradientBoostingRegressor(max_iter=100, random_state=RANDOM_STATE),
                            "Support Vector Regressor (SVR)": SVR(C=10.0, kernel="rbf"),
                            "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=7, n_jobs=-1)
                        }
                        scale = algo in ["Ridge Regression", "Lasso Regression", "ElasticNet Regression", "Support Vector Regressor (SVR)", "K-Nearest Neighbors Regressor"]
                        pipe = Pipeline([("prep", feature_pipeline(Xtr, scale=scale)), ("model", models[algo])])
                        try:
                            with st.spinner(f"Training {algo}..."):
                                pipe.fit(Xtr, ytr)
                            test_pred = pipe.predict(Xte)
                            one_pred = float(pipe.predict(pd.DataFrame([inputs], columns=features))[0])
                            st.success(f"Predicted {target}: **{one_pred:.4f}**")
                            st.caption(f"Algorithm used: {algo}" + (f" · trained on a reproducible sample of {len(X):,} rows" if sampled else ""))
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Test R²", f"{r2_score(yte, test_pred):.3f}")
                            c2.metric("Test RMSE", f"{np.sqrt(mean_squared_error(yte, test_pred)):.3f}")
                            c3.metric("Test MAE", f"{mean_absolute_error(yte, test_pred):.3f}")
                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")

    else:
        st.write("Clustering assigns a cluster label rather than predicting a known target.")
        features = st.multiselect("Numeric features for clustering", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], key="pred_cluster_features")
        algo = st.selectbox("Clustering algorithm", ["K-Means", "Agglomerative Clustering"], key="pred_cluster_algo")
        k = st.slider("Number of clusters", 2, 10, 3, key="pred_cluster_k")
        max_rows = st.slider("Maximum clustering rows", 500, 10000, 3000, step=500, key="pred_cluster_rows")
        if features:
            inputs = {}
            cols = st.columns(3)
            for i, col in enumerate(features):
                with cols[i % 3]:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    inputs[col] = st.number_input(col, value=float(vals.median()) if len(vals) else 0.0, key=f"pred_cluster_num_{col}")
        if st.button("Fit clustering and assign input", type="primary", key="predict_cluster"):
            if len(features) < 2:
                st.error("Choose at least two numeric features.")
            else:
                work = df[features].replace([np.inf, -np.inf], np.nan).dropna(how="all")
                work = work.fillna(work.median(numeric_only=True)).dropna()
                if len(work) > max_rows:
                    work = work.sample(max_rows, random_state=RANDOM_STATE)
                prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
                try:
                    Xscaled = prep.fit_transform(work)
                    if algo == "K-Means":
                        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                    else:
                        model = AgglomerativeClustering(n_clusters=k)
                    labels = model.fit_predict(Xscaled)
                    one = prep.transform(pd.DataFrame([inputs], columns=features))
                    if algo == "K-Means":
                        assigned = int(model.predict(one)[0])
                    else:
                        # Agglomerative has no predict(); assign to the nearest fitted cluster centroid.
                        centers = np.vstack([Xscaled[labels == label].mean(axis=0) for label in sorted(np.unique(labels))])
                        assigned = int(np.argmin(((centers - one[0]) ** 2).sum(axis=1)))
                    st.success(f"Input assigned to cluster: **{assigned}**")
                    st.caption(f"Algorithm used: {algo}. Agglomerative assignment uses the nearest cluster centroid because sklearn's AgglomerativeClustering has no predict() method." if algo == "Agglomerative Clustering" else f"Algorithm used: {algo}.")
                    if len(np.unique(labels)) > 1:
                        st.metric("Silhouette score", f"{silhouette_score(Xscaled, labels, sample_size=min(2000, len(labels)), random_state=RANDOM_STATE):.3f}")
                except Exception as exc:
                    st.error(f"Clustering failed: {exc}")


# ---------------------------------------------------------------------
# Classification comparison
# ---------------------------------------------------------------------
elif page == "Classification":
    st.markdown("## Classification model comparison")
    if not classification_targets:
        st.warning("No suitable classification target columns were found.")
    else:
        target = st.selectbox("Classification target", classification_targets)
        available = choose_features([], target, exclude_labels=True)
        default_features = [
            c for c in available if c in [
                "City", "State", "EV Level1 EVSE Num", "EV Level2 EVSE Num",
                "EV DC Fast Count", "EV Network", "Latitude", "Longitude",
                "Facility Type", "Access Code", "Nearest_Station_Distance_km"
            ]
        ]
        selected = st.multiselect("Input features", available, default=default_features[:10])
        max_rows = st.select_slider(
            "Maximum rows for comparison (higher may take longer)",
            options=[5000, 10000, 20000, 50000], value=20000, key="clf_max_rows"
        )
        algorithms = st.multiselect(
            "Algorithms to compare",
            ["Logistic Regression", "KNN", "Gaussian Naive Bayes", "Decision Tree",
             "Random Forest", "AdaBoost", "Gradient Boosting", "Bagging",
             "Extra Trees", "Hist Gradient Boosting", "MLP Classifier"],
            default=["Logistic Regression", "KNN", "Gaussian Naive Bayes", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", "Bagging", "Extra Trees", "MLP Classifier"]
        )

        if st.button("Run classification comparison", type="primary", key="run_clf") or st.session_state.get("run_clf_done", False):
            st.session_state["run_clf_done"] = True
            if not selected:
                st.error("Choose at least one feature.")
            elif not algorithms:
                st.error("Choose at least one algorithm.")
            else:
                work = df[selected + [target]].dropna(subset=[target]).copy()
                X, y = work[selected], work[target].astype(str)
                X, y, was_sampled = limit_training_data(X, y, max_rows, classification=True)
                if add_data_quality_warning(y):
                    class_counts = y.value_counts()
                    rare = class_counts[class_counts < 2].index
                    if len(rare):
                        keep = ~y.isin(rare)
                        X, y = X.loc[keep], y.loc[keep]
                        st.warning(f"Excluded {len(rare)} class(es) with fewer than two rows from this comparison.")
                    if y.nunique() < 2:
                        st.error("Not enough classes remain after filtering rare classes.")
                    else:
                        stratify = y if y.value_counts().min() >= 2 else None
                        Xtr, Xte, ytr, yte = train_test_split(
                            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
                        )
                        model_map = {
                            "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
                            "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
                            "Gaussian Naive Bayes": GaussianNB(),
                            "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE, class_weight="balanced"),
                            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
                            "AdaBoost": AdaBoostClassifier(n_estimators=60, random_state=RANDOM_STATE),
                            "Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=RANDOM_STATE),
                            "Bagging": BaggingClassifier(n_estimators=30, max_samples=0.8, random_state=RANDOM_STATE, n_jobs=-1),
                             "Support Vector Classifier (SVC)": SVC(C=1.0, kernel="rbf"),
                            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, early_stopping=True, random_state=RANDOM_STATE),
                        }
                        rows = []
                        progress = st.progress(0)
                        status = st.empty()
                        for i, name in enumerate(algorithms):
                            start = time.perf_counter()
                            try:
                                scale = name in ["Logistic Regression", "KNN", "Gaussian Naive Bayes", "MLP Classifier", "Support Vector Classifier (SVC)"]
                                pipe = Pipeline([
                                    ("prep", feature_pipeline(Xtr, scale=scale)),
                                    ("model", clone(model_map[name]))
                                ])
                                pipe.fit(Xtr, ytr)
                                train_seconds = time.perf_counter() - start
                                pstart = time.perf_counter()
                                pred = pipe.predict(Xte)
                                predict_seconds = time.perf_counter() - pstart
                                rows.append({
                                    "Algorithm": name,
                                    "Accuracy": accuracy_score(yte, pred),
                                    "Weighted precision": precision_score(yte, pred, average="weighted", zero_division=0),
                                    "Macro precision": precision_score(yte, pred, average="macro", zero_division=0),
                                    "Weighted recall": recall_score(yte, pred, average="weighted", zero_division=0),
                                    "Macro recall": recall_score(yte, pred, average="macro", zero_division=0),
                                    "Weighted F1": f1_score(yte, pred, average="weighted", zero_division=0),
                                    "Macro F1": f1_score(yte, pred, average="macro", zero_division=0),
                                    "Training time (s)": train_seconds,
                                    "Prediction time (s)": predict_seconds,
                                })
                            except Exception as exc:
                                rows.append({"Algorithm": name, "Error": str(exc)[:250]})
                            progress.progress((i + 1) / len(algorithms))
                            status.write(f"Finished: {name}")
                        result = pd.DataFrame(rows)
                        st.subheader("Algorithm comparison table")
                        st.dataframe(result, use_container_width=True, hide_index=True)
                        if was_sampled:
                            st.info(f"To keep the app responsive, models used a reproducible sample of {len(X):,} rows from {len(work):,} eligible rows.")
                        metric_options = [
                            "Accuracy", "Weighted precision", "Macro precision",
                            "Weighted recall", "Macro recall", "Weighted F1", "Macro F1",
                            "Training time (s)", "Prediction time (s)"
                        ]
                        available_metrics = [m for m in metric_options if m in result.columns and result[m].notna().any()]
                        if available_metrics:
                            metric = st.selectbox("Metric to plot", available_metrics, key="clf_metric_plot")
                            higher = metric not in ["Training time (s)", "Prediction time (s)"]
                            plot_metric_comparison(result, metric, higher_is_better=higher)
                            st.caption("Each chart is generated from the actual held-out test results. Higher is better for predictive metrics; lower is better for time.")
                        best_metric = "Weighted F1"
                        if best_metric in result.columns and result[best_metric].notna().any():
                            best = result.dropna(subset=[best_metric]).sort_values(best_metric, ascending=False).iloc[0]
                            st.success(f"Best evaluated model by weighted F1: {best['Algorithm']} ({best[best_metric]:.3f}).")
                        with st.expander("Inspect a classification report and confusion matrix"):
                            valid_algorithms = result.loc[result.get("Error", pd.Series(index=result.index, dtype=object)).isna(), "Algorithm"].tolist() if "Error" in result.columns else result["Algorithm"].tolist()
                            inspect_name = st.selectbox("Choose algorithm to inspect", valid_algorithms, key="clf_inspect")
                            if st.button("Re-fit selected model for detailed report", key="clf_detail"):
                                try:
                                    scale = inspect_name in ["Logistic Regression", "KNN", "Gaussian Naive Bayes", "MLP Classifier"]
                                    pipe = Pipeline([("prep", feature_pipeline(Xtr, scale=scale)), ("model", clone(model_map[inspect_name]))])
                                    pipe.fit(Xtr, ytr)
                                    pred = pipe.predict(Xte)
                                    labels = sorted(pd.unique(pd.concat([yte, pd.Series(pred)])))
                                    cm = confusion_matrix(yte, pred, labels=labels)
                                    plotly_chart(px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto",
                                                           title=f"Confusion matrix — {inspect_name}",
                                                           labels=dict(x="Predicted", y="Actual", color="Count")))
                                    st.dataframe(pd.DataFrame(classification_report(yte, pred, output_dict=True, zero_division=0)).T,
                                                 use_container_width=True)
                                except Exception as exc:
                                    st.error(f"Could not generate detailed report: {exc}")


# ---------------------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------------------
elif page == "Regression":
    st.markdown("## Regression model comparison")
    if not regression_targets:
        st.warning("No suitable numeric regression target columns were found.")
    else:
        target = st.selectbox("Numeric target", regression_targets)
        available = [c for c in df.columns if c != target and c not in ["Profit_Potential", "Competition_Level", "Status Code"]]
        default_features = [
            c for c in available if c in [
                "EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count",
                "Latitude", "Longitude", "EV Workplace Charging", "Open Year", "Open Month"
            ]
        ]
        selected = st.multiselect("Input features", available, default=default_features[:10])
        max_rows = st.select_slider(
            "Maximum rows for comparison (higher may take longer)",
            options=[5000, 10000, 20000, 50000], value=20000, key="reg_max_rows"
        )
        algorithms = st.multiselect(
            "Algorithms to compare",
            ["Linear Regression", "Ridge Regression", "Lasso Regression",
             "ElasticNet Regression", "Polynomial Regression (degree 2)",
             "Decision Tree", "Random Forest", "Gradient Boosting",
             "Support Vector Regressor (SVR)", "KNN Regressor"],
            default=["Linear Regression", "Ridge Regression", "Lasso Regression",
                     "ElasticNet Regression", "Polynomial Regression (degree 2)",
                     "Decision Tree", "Random Forest", "Gradient Boosting",
                     "Support Vector Regressor (SVR)", "KNN Regressor"]
        )

        if st.button("Run regression comparison", type="primary", key="run_reg") or st.session_state.get("run_reg_done", False):
            st.session_state["run_reg_done"] = True
            if not selected:
                st.error("Choose at least one input feature.")
            elif not algorithms:
                st.error("Choose at least one algorithm.")
            else:
                work = df[selected + [target]].copy()
                work[target] = pd.to_numeric(work[target], errors="coerce")
                work = work.dropna(subset=[target])
                X, y = work[selected], work[target]
                X, y, was_sampled = limit_training_data(X, y, max_rows, classification=False)
                if y.nunique() < 2:
                    st.error("The selected target has insufficient variation for regression.")
                else:
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, y, test_size=0.2, random_state=RANDOM_STATE
                    )
                    model_map = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(alpha=1.0),
                        "Lasso Regression": Lasso(alpha=0.001, max_iter=2000),
                        "ElasticNet Regression": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000),
                        "Polynomial Regression (degree 2)": Pipeline([
                            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                            ("reg", Ridge(alpha=1.0))
                        ]),
                        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE),
                        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
                        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
                        "KNN Regressor": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
                        "Support Vector Regressor (SVR)": SVR(C=10.0, kernel="rbf"),
                    }
                    rows = []
                    predictions = {}
                    progress = st.progress(0)
                    status = st.empty()
                    for i, name in enumerate(algorithms):
                        start = time.perf_counter()
                        try:
                            scale = name in ["Ridge Regression", "Lasso Regression", "ElasticNet Regression", "KNN Regressor", "Support Vector Regressor (SVR)"]
                            pipe = Pipeline([
                                ("prep", feature_pipeline(Xtr, scale=scale)),
                                ("model", clone(model_map[name]))
                            ])
                            pipe.fit(Xtr, ytr)
                            train_seconds = time.perf_counter() - start
                            pstart = time.perf_counter()
                            pred = pipe.predict(Xte)
                            predict_seconds = time.perf_counter() - pstart
                            predictions[name] = pred
                            rows.append({
                                "Algorithm": name,
                                "R²": r2_score(yte, pred),
                                "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
                                "MAE": mean_absolute_error(yte, pred),
                                "Training time (s)": train_seconds,
                                "Prediction time (s)": predict_seconds,
                            })
                        except Exception as exc:
                            rows.append({"Algorithm": name, "Error": str(exc)[:250]})
                        progress.progress((i + 1) / len(algorithms))
                        status.write(f"Finished: {name}")
                    result = pd.DataFrame(rows)
                    st.subheader("Algorithm comparison table")
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    if was_sampled:
                        st.info(f"To keep the app responsive, models used a reproducible sample of {len(X):,} rows from {len(work):,} eligible rows.")
                    metric_options = ["R²", "RMSE", "MAE", "Training time (s)", "Prediction time (s)"]
                    available_metrics = [m for m in metric_options if m in result.columns and result[m].notna().any()]
                    if available_metrics:
                        metric = st.selectbox("Metric to plot", available_metrics, key="reg_metric_plot")
                        higher = metric == "R²"
                        plot_metric_comparison(result, metric, higher_is_better=higher)

                    if predictions:
                        inspect_model = st.selectbox("Model diagnostics", list(predictions.keys()), key="reg_diag_model")
                        pred = predictions[inspect_model]
                        diag = pd.DataFrame({"Actual": yte.to_numpy(), "Predicted": pred})
                        diag["Residual"] = diag["Actual"] - diag["Predicted"]
                        d1, d2 = st.columns(2)
                        with d1:
                            plotly_chart(px.scatter(diag, x="Actual", y="Predicted",
                                title=f"Actual vs predicted — {inspect_model}",
                                opacity=0.65))
                        with d2:
                            plotly_chart(px.histogram(diag, x="Residual", nbins=35,
                                title=f"Residual distribution — {inspect_model}"))
                        plotly_chart(px.scatter(diag, x="Predicted", y="Residual",
                            title=f"Residuals vs predicted — {inspect_model}", opacity=0.65))
                        st.caption("A useful regression model should generally have residuals centered around zero without a strong visible pattern.")


# ---------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------
elif page == "Clustering":
    st.markdown("## Clustering algorithm comparison")
    st.write("Compare K-Means and Agglomerative Clustering using the same sampled rows and numeric features.")

    cluster_features = st.multiselect(
        "Numeric features for clustering", numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))], key="cluster_features"
    )
    max_rows = st.slider("Rows to cluster (for performance)", 500, 10000, 3000, step=500, key="cluster_rows")
    n_clusters = st.slider("Number of clusters", 2, 10, 3, key="cluster_k")
    algorithms = st.multiselect(
        "Algorithms to compare",
        ["K-Means", "Agglomerative Clustering"],
        default=["K-Means", "Agglomerative Clustering"], key="cluster_algorithms"
    )

    if st.button("Run clustering comparison", type="primary", key="run_cluster"):
        if len(cluster_features) < 2:
            st.error("Select at least two numeric features.")
        elif not algorithms:
            st.error("Select at least one clustering algorithm.")
        else:
            work = df[cluster_features].replace([np.inf, -np.inf], np.nan).copy()
            work = work.dropna(axis=1, how="all")
            work = work.dropna(how="all")
            sampled = len(work) > max_rows
            if sampled:
                work = work.sample(n=max_rows, random_state=RANDOM_STATE)
            if work.shape[1] < 2 or len(work) <= n_clusters:
                st.error("Not enough usable rows/features for the selected cluster count.")
            else:
                prep = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])
                try:
                    X_scaled = prep.fit_transform(work)
                    comparison_rows = []
                    cluster_outputs = {}
                    progress = st.progress(0)
                    status = st.empty()
                    for i, name in enumerate(algorithms):
                        try:
                            if name == "K-Means":
                                model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
                            else:
                                model = AgglomerativeClustering(n_clusters=n_clusters)
                            t0 = time.perf_counter()
                            labels = model.fit_predict(X_scaled)
                            elapsed = time.perf_counter() - t0
                            unique_labels = np.unique(labels)
                            valid_partition = 1 < len(unique_labels) < len(labels)
                            sil = silhouette_score(X_scaled, labels, sample_size=min(2000, len(labels)), random_state=RANDOM_STATE) if valid_partition else np.nan
                            dbi = davies_bouldin_score(X_scaled, labels) if valid_partition else np.nan
                            chi = calinski_harabasz_score(X_scaled, labels) if valid_partition else np.nan
                            comparison_rows.append({
                                "Algorithm": name,
                                "Clusters found": len(unique_labels),
                                "Silhouette Score": sil,
                                "Davies-Bouldin Index": dbi,
                                "Calinski-Harabasz Index": chi,
                                "Fit time (s)": elapsed,
                            })
                            cluster_outputs[name] = labels
                        except Exception as exc:
                            comparison_rows.append({"Algorithm": name, "Error": str(exc)[:250]})
                        progress.progress((i + 1) / len(algorithms))
                        status.write(f"Finished: {name}")

                    result = pd.DataFrame(comparison_rows)
                    st.subheader("Clustering comparison metrics")
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    if sampled:
                        st.info(f"Metrics use a reproducible sample of {len(work):,} rows from the dataset for performance.")
                    st.caption("Silhouette: higher is better. Davies-Bouldin: lower is better. Calinski-Harabasz: higher is generally better.")
                    valid_metrics = [m for m in ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index", "Fit time (s)"] if m in result.columns and result[m].notna().any()]
                    metric = st.selectbox("Clustering metric to plot", valid_metrics, key="cluster_metric")
                    plot_metric_comparison(result, metric, higher_is_better=metric != "Davies-Bouldin Index" and metric != "Fit time (s)")

                    inspect_algo = st.selectbox("Cluster visualization algorithm", list(cluster_outputs), key="cluster_inspect")
                    labels = cluster_outputs[inspect_algo]
                    out = work.copy()
                    out["Cluster"] = labels.astype(str)
                    st.subheader(f"{inspect_algo}: cluster visualizations")
                    counts = out["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Records")
                    plotly_chart(px.bar(counts, x="Cluster", y="Records", text_auto=True, title="Records per cluster"))
                    pca = PCA(n_components=2, random_state=RANDOM_STATE)
                    coords = pca.fit_transform(X_scaled)
                    pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": labels.astype(str)})
                    plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", title="PCA cluster visualization"))
                    st.caption(f"PCA explained variance by first two components: {pca.explained_variance_ratio_.sum():.1%}.")
                    st.dataframe(out.head(1000), use_container_width=True)
                except Exception as exc:
                    st.error(f"Clustering comparison failed: {exc}")


# ---------------------------------------------------------------------
# About
# ---------------------------------------------------------------------
elif page == "About Project":
    st.markdown("""
    <div class="hero">
      <h1>About EV ChargeLens</h1>
      <p>An interactive dashboard for exploring EV charging station data and demonstrating machine-learning workflows.</p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Data source")
    st.write(f"The application reads the fixed local Excel file **{DATA_PATH.name}**. No file-upload feature is provided.")
    st.subheader("Available sections")
    st.markdown("""
    - **Overview:** dataset summary, missing values, feature types, status charts, and correlations.
    - **Feature Explorer:** individual feature plots plus an All Features report.
    - **Predict Status:** user-entered inputs, a selected classifier, prediction, and held-out evaluation.
    - **Classification:** compares classifiers using accuracy, precision, recall, F1, and execution time.
    - **Regression:** compares regression algorithms using R², RMSE, MAE, and execution time.
    - **Clustering:** groups records using K-Means and Agglomerative Clustering.
    """)
    st.subheader("Performance and evaluation notes")
    st.markdown("""
    - The dataset is cached based on the file's modification time so the Excel workbook is not reread on every normal rerun.
    - Model comparison pages use a reproducible sample when the eligible row count exceeds the selected maximum; the app discloses this.
    - Preprocessing is fitted only on the training split.
    - Metrics are calculated on a held-out test split, not the training rows.
    - A score of 1.000 is not automatically an error; investigate target leakage, class balance, duplicate records, and target difficulty before interpreting it.
    - Runtime depends on the number of rows, selected features, categorical cardinality, and algorithm.
    """)
    st.subheader("Runtime environment")
    st.write(f"Rows: {len(df):,} • Columns: {len(df.columns):,} • Numeric: {len(numeric_cols)} • Categorical: {len(categorical_cols)}")
