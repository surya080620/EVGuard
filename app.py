import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, r2_score,
    mean_absolute_error, mean_squared_error, silhouette_score
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    BaggingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

st.set_page_config(page_title="EV ChargeLens | ML Analytics", page_icon="⚡", layout="wide")

# Use the project's fixed dataset. No upload widget is provided.
DATA_PATH = "EV_Charging_Station.xlsx"

st.markdown("""
<style>
.stApp {background: #081225; color: #f3f7ff;}
[data-testid="stSidebar"] {background: #101b30;}
.block-container {padding-top: 1.6rem;}
.hero {padding: 28px 32px; border: 1px solid #24486e; border-radius: 20px;
background: linear-gradient(120deg,#102a4c,#123a59); margin-bottom: 18px;}
.hero h1 {color:#fff; margin-bottom:8px;}
.hero p {color:#c4d8ef; margin:0;}
.metric-card {background:#111f34; border:1px solid #293b55; padding:16px 18px; border-radius:14px;}
.small-note {color:#a7bad3; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    if "Open Date" in df.columns:
        # The supplied file stores dates as Excel serial values in some rows.
        raw = df["Open Date"]
        numeric = pd.to_numeric(raw, errors="coerce")
        converted = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
        parsed = pd.to_datetime(raw, errors="coerce")
        df["Open Date"] = parsed.fillna(converted)
        df["Open Year"] = df["Open Date"].dt.year
        df["Open Month"] = df["Open Date"].dt.month
        df = df.drop(columns=["Open Date"])
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load {DATA_PATH}. Place EV_Charging_Station.xlsx in the same folder as app.py. Details: {e}")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
target_candidates = [c for c in ["Status Code", "Profit_Potential", "Competition_Level", "Opportunity_Score"] if c in df.columns]

with st.sidebar:
    st.markdown("## ⚡ EV ChargeLens")
    st.caption("EV Charging Station ML Analytics")
    st.markdown("---")
    page = st.radio("NAVIGATION", [
        "Overview", "Feature Explorer", "Predict Status",
        "Classification", "Regression", "Clustering", "About Project"
    ])
    st.markdown("---")
    st.success(f"Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")
    st.caption("Fixed local dataset • no file upload")

def plotly_chart(fig, key=None):
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)

def prep_pipeline(X, scale=False):
    nums = X.select_dtypes(include=np.number).columns.tolist()
    cats = X.select_dtypes(exclude=np.number).columns.tolist()
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), nums),
        ("cat", Pipeline(cat_steps), cats)
    ], remainder="drop")

def get_feature_data():
    return df.copy()

def feature_plot(data, feature, chart_type):
    series = data[feature]
    if pd.api.types.is_numeric_dtype(series):
        temp = data[[feature]].copy()
        if chart_type == "Distribution":
            fig = px.histogram(temp, x=feature, nbins=40, marginal="box",
                               title=f"Distribution of {feature}")
        elif chart_type == "Box plot":
            fig = px.box(temp, y=feature, points="outliers", title=f"Box plot: {feature}")
        elif chart_type == "Trend by row":
            temp = temp.reset_index(names="Record")
            fig = px.line(temp, x="Record", y=feature, title=f"{feature} by record order")
        else:
            fig = px.histogram(temp, x=feature, title=f"Frequency of {feature}")
    else:
        counts = series.fillna("Missing").astype(str).value_counts().head(30).rename_axis(feature).reset_index(name="Count")
        fig = px.bar(counts, x=feature, y="Count", title=f"Top categories: {feature}")
        fig.update_layout(xaxis_tickangle=-35)
    return fig

if page == "Overview":
    st.markdown("""<div class="hero"><p style="letter-spacing:2px;font-weight:700;color:#60e0ee;">EV CHARGING • MACHINE LEARNING</p>
    <h1>Explore the EV Charging Station Dataset</h1>
    <p>Analyze station availability, charging infrastructure, competition, and investment opportunity using interactive Plotly visualizations and machine-learning models.</p></div>""", unsafe_allow_html=True)
    missing = int(df.isna().sum().sum())
    countries_or_states = df["State"].nunique() if "State" in df else 0
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Records", f"{len(df):,}")
    m2.metric("Features", f"{len(df.columns):,}")
    m3.metric("Missing cells", f"{missing:,}")
    m4.metric("States", f"{countries_or_states:,}")
    st.subheader("Dataset preview")
    st.dataframe(df.head(15), use_container_width=True, height=350)
    c1,c2 = st.columns(2)
    with c1:
        if "Status Code" in df:
            vc = df["Status Code"].fillna("Missing").astype(str).value_counts().rename_axis("Status").reset_index(name="Stations")
            plotly_chart(px.bar(vc, x="Status", y="Stations", title="Station status distribution", text_auto=True))
    with c2:
        if "Facility Type" in df:
            vc = df["Facility Type"].fillna("Missing").astype(str).value_counts().head(12).rename_axis("Facility Type").reset_index(name="Stations")
            plotly_chart(px.bar(vc, x="Facility Type", y="Stations", title="Top facility types", text_auto=True))
    st.subheader("Data quality")
    miss = df.isna().sum().sort_values(ascending=False).head(15).reset_index()
    miss.columns = ["Feature", "Missing values"]
    plotly_chart(px.bar(miss, x="Feature", y="Missing values", title="Features with most missing values"))
    st.caption("All charts are built with Plotly. The application reads the Excel file directly; there is no upload control.")

elif page == "Feature Explorer":
    st.markdown("## Feature Explorer")
    st.write("Choose one feature from the dropdown to view its individual Plotly visualization.")
    feature = st.selectbox("Select feature", df.columns.tolist())
    chart_options = ["Distribution", "Box plot", "Trend by row"] if pd.api.types.is_numeric_dtype(df[feature]) else ["Category frequency"]
    chart_type = st.selectbox("Plot type", chart_options)
    plotly_chart(feature_plot(df, feature, chart_type))
    st.subheader("Feature summary")
    if pd.api.types.is_numeric_dtype(df[feature]):
        st.dataframe(df[feature].describe().to_frame("Value"), use_container_width=True)
    else:
        summary = df[feature].fillna("Missing").astype(str).value_counts().head(20).rename_axis("Value").reset_index(name="Count")
        st.dataframe(summary, use_container_width=True)

elif page == "Predict Status":
    st.markdown("## Predict station status")
    st.write("Enter station details. The app trains a classification model on the existing dataset and predicts the status category.")
    target = st.selectbox("Target to predict", [c for c in ["Status Code", "Profit_Potential", "Competition_Level"] if c in df.columns], index=0)
    excluded = [target]
    # Avoid direct leakage from related outcome/derived label fields.
    leak_cols = {"Opportunity_Score", "Profit_Potential", "Competition_Level", "Status Code"} - {target}
    features = [c for c in df.columns if c not in excluded and c not in leak_cols]
    selected_features = st.multiselect("Features used by the model", features, default=[c for c in features if c in ["City","State","EV Level1 EVSE Num","EV Level2 EVSE Num","EV DC Fast Count","EV Network","Latitude","Longitude","Facility Type","Access Code","Nearest_Station_Distance_km","EV Workplace Charging"]])
    if not selected_features:
        st.warning("Select at least one input feature.")
    else:
        inputs = {}
        cols = st.columns(3)
        for i, col in enumerate(selected_features):
            with cols[i % 3]:
                if col in numeric_cols:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    default = float(vals.median()) if len(vals) else 0.0
                    inputs[col] = st.number_input(col, value=default, format="%.4f")
                else:
                    vals = df[col].dropna().astype(str).value_counts().index.tolist()
                    options = vals[:100] if vals else ["Unknown"]
                    inputs[col] = st.selectbox(col, options, key=f"predict_{col}")
        model_name = st.selectbox("Algorithm", ["Random Forest Classifier", "Decision Tree Classifier", "Logistic Regression", "K-Nearest Neighbors", "Gradient Boosting Classifier"])
        if st.button("Train model and predict", type="primary"):
            work = df[selected_features + [target]].dropna(subset=[target]).copy()
            X, y = work[selected_features], work[target].astype(str)
            if y.nunique() < 2:
                st.error("The selected target has fewer than two classes.")
            else:
                stratify = y if y.value_counts().min() >= 2 else None
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
                model_map = {
                    "Random Forest Classifier": RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1),
                    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=12, random_state=42, class_weight="balanced"),
                    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
                    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
                    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
                }
                pipe = Pipeline([("prep", prep_pipeline(X_train, scale=model_name in ["Logistic Regression","K-Nearest Neighbors"])), ("model", model_map[model_name])])
                with st.spinner("Training model..."):
                    pipe.fit(X_train, y_train)
                    pred_test = pipe.predict(X_test)
                    pred_one = pipe.predict(pd.DataFrame([inputs]))
                st.success(f"Predicted {target}: **{pred_one[0]}**")
                st.caption(f"Algorithm used: {model_name}")
                a,b,c,d = st.columns(4)
                a.metric("Accuracy", f"{accuracy_score(y_test, pred_test):.3f}")
                a2 = precision_score(y_test, pred_test, average="weighted", zero_division=0)
                b.metric("Weighted precision", f"{a2:.3f}")
                c.metric("Weighted recall", f"{recall_score(y_test, pred_test, average='weighted', zero_division=0):.3f}")
                d.metric("Weighted F1", f"{f1_score(y_test, pred_test, average='weighted', zero_division=0):.3f}")
                labels = sorted(pd.unique(pd.concat([y_test, pd.Series(pred_test)])))
                cm = confusion_matrix(y_test, pred_test, labels=labels)
                plotly_chart(px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto", title="Confusion matrix", labels=dict(x="Predicted", y="Actual", color="Count")))
                st.caption("Metrics are measured on a held-out test split, not on the training rows.")

elif page == "Classification":
    st.markdown("## Classification model comparison")
    target = st.selectbox("Classification target", [c for c in ["Status Code","Profit_Potential","Competition_Level"] if c in df.columns])
    available = [c for c in df.columns if c not in [target, "Opportunity_Score"] and c not in ({"Status Code","Profit_Potential","Competition_Level"}-{target})]
    selected = st.multiselect("Input features", available, default=[c for c in available if c in ["City","State","EV Level1 EVSE Num","EV Level2 EVSE Num","EV DC Fast Count","EV Network","Latitude","Longitude","Facility Type","Access Code","Nearest_Station_Distance_km"]])
    if st.button("Run classification comparison", type="primary", key="run_clf"):
        if not selected:
            st.error("Choose at least one feature.")
        else:
            work = df[selected+[target]].dropna(subset=[target]).copy()
            X, y = work[selected], work[target].astype(str)
            if y.value_counts().min() < 2:
                st.error("At least one class has fewer than two rows; cannot create a stratified split.")
            else:
                Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
                    "KNN": KNeighborsClassifier(n_neighbors=7),
                    "Gaussian Naive Bayes": GaussianNB(),
                    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42, class_weight="balanced"),
                    "SVC": SVC(C=1.0, kernel="linear", class_weight="balanced"),
                    "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced"),
                    "AdaBoost": AdaBoostClassifier(n_estimators=80, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Bagging": BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                    "Extra Trees": ExtraTreesClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced")
                }
                rows=[]
                progress=st.progress(0)
                status=st.empty()
                for i,(name,model) in enumerate(models.items()):
                    try:
                        scale = name in ["Logistic Regression","KNN","SVC","Gaussian Naive Bayes"]
                        pipe=Pipeline([("prep",prep_pipeline(Xtr,scale=scale)),("model",model)])
                        pipe.fit(Xtr,ytr)
                        pred=pipe.predict(Xte)
                        rows.append({"Algorithm":name,"Accuracy":accuracy_score(yte,pred),
                                     "Precision (weighted)":precision_score(yte,pred,average="weighted",zero_division=0),
                                     "Recall (weighted)":recall_score(yte,pred,average="weighted",zero_division=0),
                                     "F1 (weighted)":f1_score(yte,pred,average="weighted",zero_division=0)})
                    except Exception as e:
                        rows.append({"Algorithm":name,"Accuracy":np.nan,"Precision (weighted)":np.nan,"Recall (weighted)":np.nan,"F1 (weighted)":np.nan,"Error":str(e)[:150]})
                    progress.progress((i+1)/len(models)); status.write(f"Evaluated {name}")
                result=pd.DataFrame(rows).sort_values("F1 (weighted)",ascending=False,na_position="last")
                st.subheader("Algorithm comparison table")
                st.dataframe(result, use_container_width=True, hide_index=True)
                valid=result.dropna(subset=["Accuracy"])
                if len(valid):
                    metric=st.selectbox("Metric to plot",["Accuracy","Precision (weighted)","Recall (weighted)","F1 (weighted)"])
                    plotly_chart(px.bar(valid.sort_values(metric,ascending=False),x="Algorithm",y=metric,color=metric,title=f"Classification algorithms compared by {metric}",text_auto=".3f"))
                    st.caption(f"Best by weighted F1: {result.iloc[0]['Algorithm']}. All metrics use the same held-out test set.")

elif page == "Regression":
    st.markdown("## Regression model comparison")
    target = st.selectbox("Numeric target", [c for c in ["Opportunity_Score","Nearest_Station_Distance_km","EV Level2 EVSE Num","EV DC Fast Count"] if c in df.columns])
    available = [c for c in df.columns if c != target and c not in ["Profit_Potential","Competition_Level","Status Code"]]
    selected = st.multiselect("Input features", available, default=[c for c in available if c in ["EV Level1 EVSE Num","EV Level2 EVSE Num","EV DC Fast Count","Latitude","Longitude","Nearest_Station_Distance_km","EV Workplace Charging","Open Year"] and c != target])
    if st.button("Run regression comparison", type="primary", key="run_reg"):
        if not selected:
            st.error("Choose at least one input feature.")
        else:
            work=df[selected+[target]].copy()
            work[target]=pd.to_numeric(work[target],errors="coerce")
            work=work.dropna(subset=[target])
            X,y=work[selected],work[target]
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
            models={
                "Linear Regression":LinearRegression(),
                "Ridge Regression":Ridge(alpha=1.0),
                "Lasso Regression":Lasso(alpha=0.001,max_iter=3000),
                "ElasticNet Regression":ElasticNet(alpha=0.001,l1_ratio=0.5,max_iter=3000),
                "Polynomial Regression (degree 2)":Pipeline([("poly",PolynomialFeatures(degree=2,include_bias=False)),("reg",LinearRegression())]),
                "Decision Tree":DecisionTreeRegressor(max_depth=12,random_state=42),
                "Random Forest":RandomForestRegressor(n_estimators=120,max_depth=18,random_state=42,n_jobs=-1),
                "Gradient Boosting":GradientBoostingRegressor(random_state=42),
                "SVR":SVR(C=10,epsilon=0.1),
                "KNN Regressor":KNeighborsRegressor(n_neighbors=7)
            }
            rows=[]; progress=st.progress(0); status=st.empty()
            for i,(name,model) in enumerate(models.items()):
                try:
                    scale=name in ["Ridge Regression","Lasso Regression","ElasticNet Regression","SVR","KNN Regressor"]
                    pipe=Pipeline([("prep",prep_pipeline(Xtr,scale=scale)),("model",model)])
                    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
                    rows.append({"Algorithm":name,"R²":r2_score(yte,pred),"RMSE":mean_squared_error(yte,pred)**0.5,"MAE":mean_absolute_error(yte,pred)})
                except Exception as e:
                    rows.append({"Algorithm":name,"R²":np.nan,"RMSE":np.nan,"MAE":np.nan,"Error":str(e)[:150]})
                progress.progress((i+1)/len(models)); status.write(f"Evaluated {name}")
            result=pd.DataFrame(rows).sort_values("R²",ascending=False,na_position="last")
            st.subheader("Algorithm comparison table")
            st.dataframe(result,use_container_width=True,hide_index=True)
            valid=result.dropna(subset=["R²"])
            if len(valid):
                metric=st.selectbox("Metric to plot",["R²","RMSE","MAE"])
                ascending=metric!="R²"
                plotly_chart(px.bar(valid.sort_values(metric,ascending=ascending),x="Algorithm",y=metric,color=metric,title=f"Regression algorithms compared by {metric}",text_auto=".3f"))
                st.caption(f"Best by R²: {result.iloc[0]['Algorithm']}. Lower RMSE/MAE is better; higher R² is better.")

elif page == "Clustering":
    st.markdown("## Clustering comparison")
    st.write("Clustering is unsupervised, so there is no target column. The app compares cluster quality with silhouette score when the result has 2–10 clusters.")
    default_feats=[c for c in ["Latitude","Longitude","EV Level1 EVSE Num","EV Level2 EVSE Num","EV DC Fast Count","Nearest_Station_Distance_km","Opportunity_Score"] if c in numeric_cols]
    selected=st.multiselect("Numeric features for clustering",numeric_cols,default=default_feats)
    sample_n=st.slider("Rows to sample for clustering",500,10000,3000,step=500)
    k=st.slider("K-Means / Agglomerative cluster count",2,10,4)
    if st.button("Run clustering comparison",type="primary",key="run_cluster"):
        if len(selected)<2:
            st.error("Select at least two numeric features.")
        else:
            X=df[selected].replace([np.inf,-np.inf],np.nan)
            X=X.fillna(X.median(numeric_only=True)).sample(n=min(sample_n,len(X)),random_state=42)
            Xs=StandardScaler().fit_transform(X)
            cluster_models={
                "K-Means":KMeans(n_clusters=k,n_init=10,random_state=42),
                "Agglomerative":AgglomerativeClustering(n_clusters=k),
                "DBSCAN":DBSCAN(eps=0.8,min_samples=10)
            }
            rows=[]; assignments={}
            for name,model in cluster_models.items():
                try:
                    labels=model.fit_predict(Xs)
                    assignments[name]=labels
                    n_clusters=len(set(labels)-{-1})
                    mask=labels!=-1
                    score=np.nan
                    if n_clusters>=2 and mask.sum()>n_clusters and len(set(labels[mask]))>=2:
                        # Score a bounded subset to keep interactive execution responsive.
                        ids=np.where(mask)[0]
                        if len(ids)>1500: ids=np.random.RandomState(42).choice(ids,1500,replace=False)
                        score=silhouette_score(Xs[ids],labels[ids])
                    rows.append({"Algorithm":name,"Clusters (excluding noise)":n_clusters,"Noise points":int((labels==-1).sum()),"Silhouette score":score})
                except Exception as e:
                    rows.append({"Algorithm":name,"Clusters (excluding noise)":0,"Noise points":0,"Silhouette score":np.nan,"Error":str(e)[:150]})
            result=pd.DataFrame(rows).sort_values("Silhouette score",ascending=False,na_position="last")
            st.subheader("Clustering comparison table")
            st.dataframe(result,use_container_width=True,hide_index=True)
            plotly_chart(px.bar(result,x="Algorithm",y="Silhouette score",title="Clustering algorithm comparison (silhouette score)",text_auto=".3f"))
            chosen=st.selectbox("Plot cluster assignment from",list(assignments.keys()))
            vis_x=st.selectbox("X axis",selected,index=0)
            vis_y=st.selectbox("Y axis",selected,index=min(1,len(selected)-1))
            plot_df=X.copy()
            plot_df["Cluster"]=assignments[chosen].astype(str)
            plotly_chart(px.scatter(plot_df,x=vis_x,y=vis_y,color="Cluster",title=f"{chosen}: clusters by selected features",opacity=0.7))
            st.caption("Silhouette score ranges roughly from -1 to 1; higher is generally better. DBSCAN noise points are labelled -1.")

elif page == "About Project":
    st.markdown("## About this project")
    st.write("EV ChargeLens is a machine-learning analytics dashboard for the supplied EV charging station dataset.")
    st.markdown("""
**Main sections**
- **Overview:** dataset dimensions, preview, missing values, and summary charts.
- **Feature Explorer:** one dropdown-selected feature and interactive Plotly charts.
- **Predict Status:** enter feature values, train a selected classifier, and view the predicted category plus held-out metrics.
- **Classification:** compare ten classifiers using accuracy, weighted precision, weighted recall, and weighted F1.
- **Regression:** compare ten regressors using R², RMSE, and MAE.
- **Clustering:** compare K-Means, Agglomerative Clustering, and DBSCAN using cluster counts and silhouette score.

**Dataset fields used as outcomes**
- `Status Code`, `Profit_Potential`, and `Competition_Level` are classification targets.
- `Opportunity_Score` is the default regression target.
- Clustering is unsupervised and does not require a target.

**Important evaluation note:** results depend on the selected features, target, preprocessing, and train/test split. A high score is not guaranteed; use a held-out test set and avoid including target-derived fields as predictors.
""")
    st.info("All visualizations use Plotly. The app reads EV_Charging_Station.xlsx from the project folder and intentionally has no file-upload feature.")
