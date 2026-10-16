"""
RWD-Analyst analysis pipelines beyond target trial emulation, and the router
that selects the pipeline for a query:
  * supervised prediction (regularized logistic / linear regression, AUC, top features)
  * descriptive cohort characterization
  * unsupervised clustering (K-means, number of clusters chosen by silhouette score)
  * patient similarity retrieval (Euclidean distance in a scaled feature space)
"""

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import parse_json_response

logger = logging.getLogger(__name__)

RWD_TASKS = ["supervised_prediction", "descriptive_analysis", "unsupervised_clustering", "patient_retrieval",
             "trial_emulation", "unknown"]


def _feature_transformer(df, feature_cols):
    numeric = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    categorical = df[feature_cols].select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


# ============================================================================ supervised prediction
def run_supervised_prediction_pipeline(query: str, df: pd.DataFrame, feature_cols: list, outcome_col: str, llm):
    """Train a regularized regression model, report AUC (or R^2) and the ten most influential features."""
    try:
        data = df[feature_cols + [outcome_col]].dropna(subset=[outcome_col])
        y = data[outcome_col]
        task_type = "binary_classification" if y.nunique() == 2 else "continuous_regression"
        X_train, X_test, y_train, y_test = train_test_split(
            data[feature_cols], y, test_size=0.2, random_state=42,
            stratify=y if task_type == "binary_classification" else None)
        transformer = _feature_transformer(data, feature_cols)
        estimator = LogisticRegression(max_iter=2000, C=1.0) if task_type == "binary_classification" else LinearRegression()
        model = Pipeline([("prep", transformer), ("model", estimator)]).fit(X_train, y_train)

        if task_type == "binary_classification":
            metrics = {"model_type": "L2-regularized logistic regression",
                       "auc": round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4)}
        else:
            metrics = {"model_type": "Linear regression", "r_squared": round(r2_score(y_test, model.predict(X_test)), 4)}
        names = model.named_steps["prep"].get_feature_names_out()
        coefs = pd.Series(np.ravel(model.named_steps["model"].coef_), index=[n.split("__", 1)[-1] for n in names])
        top_features = coefs[coefs.abs().sort_values(ascending=False).index[:10]].round(4).to_dict()
        metrics.update({"n_train": len(X_train), "n_test": len(X_test), "outcome_rate": round(float(y.mean()), 4)})

        prompt = f"""
    A user wanted to perform the following task: "{query}"

    To help them, a machine learning model was built on real-world data. Here are the results:
    - Model Type: {metrics['model_type']}
    - Performance: {metrics}
    - The top 10 most influential features and their standardized coefficients were: {top_features}

    Please write a brief, easy-to-understand summary for the user.
    - Start by stating what was done (e.g., "A machine learning model was built to...").
    - Explain the performance metric (AUC or R-squared) in simple terms.
    - List a few of the most important features. Make sure to look at the sign of the coefficients when interpreting them.
    - Note that the coefficients describe associations, not causal effects.
    """
        return {"summary": llm.invoke(prompt), "metrics": metrics, "top_features": top_features}
    except Exception as e:
        logger.warning("Supervised prediction failed: %s", e)
        return {"summary": None, "error": str(e)}


# ============================================================================ descriptive analysis
def calculate_descriptive_statistics(df: pd.DataFrame, col_info: dict) -> dict:
    """Summary statistics for every column (numeric: mean/SD/quartiles; categorical: counts)."""
    stats = {}
    for col in df.columns:
        s = df[col]
        col_stats = {"description": col_info.get(col, "No description provided."), "total_count": len(s),
                     "missing_percent": round(float(s.isnull().mean() * 100), 1)}
        numeric = pd.to_numeric(s, errors="coerce")
        if s.notnull().sum() and numeric.notnull().sum() / s.notnull().sum() > 0.9:
            desc = numeric.describe()
            col_stats.update({"type": "Numeric", **{k: (round(float(desc[k]), 2) if pd.notna(desc[k]) else None)
                                                   for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]}})
        else:
            counts = s.value_counts(dropna=False)
            col_stats.update({"type": "Categorical", "value_counts": {
                str(k): f"{v} ({v / len(s) * 100:.1f}%)" for k, v in counts.head(50).items()}})
        stats[col] = col_stats
    return stats


def run_descriptive_analysis(query: str, df: pd.DataFrame, col_info: dict, subject_id_col: Optional[str], llm):
    """Baseline characterization of the cohort, summarized by the LLM with respect to the query."""
    try:
        if subject_id_col and subject_id_col in df.columns:
            df = df.drop_duplicates(subset=[subject_id_col]).drop(columns=[subject_id_col])
        statistics = calculate_descriptive_statistics(df, col_info)
        prompt = f"""
    You are a clinical data analyst. Your task is to provide a descriptive summary of a patient cohort ("cohort characterization").
    You will be given a user's query and a complete set of pre-calculated statistics for all columns.

    Your goal is to synthesize these statistics into a high-level, natural language summary that *specifically addresses the user's query*. Do not just list all the statistics.
    Focus on the most important findings (e.g., mean/median for numeric, top categories for categorical, and any significant missing data). Be concise and clear.

    **User Query:**
    "{query}"

    **Calculated Statistics (with column descriptions):**
    {json.dumps(statistics, indent=2)}

    **Your Summary (addressing the user's query):**
    """
        return {"summary": llm.invoke(prompt), "statistics": statistics}
    except Exception as e:
        logger.warning("Descriptive analysis failed: %s", e)
        return {"summary": None, "error": str(e)}


# ============================================================================ clustering
def perform_clustering(X_processed, max_k=8, seed=42):
    """K-means for k = 2..max_k; the number of clusters maximizes the silhouette score."""
    best = (-1, 2, None)
    for k in range(2, min(max_k, X_processed.shape[0] - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X_processed)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X_processed, labels, sample_size=min(5000, X_processed.shape[0]), random_state=seed)
        if score > best[0]:
            best = (score, k, labels)
    return best[2], best[1], best[0]


def run_clustering_pipeline(query: str, df: pd.DataFrame, feature_cols: list, subject_id_col: str, col_info: dict, llm):
    """Discover patient phenotypes with K-means and describe each cluster."""
    try:
        X = _feature_transformer(df, feature_cols).fit_transform(df[feature_cols])
        labels, k, silhouette = perform_clustering(X)
        data = df.copy()
        data["cluster"] = labels
        summaries = {}
        for i in range(k):
            sub = data[data["cluster"] == i]
            stats = calculate_descriptive_statistics(sub[feature_cols], col_info)
            summaries[f"Cluster {i}"] = {"n_patients": len(sub), "statistics": {
                c: {key: v for key, v in s.items() if key in ("mean", "50%", "value_counts")} for c, s in stats.items()}}
        prompt = f"""
    A user performed unsupervised clustering (K-means, k={k} selected by silhouette score {silhouette:.3f}) to find patient phenotypes.

    The user's original request was:
    "{query}"

    Descriptive statistics of each cluster:
    {json.dumps(summaries, indent=2, default=str)}

    Your task is to synthesize these into a single, high-level report.
    - Start by stating how many phenotypes (clusters) were found.
    - For each phenotype, provide a concise name (e.g., "Phenotype 1: Older, High Comorbidity") and summarize its key characteristics.
    - Compare and contrast the phenotypes.
    - Conclude by directly answering the user's original query.
    """
        return {"summary": llm.invoke(prompt), "n_clusters": k, "silhouette": silhouette, "cluster_results": summaries}
    except Exception as e:
        logger.warning("Clustering failed: %s", e)
        return {"summary": None, "error": str(e)}


# ============================================================================ patient similarity
def run_patient_similarity_pipeline(query: str, df: pd.DataFrame, col_info: dict, subject_id_col: str, llm, top_k=10):
    """Translate the described patient into a target vector and retrieve the top-k most similar patients."""
    try:
        samples = {}
        for col, desc in col_info.items():
            if col in df.columns and col != subject_id_col:
                if pd.api.types.is_numeric_dtype(df[col]):
                    samples[col] = f"Desc: '{desc}'. Type: Numeric. Range: [{df[col].min():.1f} - {df[col].max():.1f}]"
                else:
                    samples[col] = f"Desc: '{desc}'. Type: Categorical. Sample values: {df[col].value_counts().head(3).index.tolist()}"
        prompt = f"""
    You are an expert data analyst. Your task is to translate a user's natural language query into a structured JSON representation of a "target patient" for a similarity search.

    You must:
    1.  Read the **User Query**.
    2.  Use the **Column Dictionary and Samples** to identify the *exact* column names that match the query.
    3.  Estimate *plausible values* for the target patient based on the query (e.g., "old" might be 75, "high lactate" might be 8.0).
    4.  Return a JSON object with two keys:
        - "feature_cols": A list of the column names used.
        - "target_vector": A dictionary where keys are the column names and values are the estimated target values.

    **Column Dictionary and Samples:**
    {json.dumps(samples, indent=2)}

    **User Query:**
    "{query}"

    **JSON Output (do not add any other text):**
    """
        parsed = parse_json_response(llm.invoke(prompt))
        feature_cols = [c for c in parsed["feature_cols"] if c in df.columns]
        target = pd.DataFrame([{c: parsed["target_vector"].get(c) for c in feature_cols}])
        transformer = _feature_transformer(df, feature_cols).fit(df[feature_cols])
        distances = euclidean_distances(transformer.transform(target), transformer.transform(df[feature_cols]))[0]
        nearest = df.iloc[np.argsort(distances)[:top_k]].copy()
        nearest["distance"] = np.sort(distances)[:top_k]
        prompt = f"""
    A user is looking for patients similar to a described case.

    **User's Original Query:** "{query}"
    **Interpreted Target Patient:** {json.dumps(parsed["target_vector"], indent=2)}
    **The {top_k} Most Similar Patients (Euclidean distance in a standardized feature space):**
    {nearest.to_json(orient="records", indent=2)}

    Write a brief summary: confirm what the user was looking for, describe the retrieved patients (typical values and
    outcomes, if available), and explain how closely they match the target on the key features.
    """
        return {"summary": llm.invoke(prompt), "target_vector": parsed["target_vector"],
                "similar_patients": nearest.to_dict(orient="records")}
    except Exception as e:
        logger.warning("Patient similarity retrieval failed: %s", e)
        return {"summary": None, "error": str(e)}


# ============================================================================ router
def determine_rwd_task(query: str, llm) -> dict:
    """Route a query to one of the RWD analysis pipelines."""
    prompt = f"""
    You are an intelligent router for a clinical data analysis pipeline.
    Your job is to map a user's natural language query to one of six specific analysis tasks.

    **The Six Available Tasks:**

    1. **supervised_prediction**:
       - Triggers: Predicting a specific outcome (risk, mortality, length of stay) from patient characteristics,
         building a model, forecasting, prognostic factors. No specific treatment comparison is asked.
       - Keywords: "Predict", "Risk model", "Prognosis", "Will they...", "calculate risk".

    2. **descriptive_analysis**:
       - Triggers: Summarizing data, calculating statistics, creating baseline tables, checking incidence rates.
       - Keywords: "Summarize", "Describe", "Table 1", "Demographics", "Average", "Distribution", "Counts".

    3. **unsupervised_clustering**:
       - Triggers: Finding subtypes, grouping patients without a specific outcome, phenotyping.
       - Keywords: "Cluster", "Phenotypes", "Subtypes", "Group patients", "Profiles", "Patterns".

    4. **patient_retrieval**:
       - Triggers: Finding a specific patient or similar patients based on a description.
       - Keywords: "Find patient", "Search for", "Like this patient", "Retrieve", "Who is...".

    5. **trial_emulation**:
       - Triggers: Simulating a clinical trial, estimating the effect of a treatment or exposure on an outcome
         (effectiveness, efficacy, comparative effectiveness, harms of an intervention), designing trial protocols.
       - Keywords: "Emulate trial", "Simulate study", "Trial design", "What if", "Estimate effect", "effectiveness",
         "does X reduce/improve/increase Y".

    6. **unknown**:
       - Triggers: Queries that do not fit into the above categories.

    **User Query:**
    "{query}"

    **Instructions:**
    - Analyze the user's intent carefully.
    - Select the *single best* task type from the list above.
    - Return ONLY a valid JSON object. Do not add markdown formatting.

    **Output Format:**
    {{
        "task_type": "selected_task_name",
        "reasoning": "Brief explanation of why this fits."
    }}
    """
    result = parse_json_response(llm.invoke(prompt))
    if not isinstance(result, dict) or result.get("task_type") not in RWD_TASKS:
        return {"task_type": "unknown", "reasoning": "The routing response could not be parsed."}
    return result
