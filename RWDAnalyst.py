"""
RWD-Analyst: target trial emulation on real-world data.

Given a clinical question and an analysis-ready cohort (see rwd_preprocessor.py),
the agent drafts a target trial protocol, selects a covariate balancing method
(PSM or IPTW) and survival models, checks covariate balance (SMD) and asks a
clinician agent how to handle unbalanced covariates, estimates the treatment
effect, applies clone-censor-weight analysis when treatment initiation is time
dependent (immortal-time bias), runs prespecified subgroup analyses, computes
E-values and sample size, and writes a trial emulation report.
"""

import json
import logging
import re
from typing import Any, Dict

from emulation_utils import (calculate_evalues, check_balance_after_matching, clean_data,
                             clone_censor_weight, compute_cox_sample_size, is_time_dependent_treatment, perform_iptw,
                             perform_psm, run_ccw_analysis, run_cox_model, run_doubly_robust, run_kaplan_meier,
                             run_parametric_model, run_subgroup_cox, run_survival_forest)
from utils import parse_json_response

logger = logging.getLogger(__name__)

DEFAULT_GRACE_PERIOD_DAYS = 2.0


def generate_target_trial_protocol(query: str, llm: Any) -> Dict[str, Any]:
    """Draft a target trial protocol (arms, outcomes, eligibility) with a justification for each component."""
    prompt = f"""
    You are an expert clinical trialist designing a target trial protocol for an observational study
    that will be emulated using a real-world evidence (RWE) dataset like MIMIC-IV.

    Based on the user's query, define the key components of the protocol.
    For each component, you must provide a "description" and a "justification" explaining
    why that choice was made. Crucially, all eligibility criteria must be operationalizable from typical electronic
    health record (EHR) data (e.g., diagnoses via ICD codes, medications, lab values, demographics, procedures).
    Do not include criteria that cannot be found in a typical EHR, such as patient intent or non-standard survey scores.

    USER QUERY: "{query}"

    Return your protocol as a single, valid JSON object with the following structure:
    - "treatment_arm": An OBJECT with "description" and "justification" keys.
    - "comparator_arm": An OBJECT with "description" and "justification" keys.
    - "outcomes": A LIST OF OBJECTS, where each object has "description" and "justification" keys.
    - "inclusion_criteria": A LIST OF OBJECTS, where each object has "description" and "justification" keys.
    - "exclusion_criteria": A LIST OF OBJECTS, where each object has "description" and "justification" keys.

    Your response MUST be only the valid JSON object.
    """
    protocol = parse_json_response(llm.invoke(prompt))
    if not isinstance(protocol, dict):
        logger.warning("Could not parse the target trial protocol.")
        return {}
    return protocol


def build_dataframe_with_df_meta(df, metadata):
    """Build the RWD-Analyst state from the preprocessed cohort and its metadata."""
    outcome_def = metadata.get("outcome_definition") or {}
    treatment_def = metadata.get("treatment_definition") or {}
    eligibility = metadata.get("eligibility_criteria") or {}

    follow_up = outcome_def.get("follow_up_days")
    outcome_str = f"{follow_up}-day {outcome_def.get('type')}" if follow_up else str(outcome_def.get("type"))
    treatment_str = f"{', '.join(treatment_def.get('treatment', []))} vs {treatment_def.get('control')}"
    inclusion = []
    if eligibility.get("age_min") is not None:
        inclusion.append(f"Age >= {eligibility['age_min']:g} years")
    if eligibility.get("age_max") is not None:
        inclusion.append(f"Age <= {eligibility['age_max']:g} years")
    for codes in eligibility.get("diagnoses_include") or []:
        inclusion.append(f"Diagnosis codes (ICD-10): {', '.join(codes)}")
    if eligibility.get("icu_stay_required"):
        inclusion.append("ICU stay during the index admission")
    exclusion = [f"Diagnosis codes (ICD-10): {', '.join(eligibility['diagnoses_exclude'])}"] \
        if eligibility.get("diagnoses_exclude") else []

    return {"variables": {
        "cleaned_dataframe": df,
        "patient_id": metadata.get("col_id", "subject_id"),
        "Covariates": [c for c in metadata.get("cols_covariates", []) if c in df.columns],
        "Treatment": metadata.get("col_treatment"),
        "Outcome": metadata.get("col_outcome"),
        "Duration": metadata.get("col_time_to_event"),
        "treatment_time_col": metadata.get("col_treatment_time"),
        "grace_period_days": treatment_def.get("treatment_window_days") or DEFAULT_GRACE_PERIOD_DAYS,
        "follow_up_days": follow_up,
        "data_source": metadata.get("data_source", "MIMIC-IV"),
        "time_zero": metadata.get("time_zero", "hospital admission"),
        "covariate_description": metadata.get("covariate_description", "baseline covariates"),
        "treatment_definition": [treatment_str],
        "outcome_definition": [outcome_str],
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
    }}


def build_trial_emulation_protocol(state: dict, llm):
    """Write a structured trial emulation protocol from the target trial specification and the implemented definitions."""
    v = state["variables"]
    prompt = f"""
    You are a statistician tasked with drafting a clear and structured trial emulation protocol.

    **INSTRUCTIONS:**
    Based on the provided KEY TRIAL COMPONENTS, write a formal trial emulation protocol. The protocol should logically connect these components and briefly outline the analysis plan. It must include the following sections:
    1.  **Objective:** A single sentence stating the goal of the study.
    2.  **Study Population:** Detail the inclusion and exclusion criteria.
    3.  **Intervention and Comparator:** Describe the treatment being emulated.
    4.  **Outcome(s):** Specify the primary and any secondary outcomes.
    5.  **Analysis Plan:** Briefly mention the intended statistical approach (e.g., "A causal inference model will be used to estimate the treatment effect after adjusting for confounders.").
    6.  **Deviations from the target trial:** Components of the target trial that are not implemented in the data.

    **KEY TRIAL COMPONENTS:**
    {json.dumps(state['trial_info'], indent=2)}

    **OPERATIONAL DEFINITIONS IMPLEMENTED IN THE DATA:**
    Data source: {v['data_source']}
    Time zero: {v['time_zero']}
    Inclusion criteria: {v['inclusion_criteria']}
    Exclusion criteria: {v['exclusion_criteria'] or 'none'}
    Treatment strategies: {v['treatment_strategies']}
    Immortal-time bias handling: {v['immortal_time_handling']}
    Outcome: {v['outcome_definition'][0]}, counted from time zero
    Baseline covariates: {v['covariate_description']}: {', '.join(v['Covariates'])}

    The operational definitions above are exactly what was implemented. Describe the study population, treatment
    strategies, time zero and outcome as implemented. Do not describe criteria, time windows or exclusions from the KEY
    TRIAL COMPONENTS that are not among the operational definitions as applied; list them instead in a final section
    "6. Deviations from the target trial".

    **FORMATTING:**
    - Use clear headings for each section.
    - Provide ONLY the protocol text as the answer.

    **ANSWER:**
    """
    return llm.invoke(prompt)


def select_methods(df, trial_protocol, treatment_col, outcome_col, duration_col, llm):
    """Ask the LLM to select a covariate balancing method and survival analysis methods."""
    dataset_summary = {"Number of Patients": df.shape[0], "Number of Columns": df.shape[1],
                       "Treatment Column": treatment_col, "Outcome Column": outcome_col, "Duration Column": duration_col,
                       "Number Treated": int(df[treatment_col].sum())}
    prompt = f"""
    You are an expert statistician. Your task is to select the most appropriate statistical methods for a trial emulation study based on the provided protocol and data summary.

    **CONTEXT:**
    - **Trial Emulation Protocol:** {trial_protocol}
    - **Dataset Summary:** {json.dumps(dataset_summary, indent=2)}

    **INSTRUCTIONS:**
    1.  **Select a Covariate Balancing Method** from the following options:
        - Propensity Score Matching (PSM)
        - Inverse Probability of Treatment Weighting (IPTW)
        - No Balancing

    2.  **Select one or more Survival Analysis Methods** from the following options:
        - Cox Proportional Hazards
        - Kaplan-Meier Estimator
        - Parametric Survival Models
        - Random Survival Forest
        - Doubly Robust Estimation

    3.  **Provide a brief, clear rationale** for each choice, considering the data characteristics and research objectives.

    **OUTPUT FORMAT:**
    You MUST structure your response using the following XML-like tags. Do not include any other text outside these tags.
    <BALANCING_METHOD>
        <METHOD>Method Name</METHOD>
        <RATIONALE>Your rationale here</RATIONALE>
    </BALANCING_METHOD>
    <SURVIVAL_ANALYSIS>
        <METHODS>Method1, Method2</METHODS>
        <RATIONALE>Your rationale here</RATIONALE>
    </SURVIVAL_ANALYSIS>

    **ANSWER:**
    """
    response = llm.invoke(prompt)
    try:
        balancing_method = re.search(r"<METHOD>(.*?)</METHOD>", response, re.DOTALL).group(1).strip()
        rationales = re.findall(r"<RATIONALE>(.*?)</RATIONALE>", response, re.DOTALL)
        survival_methods = [m.strip() for m in re.search(r"<METHODS>(.*?)</METHODS>", response, re.DOTALL).group(1).split(",")]
        balancing_rationale = rationales[0].strip() if rationales else ""
        survival_rationale = rationales[1].strip() if len(rationales) > 1 else ""
    except AttributeError:
        balancing_method, balancing_rationale = "Propensity Score Matching (PSM)", "Default choice (response could not be parsed)."
        survival_methods, survival_rationale = ["Cox Proportional Hazards"], "Default choice (response could not be parsed)."
    if "Cox Proportional Hazards" not in survival_methods:
        survival_methods.insert(0, "Cox Proportional Hazards")  # the Cox model provides the primary effect estimate
    return balancing_method, balancing_rationale, survival_methods, survival_rationale


def apply_covariate_balancing(method, df, categorical_covariates, continuous_covariates, ignore_covariates,
                              list_of_treatment, list_of_outcome, outcome_definition, results):
    """Apply the selected balancing method; returns (balanced_df, info, unbalanced_flag, results, scaler)."""
    if "Matching" in method or "PSM" in method:
        balanced_df, info, results, flag, scaler = perform_psm(
            df, categorical_covariates, continuous_covariates, ignore_covariates, list_of_treatment, list_of_outcome,
            outcome_definition, False, results)
        return balanced_df, info, flag, results, scaler
    if "Weighting" in method or "IPTW" in method:
        balanced_df, info, results, flag = perform_iptw(df, categorical_covariates, continuous_covariates,
                                                        ignore_covariates, list_of_treatment, list_of_outcome,
                                                        outcome_definition, False, results)
        return balanced_df, info, flag, results, None
    balances = check_balance_after_matching(df, categorical_covariates + continuous_covariates, list_of_treatment[0])
    results["Balances after performing balancing"] = balances
    return df.copy(), "No balancing applied.", False, results, None


def conduct_survival_analysis(methods, df, balanced_df, categorical_covariates, continuous_covariates,
                              list_of_treatment, list_of_outcome, list_of_duration, outcome_definition):
    """Run the selected survival models; returns (results, plots, summary text, primary Cox estimate)."""
    results, plots, info = {}, {}, "Survival Analysis Summary:"
    cox_summary, cph_info, primary = run_cox_model(balanced_df, list_of_treatment, list_of_outcome, list_of_duration,
                                                   outcome_definition, categorical_covariates, continuous_covariates)
    results["Cox Proportional Hazards Summary"] = cox_summary
    info += "\n" + cph_info
    if "Kaplan-Meier Estimator" in methods:
        results["Kaplan-Meier Estimator"], plots["Kaplan-Meier Plot"] = run_kaplan_meier(
            balanced_df, list_of_treatment, list_of_outcome, list_of_duration)
    if "Parametric Survival Models" in methods:
        try:
            _, parametric_info = run_parametric_model(balanced_df, list_of_treatment, list_of_outcome, list_of_duration,
                                                      outcome_definition, categorical_covariates + continuous_covariates)
            info += "\n" + parametric_info
        except Exception as e:
            info += f"\nParametric survival model could not be fitted: {e}\n"
    if "Random Survival Forest" in methods:
        info += "\n" + run_survival_forest(balanced_df, list_of_treatment, list_of_outcome, list_of_duration,
                                           categorical_covariates, continuous_covariates, outcome_definition)
    if "Doubly Robust Estimation" in methods:
        info += "\n" + run_doubly_robust(df, list_of_treatment, list_of_outcome, categorical_covariates,
                                         continuous_covariates, outcome_definition)
    return results, plots, info, primary


def ask_clinician_about_unbalanced_covariates(unbalanced_covariates, llm):
    """The clinician agent decides whether unbalanced covariates can be dropped or substituted."""
    unbalanced_str = "".join(f"<Unbalanced_Covariate>{c}</Unbalanced_Covariate>\n" for c in unbalanced_covariates)
    issue = f'''
    Hi Clinician, I noticed the ratio of unbalanced covariates is quite high, which may impact the validity of our estimation results.
    I'd like to ask for your input: Can we drop these unbalanced covariates if they are less clinically significant?
    Alternatively, can you recommend substitute covariates that might capture similar clinical signals while improving balance?
    Here are the unbalanced covariates I identified:
    {unbalanced_str}

    Expected Structure Output:
    UNBALANCED COVARIATES
    <Unbalanced_Covariate>|<DROP or SUBSTITUTE>|<Substitute Covariate or N/A>|<Justification>
    '''
    return llm.invoke(issue)


def parse_covariate_modifications(clinician_response, llm):
    """Translate the clinician's response into covariates to drop, add or ignore."""
    if not clinician_response:
        return [], [], []
    prompt = f"""
    You are a statistician. Based on the most recent response, determine if there are any covariates to drop or add.
    You might see a response like the following:

    body mass index (BMI)|DROP|N/A|Less critical for survival analysis in sepsis patients compared to other factors.
    creatinine|SUBSTITUTE|blood urea nitrogen (BUN)|BUN can reflect kidney function similarly to creatinine.

    In this case, the covariates to drop are 'body mass index (BMI)' and 'creatinine', and the covariate to add is 'blood urea nitrogen (BUN)'.
    If a covariate can be neither dropped nor substituted, add it to the ignored covariates.

    The most recent response is as follows:
    {clinician_response}

    Use exactly this output structure (write 'None' when a list is empty):
    DROP COVARIATES: <Covariate1>,<Covariate2>, ...
    ADD COVARIATES: <Covariate1>,<Covariate2>, ...
    IGNORE COVARIATES: <Covariate1>,<Covariate2>, ...
    """
    response = llm.invoke(prompt)

    def _parse(tag):
        m = re.search(rf"{tag} COVARIATES:\s*(.*)", response)
        if not m or m.group(1).strip().lower().startswith("none"):
            return []
        return [c.strip().lower() for c in m.group(1).split(",") if c.strip()]

    return _parse("DROP"), _parse("ADD"), _parse("IGNORE")


def plan_subgroup_analyses(state, covariate_ranges, llm, max_subgroups=2):
    """Prespecify clinically relevant subgroup analyses (chosen before looking at any effect estimate)."""
    prompt = f"""
    You are a clinical epidemiologist designing a trial emulation. Before any results are available, prespecify up to
    {max_subgroups} clinically relevant subgroup analyses to explore heterogeneity of treatment effect.

    Clinical question: {state.get('initial_question')}
    Treatment: {state['variables']['treatment_definition'][0]}
    Outcome: {state['variables']['outcome_definition'][0]}

    Available baseline covariates and their observed ranges:
    {json.dumps(covariate_ranges, indent=2)}

    Choose covariates that are plausible effect modifiers for this treatment and outcome, and a clinically meaningful
    cut-off within the observed range for each (subgroups are defined as below vs. at-or-above the cut-off).

    Return ONLY a JSON list, e.g. [{{"covariate": "age", "threshold": 65, "rationale": "..."}}].
    Use covariate names exactly as listed above.
    """
    plan = parse_json_response(llm.invoke(prompt), default=[])
    valid = []
    for item in plan if isinstance(plan, list) else []:
        cov, thr = item.get("covariate"), item.get("threshold")
        if cov in covariate_ranges and isinstance(thr, (int, float)):
            lo, hi = covariate_ranges[cov]
            if lo < thr <= hi:
                valid.append(item)
    return valid[:max_subgroups]


def generate_final_report(state: dict, llm) -> str:
    """Compose the trial emulation report from all analysis results."""
    v = state["variables"]
    report_context = f"""
    ### INITIAL QUESTION
    {state["initial_question"]}

    ### DATA SOURCE
    {v['data_source']}

    ### PRIMARY OUTCOME: {v['outcome_definition'][0]}
    #### Trial Protocol
    {state['trial_protocol']}

    #### Balancing Information
    {state['balancing_information']}

    #### Main Analysis Results
    {state['analysis_results']}

    #### Prespecified Subgroup Analysis Results
    {state['subgroup_results']}

    ### SAMPLE SIZE CALCULATION
    {state['sample_size']}

    ### E-VALUE CALCULATION
    {state['e_value']}
    """
    prompt = f"""
    You are an expert medical writer and statistician tasked with composing a formal, publication-quality report based on the results of a trial emulation study.

    **CONTEXT AND DATA:**
    You are provided with the initial research question and a structured summary of all statistical analyses performed.
    ---
    {report_context}
    ---

    **INSTRUCTIONS:**
    Based ONLY on the provided context and data, generate a complete and well-structured report. The report MUST include all of the following sections, in this order:

    1.  **Title Page:** A clear and informative title for the study.
    2.  **Abstract:** A concise summary covering the study's objective, methods, key results, and conclusions.
    3.  **Introduction:** Provide background on the clinical question, the rationale for the trial emulation, and the study's objectives.
    4.  **Methods:** Briefly describe the data source, key eligibility criteria, treatment strategies, and the statistical methods used (e.g., confounder adjustment, immortal-time bias handling, outcome modeling).
    5.  **Results:** Present the main findings.
        * Start with the primary analysis results.
        * Incorporate key numerical results, such as Hazard Ratios (HR), 95% Confidence Intervals (CI), and p-values where provided.
        * Include the results of the prespecified subgroup analyses, labelled as exploratory.
    6.  **Discussion:** Interpret the key findings. Discuss the consistency of the results, the study's strengths, and its limitations (e.g., residual or unmeasured confounding, measurement error, single data source).
    7.  **Conclusion:** Briefly summarize the study's main takeaways.

    **WRITING STYLE:**
    * Be objective, formal, and use clear scientific language.
    * When reporting numerical results, explain their meaning in the context of the study (e.g., "The hazard ratio of 1.10 suggests a 10% increased risk...").
    * Pay close attention to confidence intervals and p-values when making statements about statistical significance.
    * Observational associations must not be described as proof of causation.
    * Describe the eligibility criteria, treatment strategies, time zero and outcome exactly as implemented in the Trial
      Protocol (not the deviations from the target trial).
    * Report secondary analyses separately from the primary analysis, with the caveats stated for them in the context.

    Generate the complete report. Provide ONLY the report and NOTHING else.
    """
    return llm.invoke(prompt)


def run_rwd_analyst(state, llm):
    """Run the full trial emulation workflow; adds 'current_report' and 'primary_result' to the state."""
    v = state["variables"]
    df = v["cleaned_dataframe"].copy()
    df.columns = [c.lower() for c in df.columns]
    id_col = v["patient_id"].lower()
    treatment, outcome, duration = [v["Treatment"].lower()], [v["Outcome"].lower()], [v["Duration"].lower()]
    time_col = v["treatment_time_col"].lower() if v.get("treatment_time_col") else None
    covariates = [c.lower() for c in v["Covariates"] if c.lower() not in (id_col, treatment[0], outcome[0], duration[0], time_col)]
    outcome_definition = v["outcome_definition"]
    if df[treatment[0]].nunique() < 2:
        raise ValueError("The cohort contains no treated or no untreated patients; a trial emulation is not possible.")

    time_dependent = is_time_dependent_treatment(df, treatment[0], time_col)
    grace = float(v["grace_period_days"])
    drugs = v["treatment_definition"][0].split(" vs ")[0]
    v["treatment_strategies"] = (
        f"initiate {drugs} within {grace:g} days after time zero vs do not initiate {drugs} within {grace:g} days"
        if time_dependent else v["treatment_definition"][0])
    v["immortal_time_handling"] = (
        f"clone-censor-weight with a {grace:g}-day grace period (a substantial share of initiations occur after time zero)"
        if time_dependent else "not applied (treatment initiation times are unavailable or close to time zero)")
    state["trial_protocol"] = build_trial_emulation_protocol(state, llm)
    drop, add, ignore, clinician_response = [], [], [], ""
    for attempt in range(2):
        d, a, i = parse_covariate_modifications(clinician_response, llm)
        drop, add, ignore = drop + d, add + a, ignore + i
        current = [c for c in dict.fromkeys(covariates + [c for c in add if c in df.columns]) if c not in drop]
        data, categorical, continuous, _, _, _ = clean_data(current, treatment, outcome, duration, [id_col], df,
                                                            extra_columns=[time_col] if time_col else None)
        if attempt == 0:
            balancing_method, balancing_rationale, survival_methods, survival_rationale = select_methods(
                data, state["trial_protocol"], treatment[0], outcome[0], duration[0], llm)
        results = {}
        balanced_df, balancing_info, unbalanced, results, scaler = apply_covariate_balancing(
            balancing_method, data, categorical, continuous, ignore, treatment, outcome, outcome_definition, results)
        if scaler is not None and continuous:
            balanced_df[continuous] = scaler.inverse_transform(balanced_df[continuous])
        if not unbalanced or attempt == 1:
            break
        balances = results["Balances after performing balancing"]
        clinician_response = ask_clinician_about_unbalanced_covariates(list(balances[balances > 0.1].index), llm)

    if time_dependent:
        # Models with exposure fixed at baseline are biased by immortal time; only a Cox model is kept for comparison.
        survival_methods = ["Cox Proportional Hazards"]
        survival_rationale = ("Treatment initiation is time dependent, so the primary analysis is a weighted Cox model "
                              "after clone-censor-weight; a Cox model with exposure fixed at baseline is shown for comparison.")
    state["balancing_information"] = (f"Balancing method: {balancing_method}. Rationale: {balancing_rationale}\n"
                                      f"Survival methods: {', '.join(survival_methods)}. Rationale: {survival_rationale}\n"
                                      f"{balancing_info}")
    results, plots, survival_info, primary = conduct_survival_analysis(
        survival_methods, data, balanced_df, categorical, continuous, treatment, outcome, duration, outcome_definition)
    primary["method"] = f"{balancing_method} + Cox proportional hazards"
    subgroup_data = dict(data=balanced_df, exposure_col=treatment[0], duration_col=duration[0], event_col=outcome[0],
                         weights_col="iptw_weight" if "iptw_weight" in balanced_df.columns else None)

    if time_dependent:
        clones = clone_censor_weight(data, id_col, duration[0], outcome[0], time_col, categorical + continuous, grace)
        follow_up = v.get("follow_up_days") or float(data[duration[0]].max())
        ccw_info, primary = run_ccw_analysis(clones, id_col, outcome_definition, follow_up)
        primary["method"] = f"clone-censor-weight (grace period {grace:g} days) + weighted Cox"
        state["analysis_results"] = (
            f"Treatment initiation is time dependent (a substantial share of treated patients started after admission), "
            f"so the primary analysis uses the clone-censor-weight method.\n{ccw_info}\n"
            "Secondary analysis with exposure fixed at baseline (any initiation during the index admission); "
            f"this analysis is susceptible to immortal-time bias and is shown for comparison only:\n{survival_info}")
        subgroup_data = dict(data=clones, exposure_col="arm", duration_col="stop", event_col="event",
                             weights_col="weight", entry_col="start", cluster_col=id_col)
    else:
        state["analysis_results"] = survival_info

    ranges = {c: [float(data[c].min()), float(data[c].max())] for c in continuous if data[c].nunique() > 2}
    subgroup_texts = []
    for item in plan_subgroup_analyses(state, ranges, llm):
        try:
            subgroup_texts.append(run_subgroup_cox(subgroup_col=item["covariate"], threshold=float(item["threshold"]),
                                                   **subgroup_data) + f"  Rationale: {item.get('rationale', '')}\n")
        except Exception as e:
            logger.warning("Subgroup analysis on %s failed: %s", item["covariate"], e)
    state["subgroup_results"] = "\n".join(subgroup_texts) or "No prespecified subgroup analysis could be performed."

    try:
        size, _ = compute_cox_sample_size(data, treatment[0], outcome[0], primary["effect_size"])
        state["sample_size"] = (f"Based on the observed event rate and effect size, the estimated sample size to achieve "
                                f"80% power at an alpha of 0.05 is {size} patients.")
    except Exception as e:
        state["sample_size"] = f"Sample size calculation could not be performed: {e}"
    evalue = calculate_evalues(data, outcome[0], primary["effect_size"])
    state["e_value"] = (f"E-value for the point estimate: {evalue['e_value']:.3f} (the minimum strength of association "
                        "that an unmeasured confounder would need with both treatment and outcome to explain away the "
                        "observed association).")

    state["primary_result"] = {**primary, "n_patients": int(len(data)), "n_treated": int(data[treatment[0]].sum()),
                               "outcome": outcome_definition[0], "treatment": v["treatment_strategies"]}
    state["plots"] = plots
    state["current_report"] = generate_final_report(state, llm)
    return state
