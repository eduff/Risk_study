import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calculate_and_plot_paf(
    df: pd.DataFrame,
    risk_cols: list,
    time_col: str,
    event_col: str,
    stratify_col: str = None,
    confounders: list = None,
    strata_to_include: list = None,
    alpha: float = 0.05,
    penalizer: float = 0.0,
    negligible_threshold: float = 0.5,
    min_risk_count: int = 10,
    vif_threshold: float = 10.0
):
    """
    Calculates risk contributions and plots decomposition charts with robustness checks.

    This function includes checks for collinearity (VIF) and complete separation to ensure
    model stability. If issues are detected, it plots diagnostics and adapts the model.
    It fits a multivariable Cox model to calculate the total PAF for a stable set of risk factors.

    Args:
        df (pd.DataFrame): The input dataframe.
        risk_cols (list): A list of binary (0/1) or continuous risk factor columns.
        time_col (str): The column representing survival time.
        event_col (str): The column indicating event occurrence (1 for event, 0 for censored).
        stratify_col (str, optional): A categorical column to stratify the analysis by.
        confounders (list, optional): A list of columns to adjust for as confounders.
        strata_to_include (list, optional): A list of specific strata to include in plots.
        alpha (float, optional): The significance level for the Cox model. Defaults to 0.05.
        penalizer (float, optional): The penalizer term for regularization. Defaults to 0.0.
        negligible_threshold (float, optional): PAF percentage below which a risk factor is
                                             removed from plots. Defaults to 0.5.
        min_risk_count (int, optional): The minimum number of exposed cases required to model
                                        a binary risk factor. Defaults to 10.
        vif_threshold (float, optional): VIF threshold for removing collinear variables. Defaults to 10.0.
    """
    if confounders is None:
        confounders = []

    paf_results = []
    stratum_metrics = {}

    # --- Data Cleaning ---
    essential_cols = [time_col, event_col] + risk_cols + confounders
    if stratify_col:
        essential_cols.append(stratify_col)
    
    clean_df = df[essential_cols].dropna().copy()
    clean_df = clean_df[clean_df[time_col] > 0]
    
    print(f"Analyzing {len(clean_df)} complete cases.")

    # --- Calculate Overall Population Mean Incidence Rate ---
    total_person_years = clean_df[time_col].sum() / 1000
    total_events = clean_df[event_col].sum()
    overall_mean_incidence = total_events / total_person_years if total_person_years > 0 else 0
    print(f"Overall Mean Incidence Rate: {overall_mean_incidence:.2f} per 1000 person-years")

    # Determine groups for stratification
    if stratify_col:
        all_groups = sorted(clean_df[stratify_col].unique())
        groups_to_process = strata_to_include if strata_to_include is not None else all_groups
    else:
        all_groups = ['Overall']
        groups_to_process = ['Overall']
        stratify_col = 'dummy_strata'
        clean_df[stratify_col] = 'Overall'

    for group in groups_to_process:
        if group not in all_groups:
            print(f"WARNING: Stratum '{group}' from strata_to_include not found in data. Skipping.")
            continue
            
        group_df = clean_df[clean_df[stratify_col] == group].copy()
        
        if len(group_df) < min_risk_count * 2:
            print(f"Skipping stratum '{group}' due to insufficient data overall.")
            continue
        
        # --- NEW: Convert boolean risk columns to integer (0/1) to standardize them ---
        for risk in risk_cols:
            if pd.api.types.is_bool_dtype(group_df[risk]):
                group_df[risk] = group_df[risk].astype(int)

        selected_features = risk_cols.copy()
        
        # Robust VIF Check
        collinearity_found = False
        while len(selected_features) > 1:
            predictors_for_vif = selected_features + confounders
            numeric_vif_df = group_df[predictors_for_vif].copy()
            for col in numeric_vif_df.columns:
                if not pd.api.types.is_numeric_dtype(numeric_vif_df[col]):
                    numeric_vif_df[col] = numeric_vif_df[col].astype('category').cat.codes
            numeric_vif_df.dropna(inplace=True)
            if numeric_vif_df.empty:
                print(f"WARNING: No data left for VIF check in stratum '{group}'. Skipping VIF check.")
                break
            X_vif = add_constant(numeric_vif_df, has_constant='add')
            vif_series = pd.Series(
                [variance_inflation_factor(X_vif.values, X_vif.columns.get_loc(feature)) for feature in selected_features],
                index=selected_features
            )
            if vif_series.max() < vif_threshold:
                break
            feature_to_remove = vif_series.idxmax()
            print(f"INFO: Removing '{feature_to_remove}' from stratum '{group}' due to high collinearity (VIF={vif_series.max():.2f}).")
            selected_features.remove(feature_to_remove)
            collinearity_found = True

        if collinearity_found:
            print(f"INFO: Plotting correlation matrix for stratum '{group}' to show collinearity.")
            corr_matrix = group_df[risk_cols].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
            plt.title(f"Correlation Matrix of Initial Risk Factors\nStratum: {group}")
            plt.show()

        # Separation Check
        features_to_remove_for_separation = []
        for risk in selected_features:
            if set(group_df[risk].dropna().unique()) <= {0, 1}:
                crosstab = pd.crosstab(group_df[risk], group_df[event_col])
                if 0 in crosstab.index and 1 in crosstab.index and 0 in crosstab.columns and 1 in crosstab.columns:
                    if (crosstab.loc[0, 0] == 0 or crosstab.loc[0, 1] == 0 or
                        crosstab.loc[1, 0] == 0 or crosstab.loc[1, 1] == 0):
                        print(f"INFO: Removing '{risk}' from stratum '{group}' due to complete separation.")
                        features_to_remove_for_separation.append(risk)
                else:
                    print(f"INFO: Removing '{risk}' from stratum '{group}' due to lack of variation in risk/outcome crosstab.")
                    features_to_remove_for_separation.append(risk)
        
        selected_features = [f for f in selected_features if f not in features_to_remove_for_separation]
        
        if not selected_features:
            print(f"WARNING: No stable features left to model for stratum '{group}'. Skipping.")
            continue

        person_years = group_df[time_col].sum() / 1000
        n_events = group_df[event_col].sum()
        observed_incidence = n_events / person_years if person_years > 0 else 0
        
        multivar_model_df = group_df[selected_features + [time_col, event_col] + confounders]
        cph_multi = CoxPHFitter(penalizer=penalizer)
        try:
            cph_multi.fit(multivar_model_df, duration_col=time_col, event_col=event_col)
            rr = cph_multi.predict_partial_hazard(multivar_model_df[selected_features + confounders])
            paf_total = 1 - (1 / rr.mean())
            if paf_total < 0: paf_total = 0
        except Exception as e:
            print(f"WARNING: Could not fit multivariable model for stratum '{group}'. EXCLUDING. Error: {e}")
            continue

        individual_pafs = {}
        for risk in selected_features:
            is_binary = set(group_df[risk].dropna().unique()) <= {0, 1}
            if is_binary and group_df[risk].sum() < min_risk_count:
                individual_pafs[risk] = 0
                continue
            prevalence = group_df[risk].mean() if is_binary else (group_df[risk] > group_df[risk].median()).mean()
            if prevalence <= 0 or prevalence >= 1:
                individual_pafs[risk] = 0
                continue
            
            model_df = group_df[[risk, time_col, event_col] + confounders]
            cph_single = CoxPHFitter(penalizer=penalizer)
            try:
                cph_single.fit(model_df, duration_col=time_col, event_col=event_col)
                hr = cph_single.hazard_ratios_[risk]
                paf = (prevalence * (hr - 1)) / (prevalence * (hr - 1) + 1) if hr > 1 else 0
                individual_pafs[risk] = paf
            except Exception:
                individual_pafs[risk] = 0

        sum_individual_pafs = sum(individual_pafs.values())
        if sum_individual_pafs > 0:
            for risk, paf_val in individual_pafs.items():
                scaled_paf = (paf_val / sum_individual_pafs) * paf_total
                paf_results.append({'stratum': group, 'risk_factor': risk, 'PAF': scaled_paf * 100})

        risk_deleted_incidence = observed_incidence * (1 - paf_total)
        stratum_metrics[group] = {
            'observed_incidence': observed_incidence,
            'risk_deleted_incidence': risk_deleted_incidence
        }
    
    if not paf_results:
        print("No PAF results could be calculated for any included stratum.")
        return pd.DataFrame()

    results_df = pd.DataFrame(paf_results)
    
    paf_pivot = results_df.pivot(index='stratum', columns='risk_factor', values='PAF').fillna(0)
    
    total_paf_by_risk = paf_pivot.sum()
    significant_risks = total_paf_by_risk[total_paf_by_risk >= negligible_threshold].index.tolist()
    paf_pivot = paf_pivot[significant_risks]

    if paf_pivot.empty:
        print("No significant risk factors to plot.")
        return results_df
        
    successful_strata = [g for g in groups_to_process if g in paf_pivot.index]
    paf_pivot_plot = paf_pivot.loc[successful_strata]
    paf_pivot_plot = paf_pivot_plot[paf_pivot_plot.sum().sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(14, 8))
    color_map = {risk: color for risk, color in zip(paf_pivot_plot.columns, plt.cm.viridis(np.linspace(0, 1, len(paf_pivot_plot.columns))))}

    for i, stratum in enumerate(paf_pivot_plot.index):
        metrics = stratum_metrics.get(stratum)
        if not metrics: continue
        risk_deleted_inc = metrics['risk_deleted_incidence']
        observed_inc = metrics['observed_incidence']
        start_percent = ((risk_deleted_inc - overall_mean_incidence) / overall_mean_incidence) * 100
        current_pos = start_percent
        ax.text(current_pos - 1, i, f'{risk_deleted_inc:.1f}', ha='right', va='center', fontsize=9, color='dimgrey')
        for j, risk_factor in enumerate(paf_pivot_plot.columns):
            paf_percent = paf_pivot_plot.loc[stratum, risk_factor]
            segment_width = (paf_percent / 100 * observed_inc / overall_mean_incidence) * 100
            ax.barh(i, segment_width, left=current_pos, color=color_map[risk_factor], label=risk_factor if i == 0 else "", height=0.6, edgecolor='white')
            current_pos += segment_width
        ax.text(current_pos + 1, i, f'{observed_inc:.1f}', ha='left', va='center', fontsize=9, color='black', weight='bold')

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(np.arange(len(paf_pivot_plot.index)))
    ax.set_yticklabels(paf_pivot_plot.index)
    xlabel_text = f'Percent Change from Population Mean ({overall_mean_incidence:.1f} per 1000 person-years)'
    ax.set_xlabel(xlabel_text, fontsize=12)
    ax.set_ylabel(stratify_col, fontsize=12)
    ax.set_title('Decomposition of Incidence Rate by Risk Factor and Stratum', fontsize=16, pad=20)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Modifiable Risks', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # --- Plotting: Pie Charts ---
    num_strata = len(paf_pivot_plot.index)
    fig, axes = plt.subplots(1, num_strata, figsize=(6 * num_strata, 6), squeeze=False)
    for i, stratum in enumerate(paf_pivot_plot.index):
        stratum_paf = paf_pivot_plot.loc[stratum]
        stratum_paf = stratum_paf[stratum_paf > 0]
        pie_colors = [color_map.get(risk, 'grey') for risk in stratum_paf.index]
        if stratum_paf.empty:
            axes[0, i].text(0.5, 0.5, 'No attributable risk\nfrom selected factors', ha='center', va='center')
        else:
            axes[0, i].pie(stratum_paf, labels=stratum_paf.index, autopct='%1.1f%%', startangle=90, colors=pie_colors)
        axes[0, i].set_title(f'Risk Contribution for: {stratum}', fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- Plotting for Expected vs. Observed Incidence ---
    if stratify_col != 'dummy_strata' and len(successful_strata) > 1:
        print("\n--- Analyzing Differences Between Observed and Expected Incidence ---")
        ref_stratum_name = clean_df[stratify_col].value_counts().idxmax()
        print(f"Using '{ref_stratum_name}' as the reference stratum for risk associations.")
        ref_df = clean_df[clean_df[stratify_col] == ref_stratum_name]
        cph_ref = CoxPHFitter(penalizer=penalizer)
        cph_ref.fit(ref_df[risk_cols + [time_col, event_col] + confounders], duration_col=time_col, event_col=event_col)
        ref_risk_deleted_inc = stratum_metrics[ref_stratum_name]['risk_deleted_incidence']
        comparison_data = []
        for stratum in successful_strata:
            stratum_df = clean_df[clean_df[stratify_col] == stratum]
            partial_hazards = cph_ref.predict_partial_hazard(stratum_df[risk_cols + confounders])
            mean_partial_hazard = partial_hazards.mean()
            expected_incidence = ref_risk_deleted_inc * mean_partial_hazard
            comparison_data.append({'stratum': stratum, 'Observed Incidence': stratum_metrics[stratum]['observed_incidence'], 'Expected Incidence': expected_incidence})
        comp_df = pd.DataFrame(comparison_data).set_index('stratum')
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, stratum in enumerate(comp_df.index):
            observed = comp_df.loc[stratum, 'Observed Incidence']
            expected = comp_df.loc[stratum, 'Expected Incidence']
            ax.plot([observed, expected], [i, i], color='grey', linestyle='-', linewidth=1, zorder=1)
            ax.plot(observed, i, 'o', color='royalblue', markersize=9, label='Observed Incidence' if i == 0 else "", zorder=2)
            ax.plot(expected, i, 'o', color='darkorange', markersize=9, label='Expected Incidence' if i == 0 else "", zorder=2)
        ax.set_yticks(range(len(comp_df.index)))
        ax.set_yticklabels(comp_df.index)
        ax.set_xlabel("Incidence Rate (events per 1000 person-years)")
        ax.set_ylabel(stratify_col)
        ax.set_title("Observed vs. Expected Incidence Rates\n(Expected rate based on reference stratum's risk associations)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        ax.grid(axis='x', linestyle='--')
        plt.tight_layout()
        plt.show()

    # --- Plotting for Risk Prevalence Differences ---
    if stratify_col != 'dummy_strata':
        print("\n--- Analyzing Differences in Risk Factor Prevalence Across Strata ---")
        significant_diff_risks = []
        for risk in risk_cols:
            is_binary = set(clean_df[risk].dropna().unique()) <= {0, 1}
            if is_binary:
                contingency_table = pd.crosstab(clean_df[stratify_col], clean_df[risk])
                if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2: continue
                chi2, p, _, _ = chi2_contingency(contingency_table)
                if p < 0.05:
                    significant_diff_risks.append(risk)
            else: # Continuous
                samples = [group_df[risk].dropna() for name, group_df in clean_df.groupby(stratify_col)]
                if len(samples) < 2: continue
                f_val, p = f_oneway(*samples)
                if p < 0.05:
                    significant_diff_risks.append(risk)
        
        if not significant_diff_risks:
            print("No significant differences in risk factor prevalence found across strata.")
        else:
            print(f"Found significant prevalence differences for: {significant_diff_risks}")
            n_risks = len(significant_diff_risks)
            n_cols = 2
            n_rows = (n_risks + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
            axes = axes.flatten()

            for i, risk in enumerate(significant_diff_risks):
                sns.barplot(x=stratify_col, y=risk, data=clean_df, ax=axes[i], palette='Set2', order=successful_strata)
                y_label = 'Prevalence' if set(clean_df[risk].dropna().unique()) <= {0, 1} else 'Mean Value'
                axes[i].set_ylabel(y_label)
                axes[i].set_title(f'Prevalence of {risk}')
            
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.show()

    return paf_pivot


# Example Usage:
if __name__ == '__main__':
    # --- Generate synthetic data ---
    n_samples = 5000
    df = pd.DataFrame({
        'Age': np.random.uniform(50, 80, n_samples),
        'Sex': np.random.choice([0, 1], n_samples),
        'APOE4_status': np.random.choice(['Non-carrier', 'Carrier', 'Unknown'], n_samples, p=[0.70, 0.25, 0.05]),
        'risk_smoking': np.random.choice([True, False], n_samples, p=[0.2, 0.8]), # Using boolean
        'time_to_event': np.random.weibull(2, n_samples) * 10,
    })
    
    # Add a variable that is highly correlated with another to test VIF
    df['risk_hypertension'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    df['risk_high_bp_meds'] = df['risk_hypertension'] * np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    
    df['risk_diabetes'] = 0
    df.loc[df['APOE4_status'] == 'Carrier', 'risk_diabetes'] = np.random.choice([0, 1], size=(df['APOE4_status'] == 'Carrier').sum(), p=[0.7, 0.3])
    df.loc[df['APOE4_status'] == 'Non-carrier', 'risk_diabetes'] = np.random.choice([0, 1], size=(df['APOE4_status'] == 'Non-carrier').sum(), p=[0.9, 0.1])
    
    base_risk = -3.5
    risk_score = (base_risk + 
                  (df['APOE4_status'] == 'Carrier') * 1.5 + 
                  df['risk_diabetes'] * 0.4 + 
                  df['risk_hypertension'] * 0.3 + 
                  df['risk_smoking'] * 0.5)
    event_prob = 1 / (1 + np.exp(-risk_score))
    df['event_status'] = (np.random.rand(n_samples) < event_prob).astype(int)
    
    df.loc[df['APOE4_status'] == 'Unknown', 'event_status'] = 0

    # --- Run the PAF analysis ---
    risk_factor_columns = ['risk_diabetes', 'risk_hypertension', 'risk_high_bp_meds', 'risk_smoking']
    
    print("--- Running Analysis Stratified by APOE4 Status (excluding 'Unknown') ---")
    paf_results_pivot = calculate_and_plot_paf(
        df=df,
        risk_cols=risk_factor_columns,
        time_col='time_to_event',
        event_col='event_status',
        stratify_col='APOE4_status',
        confounders=['Age', 'Sex'],
        strata_to_include=['Carrier', 'Non-carrier']
    )
    
    print("\n--- Corrected & Scaled PAF Contributions (%) ---")
    print(paf_results_pivot)



import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway

def calculate_and_plot_paf2(
    df: pd.DataFrame,
    risk_cols: list,
    time_col: str,
    event_col: str,
    stratify_col: str = None,
    confounders: list = None,
    alpha: float = 0.05,
    penalizer: float = 0.0,
    negligible_threshold: float = 0.5,
    min_risk_count: int = 40
):
    """
    Calculates risk contributions, filters negligible factors, and plots decomposition charts.

    This function fits a multivariable Cox model to calculate the total Population
    Attributable Fraction (PAF) for all risk factors combined. It then uses single-variable
    models to determine the proportional contribution of each risk factor to this total.
    It includes robustness checks to skip modeling on strata or risks with insufficient data.

    Args:
        df (pd.DataFrame): The input dataframe.
        risk_cols (list): A list of binary (0/1) or continuous risk factor columns.
        time_col (str): The column representing survival time.
        event_col (str): The column indicating event occurrence (1 for event, 0 for censored).
        stratify_col (str, optional): A categorical column to stratify the analysis by.
        confounders (list, optional): A list of columns to adjust for as confounders.
        alpha (float, optional): The significance level for the Cox model. Defaults to 0.05.
        penalizer (float, optional): The penalizer term for regularization. Defaults to 0.0.
        negligible_threshold (float, optional): PAF percentage below which a risk factor is
                                             considered negligible and removed from plots. Defaults to 0.5.
        min_risk_count (int, optional): The minimum number of exposed cases required to model
                                        a binary risk factor within a stratum. Defaults to 10.
    """
    if confounders is None:
        confounders = []

    paf_results = []
    stratum_metrics = {}

    # --- Data Cleaning ---
    essential_cols = [time_col, event_col] + risk_cols + confounders
    if stratify_col:
        essential_cols.append(stratify_col)
    
    clean_df = df[essential_cols].dropna().copy()
    clean_df = clean_df[clean_df[time_col] > 0]
    
    print(f"Analyzing {len(clean_df)} complete cases.")

    # --- Calculate Overall Population Mean Incidence Rate ---
    total_person_years = clean_df[time_col].sum() / 1000
    total_events = clean_df[event_col].sum()
    overall_mean_incidence = total_events / total_person_years if total_person_years > 0 else 0
    print(f"Overall Mean Incidence Rate: {overall_mean_incidence:.2f} per 1000 person-years")

    # Determine groups for stratification
    if stratify_col:
        groups = sorted(clean_df[stratify_col].unique())
    else:
        groups = ['Overall']
        stratify_col = 'dummy_strata'
        clean_df[stratify_col] = 'Overall'

    for group in groups:
        group_df = clean_df[clean_df[stratify_col] == group].copy()
        
        if len(group_df) < min_risk_count * 2: # Need at least some data to model
            print(f"Skipping stratum '{group}' due to insufficient data overall.")
            continue
            
        person_years = group_df[time_col].sum() / 1000
        n_events = group_df[event_col].sum()
        observed_incidence = n_events / person_years if person_years > 0 else 0
        
        multivar_model_df = group_df[risk_cols + [time_col, event_col] + confounders]
        cph_multi = CoxPHFitter(penalizer=penalizer)
        try:
            cph_multi.fit(multivar_model_df, duration_col=time_col, event_col=event_col)
            rr = cph_multi.predict_partial_hazard(multivar_model_df[risk_cols + confounders])
            paf_total = 1 - (1 / rr.mean())
            if paf_total < 0: paf_total = 0
        except Exception as e:
            print(f"WARNING: Could not fit multivariable model for stratum '{group}'. EXCLUDING from plot. Error: {e}")
            continue

        individual_pafs = {}
        for risk in risk_cols:
            is_binary = set(group_df[risk].dropna().unique()) <= {0, 1}
            
            # --- Robustness Check for Sparse Data ---
            if is_binary and group_df[risk].sum() < min_risk_count:
                print(f"INFO: Skipping risk '{risk}' in stratum '{group}' due to insufficient cases (< {min_risk_count}).")
                individual_pafs[risk] = 0
                continue

            prevalence = group_df[risk].mean() if is_binary else (group_df[risk] > group_df[risk].median()).mean()
            if prevalence <= 0 or prevalence >= 1 or group_df[risk].nunique() < 2:
                individual_pafs[risk] = 0
                continue
            
            model_df = group_df[[risk, time_col, event_col] + confounders]
            cph_single = CoxPHFitter(penalizer=penalizer)
            try:
                cph_single.fit(model_df, duration_col=time_col, event_col=event_col)
                hr = cph_single.hazard_ratios_[risk]
                paf = (prevalence * (hr - 1)) / (prevalence * (hr - 1) + 1) if hr > 1 else 0
                individual_pafs[risk] = paf
            except Exception:
                individual_pafs[risk] = 0

        sum_individual_pafs = sum(individual_pafs.values())
        if sum_individual_pafs > 0:
            for risk, paf_val in individual_pafs.items():
                scaled_paf = (paf_val / sum_individual_pafs) * paf_total
                paf_results.append({'stratum': group, 'risk_factor': risk, 'PAF': scaled_paf * 100})

        risk_deleted_incidence = observed_incidence * (1 - paf_total)
        stratum_metrics[group] = {
            'observed_incidence': observed_incidence,
            'risk_deleted_incidence': risk_deleted_incidence
        }

    if not paf_results:
        print("No PAF results could be calculated for any stratum.")
        return pd.DataFrame()

    results_df = pd.DataFrame(paf_results)
    
    # --- Filter out negligible risk factors ---
    paf_pivot = results_df.pivot(index='stratum', columns='risk_factor', values='PAF')
    total_paf_by_risk = paf_pivot.sum()
    significant_risks = total_paf_by_risk[total_paf_by_risk >= negligible_threshold].index.tolist()
    paf_pivot = paf_pivot[significant_risks]

    # --- Plotting: Stacked Bar Chart ---
    if paf_pivot.empty:
        print("No significant risk factors to plot.")
        return results_df
        
    successful_strata = [g for g in groups if g in paf_pivot.index]
    paf_pivot_plot = paf_pivot.loc[successful_strata]
    paf_pivot_plot = paf_pivot_plot[paf_pivot_plot.sum().sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(14, 8))
    color_map = {risk: color for risk, color in zip(paf_pivot_plot.columns, plt.cm.viridis(np.linspace(0, 1, len(paf_pivot_plot.columns))))}

    for i, stratum in enumerate(paf_pivot_plot.index):
        metrics = stratum_metrics.get(stratum)
        if not metrics: continue
        risk_deleted_inc = metrics['risk_deleted_incidence']
        observed_inc = metrics['observed_incidence']
        start_percent = ((risk_deleted_inc - overall_mean_incidence) / overall_mean_incidence) * 100
        current_pos = start_percent
        ax.text(current_pos - 1, i, f'{risk_deleted_inc:.1f}', ha='right', va='center', fontsize=9, color='dimgrey')
        for j, risk_factor in enumerate(paf_pivot_plot.columns):
            paf_percent = paf_pivot_plot.loc[stratum, risk_factor]
            segment_width = (paf_percent / 100 * observed_inc / overall_mean_incidence) * 100
            ax.barh(i, segment_width, left=current_pos, color=color_map[risk_factor], label=risk_factor if i == 0 else "", height=0.6, edgecolor='white')
            current_pos += segment_width
        ax.text(current_pos + 1, i, f'{observed_inc:.1f}', ha='left', va='center', fontsize=9, color='black', weight='bold')

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(np.arange(len(paf_pivot_plot.index)))
    ax.set_yticklabels(paf_pivot_plot.index)
    xlabel_text = f'Percent Change from Population Mean ({overall_mean_incidence:.1f} per 1000 person-years)'
    ax.set_xlabel(xlabel_text, fontsize=12)
    ax.set_ylabel(stratify_col, fontsize=12)
    ax.set_title('Decomposition of Incidence Rate by Risk Factor and Stratum', fontsize=16, pad=20)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Modifiable Risks', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # --- Plotting: Pie Charts ---
    num_strata = len(paf_pivot_plot.index)
    fig, axes = plt.subplots(1, num_strata, figsize=(6 * num_strata, 6), squeeze=False)
    for i, stratum in enumerate(paf_pivot_plot.index):
        stratum_paf = paf_pivot_plot.loc[stratum]
        stratum_paf = stratum_paf[stratum_paf > 0]
        pie_colors = [color_map[risk] for risk in stratum_paf.index]
        if stratum_paf.empty:
            axes[0, i].text(0.5, 0.5, 'No attributable risk\nfrom selected factors', ha='center', va='center')
        else:
            axes[0, i].pie(stratum_paf, labels=stratum_paf.index, autopct='%1.1f%%', startangle=90, colors=pie_colors)
        axes[0, i].set_title(f'Risk Contribution for: {stratum}', fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- Plotting for Expected vs. Observed Incidence ---
    if stratify_col != 'dummy_strata' and len(successful_strata) > 1:
        print("\n--- Analyzing Differences Between Observed and Expected Incidence ---")
        ref_stratum_name = clean_df[stratify_col].value_counts().idxmax()
        print(f"Using '{ref_stratum_name}' as the reference stratum for risk associations.")
        ref_df = clean_df[clean_df[stratify_col] == ref_stratum_name]
        cph_ref = CoxPHFitter(penalizer=penalizer)
        cph_ref.fit(ref_df[risk_cols + [time_col, event_col] + confounders], duration_col=time_col, event_col=event_col)
        ref_risk_deleted_inc = stratum_metrics[ref_stratum_name]['risk_deleted_incidence']
        comparison_data = []
        for stratum in successful_strata:
            stratum_df = clean_df[clean_df[stratify_col] == stratum]
            partial_hazards = cph_ref.predict_partial_hazard(stratum_df[risk_cols + confounders])
            mean_partial_hazard = partial_hazards.mean()
            expected_incidence = ref_risk_deleted_inc * mean_partial_hazard
            comparison_data.append({'stratum': stratum, 'Observed Incidence': stratum_metrics[stratum]['observed_incidence'], 'Expected Incidence': expected_incidence})
        comp_df = pd.DataFrame(comparison_data).set_index('stratum')
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, stratum in enumerate(comp_df.index):
            observed = comp_df.loc[stratum, 'Observed Incidence']
            expected = comp_df.loc[stratum, 'Expected Incidence']
            ax.plot([observed, expected], [i, i], color='grey', linestyle='-', linewidth=1, zorder=1)
            ax.plot(observed, i, 'o', color='royalblue', markersize=9, label='Observed Incidence' if i == 0 else "", zorder=2)
            ax.plot(expected, i, 'o', color='darkorange', markersize=9, label='Expected Incidence' if i == 0 else "", zorder=2)
        ax.set_yticks(range(len(comp_df.index)))
        ax.set_yticklabels(comp_df.index)
        ax.set_xlabel("Incidence Rate (events per 1000 person-years)")
        ax.set_ylabel(stratify_col)
        ax.set_title("Observed vs. Expected Incidence Rates\n(Expected rate based on reference stratum's risk associations)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        ax.grid(axis='x', linestyle='--')
        plt.tight_layout()
        plt.show()

    # --- Plotting for Risk Prevalence Differences ---
    if stratify_col != 'dummy_strata':
        print("\n--- Analyzing Differences in Risk Factor Prevalence Across Strata ---")
        significant_diff_risks = []
        for risk in risk_cols:
            is_binary = set(clean_df[risk].dropna().unique()) <= {0, 1}
            if is_binary:
                contingency_table = pd.crosstab(clean_df[stratify_col], clean_df[risk])
                if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2: continue
                chi2, p, _, _ = chi2_contingency(contingency_table)
                if p < 0.05:
                    significant_diff_risks.append(risk)
            else: # Continuous
                samples = [group_df[risk].dropna() for name, group_df in clean_df.groupby(stratify_col)]
                if len(samples) < 2: continue
                f_val, p = f_oneway(*samples)
                if p < 0.05:
                    significant_diff_risks.append(risk)
        
        if not significant_diff_risks:
            print("No significant differences in risk factor prevalence found across strata.")
        else:
            print(f"Found significant prevalence differences for: {significant_diff_risks}")
            n_risks = len(significant_diff_risks)
            n_cols = 2
            n_rows = (n_risks + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
            axes = axes.flatten()

            for i, risk in enumerate(significant_diff_risks):
                sns.barplot(x=stratify_col, y=risk, data=clean_df, ax=axes[i], palette='Set2', order=successful_strata)
                y_label = 'Prevalence' if set(clean_df[risk].dropna().unique()) <= {0, 1} else 'Mean Value'
                axes[i].set_ylabel(y_label)
                axes[i].set_title(f'Prevalence of {risk}')
            
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.show()

    return paf_pivot


# Example Usage:
if __name__ == '__main__':
    # --- Generate synthetic data ---
    n_samples = 5000
    df = pd.DataFrame({
        'Age': np.random.uniform(50, 80, n_samples),
        'Sex': np.random.choice([0, 1], n_samples),
        'APOE4_status': np.random.choice(['Non-carrier', 'Carrier', 'Unknown'], n_samples, p=[0.70, 0.25, 0.05]),
        'risk_smoking': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'risk_negligible': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'time_to_event': np.random.weibull(2, n_samples) * 10,
    })
    
    df['risk_diabetes'] = 0
    df.loc[df['APOE4_status'] == 'Carrier', 'risk_diabetes'] = np.random.choice([0, 1], size=(df['APOE4_status'] == 'Carrier').sum(), p=[0.7, 0.3])
    df.loc[df['APOE4_status'] == 'Non-carrier', 'risk_diabetes'] = np.random.choice([0, 1], size=(df['APOE4_status'] == 'Non-carrier').sum(), p=[0.9, 0.1])
    
    df['risk_hypertension'] = 0
    df.loc[df['APOE4_status'] == 'Carrier', 'risk_hypertension'] = np.random.choice([0, 1], size=(df['APOE4_status'] == 'Carrier').sum(), p=[0.4, 0.6])
    df.loc[df['APOE4_status'] == 'Non-carrier', 'risk_hypertension'] = np.random.choice([0, 1], size=(df['APOE4_status'] == 'Non-carrier').sum(), p=[0.7, 0.3])

    # --- Add a rare risk to test the new robustness check ---
    df['risk_rare'] = 0
    # Assign only 5 cases to the 'Carrier' group
    carrier_indices = df[df['APOE4_status'] == 'Carrier'].index
    rare_indices = np.random.choice(carrier_indices, 5, replace=False)
    df.loc[rare_indices, 'risk_rare'] = 1

    base_risk = -3.5
    risk_score = (base_risk + 
                  (df['APOE4_status'] == 'Carrier') * 1.5 + 
                  df['risk_diabetes'] * 0.4 + 
                  df['risk_hypertension'] * 0.3 +
                  df['risk_smoking'] * 0.5 +
                  df['risk_negligible'] * 0.1 +
                  df['risk_rare'] * 0.8) # Give it a strong effect
    event_prob = 1 / (1 + np.exp(-risk_score))
    df['event_status'] = (np.random.rand(n_samples) < event_prob).astype(int)
    
    df.loc[df['APOE4_status'] == 'Unknown', 'event_status'] = 0
    df.loc[df['APOE4_status'] == 'Unknown', 'time_to_event'] = 0.1

    # --- Run the PAF analysis ---
    risk_factor_columns = ['risk_diabetes', 'risk_hypertension', 'risk_smoking', 'risk_negligible', 'risk_rare']
    
    print("--- Running Analysis Stratified by APOE4 Status ---")
    paf_results_pivot = calculate_and_plot_paf(
        df=df,
        risk_cols=risk_factor_columns,
        time_col='time_to_event',
        event_col='event_status',
        stratify_col='APOE4_status',
        confounders=['Age', 'Sex'],
        negligible_threshold=0.5,
        min_risk_count=10 # Set minimum cases to 10
    )
    
    print("\n--- Corrected & Scaled PAF Contributions (%) ---")
    print(paf_results_pivot)
