import pandas as pd
import numpy as np

def generate_ad_risk_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a minimal set of surrogates for the 14 Lancet modifiable risk factors for Alzheimer's Disease
    derived from UK Biobank data.

    Args:
        ukb_main_df: A pandas DataFrame containing the raw UK Biobank data.

    Returns:
        A pandas DataFrame with one row per participant and columns for each of the 14 risk factor surrogates,
        where a higher value indicates a higher risk for AD.
    """

    # --- 1. Less Education ---
    # Field 6138: Qualifications
    # Recoding: Higher values indicate lower educational attainment (higher risk)
    education_mapping = {
        1: 6,  # College or University degree
        2: 5,  # A levels/AS levels or equivalent
        3: 4,  # O levels/GCSEs or equivalent
        4: 3,  # CSEs or equivalent
        5: 2,  # NVQ or HND or HNC or equivalent
        6: 1,  # Other professional qualifications eg: nursing, teaching
        -7: np.nan, # None of the above
        -3: np.nan  # Prefer not to answer
    }
    ukb_main_df['risk_less_education'] = ukb_main_df['6138-0.0'].map(education_mapping)
    # Invert the scale so higher is higher risk
    ukb_main_df['risk_less_education'] = 7 - ukb_main_df['risk_less_education']


    # --- 2. Hypertension ---
    # Field 20002: Self-reported non-cancer illness
    # Coding 1071 corresponds to hypertension
    ukb_main_df['risk_hypertension'] = ukb_main_df['20002-0.0'].apply(lambda x: 1 if x == 1071 else 0)


    # --- 3. Hearing Impairment ---
    # Field 2247: Hearing difficulty/problems
    # Recoding: 'Yes' (1) indicates higher risk.
    ukb_main_df['risk_hearing_impairment'] = ukb_main_df['2247-0.0'].apply(lambda x: 1 if x == 1 else 0)


    # --- 4. Smoking ---
    # Field 20116: Smoking status
    # Recoding: Current smokers (2) and previous smokers (1) have higher risk.
    smoking_mapping = {
        0: 0,  # Never
        1: 1,  # Previous
        2: 2,  # Current
        -3: np.nan # Prefer not to answer
    }
    ukb_main_df['risk_smoking'] = ukb_main_df['20116-0.0'].map(smoking_mapping)


    # --- 5. Obesity ---
    # Field 21001: Body mass index (BMI)
    # Higher BMI is a direct risk factor.
    ukb_main_df['risk_obesity'] = ukb_main_df['21001-0.0']


    # --- 6. Depression ---
    # Field 20126: Self-reported depression
    ukb_main_df['risk_depression'] = ukb_main_df['20126-0.0'].apply(lambda x: 1 if x == 1 else 0)


    # --- 7. Physical Inactivity ---
    # Field 22040: International Physical Activity Questionnaire (IPAQ) activity level
    # Recoding: Inverting the scale as higher IPAQ score is protective.
    physical_activity_mapping = {
        3: 1, # High
        2: 2, # Moderate
        1: 3, # Low
    }
    ukb_main_df['risk_physical_inactivity'] = ukb_main_df['22040-0.0'].map(physical_activity_mapping)


    # --- 8. Diabetes ---
    # Field 2443: Doctor diagnosed diabetes
    ukb_main_df['risk_diabetes'] = ukb_main_df['2443-0.0'].apply(lambda x: 1 if x == 1 else 0)


    # --- 9. Social Isolation ---
    # Field 1031: Frequency of friend/family visits
    # Recoding: Less frequent visits indicate higher isolation (higher risk).
    social_isolation_mapping = {
        1: 5, # Daily or almost daily
        2: 4, # 2-4 times a week
        3: 3, # Once a week
        4: 2, # 1-3 times a month
        5: 1, # Never or almost never
        -7: np.nan, # None of the above
        -3: np.nan  # Prefer not to answer
    }
    ukb_main_df['risk_social_isolation'] = ukb_main_df['1031-0.0'].map(social_isolation_mapping)


    # --- 10. Excessive Alcohol Consumption ---
    # Field 1558: Alcohol intake frequency
    # Recoding: Higher frequency indicates higher risk.
    alcohol_mapping = {
        1: 6,  # Daily or almost daily
        2: 5,  # Three or four times a week
        3: 4,  # Once or twice a week
        4: 3,  # One to three times a month
        5: 2,  # Special occasions only
        6: 1,  # Never
        -3: np.nan  # Prefer not to answer
    }
    ukb_main_df['risk_alcohol'] = ukb_main_df['1558-0.0'].map(alcohol_mapping)


    # --- 11. Traumatic Brain Injury ---
    # Field 20002: Self-reported non-cancer illness
    # Coding 1081 corresponds to head injury
    ukb_main_df['risk_tbi'] = ukb_main_df['20002-0.0'].apply(lambda x: 1 if x == 1081 else 0)


    # --- 12. Air Pollution ---
    # Field 24003: Nitrogen dioxide air pollution; 2010
    # Higher value is a direct risk.
    ukb_main_df['risk_air_pollution'] = ukb_main_df['24003-0.0']


    # --- 13. High Cholesterol ---
    # Field 20002: Self-reported non-cancer illness
    # Coding 1473 corresponds to high cholesterol
    ukb_main_df['risk_high_cholesterol'] = ukb_main_df['20002-0.0'].apply(lambda x: 1 if x == 1473 else 0)


    # --- 14. Uncorrected Vision Loss ---
    # Field 6142: Wears glasses or contact lenses
    # Assuming those who need glasses but don't wear them are at higher risk.
    # This is a simplification; a more detailed assessment would be needed.
    # Recoding: 1 if 'Yes', 0 if 'No'
    ukb_main_df['risk_vision_loss'] = ukb_main_df['6142-0.0'].apply(lambda x: 1 if x == 1 else 0)


    # --- Consolidate Risk Factors ---
    risk_factors_df = ukb_main_df[[
        'risk_less_education',
        'risk_hypertension',
        'risk_hearing_impairment',
        'risk_smoking',
        'risk_obesity',
        'risk_depression',
        'risk_physical_inactivity',
        'risk_diabetes',
        'risk_social_isolation',
        'risk_alcohol',
        'risk_tbi',
        'risk_air_pollution',
        'risk_high_cholesterol',
        'risk_vision_loss'
    ]].copy()

    return risk_factors_df

if __name__ == '__main__':
    # This is a placeholder for loading the actual UK Biobank data.
    # Researchers with access should replace this with their data loading mechanism.
    # For demonstration, we create a dummy dataframe.
    dummy_data = {
        'eid': range(100),
        '6138-0.0': np.random.randint(1, 7, 100),
        '20002-0.0': np.random.choice([1071, 1081, 1473, 9999], 100),
        '2247-0.0': np.random.randint(0, 2, 100),
        '20116-0.0': np.random.randint(0, 3, 100),
        '21001-0.0': np.random.uniform(18, 40, 100),
        '20126-0.0': np.random.randint(0, 2, 100),
        '22040-0.0': np.random.randint(1, 4, 100),
        '2443-0.0': np.random.randint(0, 2, 100),
        '1031-0.0': np.random.randint(1, 6, 100),
        '1558-0.0': np.random.randint(1, 7, 100),
        '24003-0.0': np.random.uniform(10, 50, 100),
        '6142-0.0': np.random.randint(0, 2, 100)
    }
    ukb_dummy_df = pd.DataFrame(dummy_data)

    # Generate the risk surrogates
    ad_risk_surrogates = generate_ad_risk_surrogates(ukb_dummy_df)

    # Display the first few rows of the generated surrogates
    print(ad_risk_surrogates.head())
    
    
    
def generate_dietary_risk_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a minimal set of surrogates for dietary components associated with dementia risk,
    derived from UK Biobank food frequency questionnaire (FFQ) data. This function assumes
    the input DataFrame contains the necessary UK Biobank fields.

    The selected dietary risk components are:
    1. High Saturated Fat Intake
    2. High Simple Carbohydrate (Sugar) Intake
    3. Low Dietary Fiber Intake
    4. Low Omega-3 Fatty Acid Intake

    Args:
        ukb_main_df: A pandas DataFrame containing the raw UK Biobank data, indexed by 'eid'.

    Returns:
        A pandas DataFrame with one row per participant and columns for each of the dietary risk
        factor surrogates, where a higher value indicates a higher dementia risk.
        The DataFrame is indexed by 'eid'.
    """
    # Create a new DataFrame to hold the surrogates, using the index from the input
    surrogates = pd.DataFrame(index=ukb_main_df.index)

    # --- 1. High Saturated Fat Intake (Risk Factor) ---
    # We create a composite score from meat, milk, and spread types.
    
    # Processed meat intake (Field 1349)
    proc_meat_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, -1: 0, -3: 0}
    proc_meat_score = ukb_main_df['1349-0.0'].map(proc_meat_map).fillna(0)
    
    # Unprocessed red meat intake (Fields 1369, 1379)
    red_meat_map = {1: 1, 2: 2, 3: 3, 4: 4, -1: 0, -3: 0}
    beef_score = ukb_main_df['1369-0.0'].map(red_meat_map).fillna(0)
    lamb_score = ukb_main_df['1379-0.0'].map(red_meat_map).fillna(0)
    
    # Milk type (Field 1418): Whole milk has more saturated fat.
    milk_map = {1: 2, 2: 1, 3: 0, 4: 0, 5: 1, -1: 0, -3: 0} # Higher score for higher fat
    milk_score = ukb_main_df['1418-0.0'].map(milk_map).fillna(0)

    # Spread type (Field 1428): Butter has more saturated fat.
    spread_map = {1: 2, 2: 1, 3: 1, 4: 0, -1: 0, -3: 0} # Higher score for butter
    spread_score = ukb_main_df['1428-0.0'].map(spread_map).fillna(0)

    # Combine into a single saturated fat risk score
    surrogates['risk_saturated_fat'] = proc_meat_score + beef_score + lamb_score + milk_score + spread_score

    # --- 2. High Simple Carbohydrate (Sugar) Intake (Risk Factor) ---
    # We use cereal type as a proxy for intake of sugary foods.
    # Field 1468: Cereal type.
    cereal_map = {
        1: 0, # Porridge or oats (Low Sugar)
        2: 0, # Bran cereal
        3: 1, # Weetabix, Shredded Wheat
        4: 2, # Cornflakes, Rice Krispies
        5: 3, # Frosted Flakes, Sugar Puffs (High Sugar)
        6: 3, # Other
        -1: 0,
        -3: 0
    }
    surrogates['risk_simple_carbs'] = ukb_main_df['1468-0.0'].map(cereal_map).fillna(0)


    # --- 3. Low Dietary Fiber Intake (Protective Factor, Inverted to Risk) ---
    # We create a composite score based on fruit, vegetables, and bread type.
    # Lower consumption of high-fiber foods results in a higher risk score.
    portions_map = {-1: 0, -3: 0}
    fruit_portions = ukb_main_df['1309-0.0'].map(portions_map).fillna(ukb_main_df['1309-0.0'])
    cooked_veg_portions = ukb_main_df['1289-0.0'].map(portions_map).fillna(ukb_main_df['1289-0.0'])
    raw_veg_portions = ukb_main_df['1299-0.0'].map(portions_map).fillna(ukb_main_df['1299-0.0'])
    
    # Bread type (Field 1448): White bread is lower in fiber.
    bread_risk_map = {
        1: -1, # Wholemeal or wholegrain (Protective)
        2: -1, # Brown (Protective)
        3: 1,  # White (Risk)
        4: 0,  # Other
        -1: 0,
        -3: 0
    }
    bread_fiber_score = ukb_main_df['1448-0.0'].map(bread_risk_map).fillna(0)

    # Invert the F&V intake to be a risk score, and add the bread risk.
    total_fv_intake = fruit_portions + cooked_veg_portions + raw_veg_portions
    low_fv_risk = (10 - total_fv_intake).clip(lower=0)
    surrogates['risk_low_fiber'] = low_fv_risk + bread_fiber_score
    surrogates['risk_low_fiber'] = surrogates['risk_low_fiber'].clip(lower=0)


    # --- 4. Low Omega-3 Fatty Acid Intake (Protective Factor, Inverted to Risk) ---
    # Field 1329: Oily fish intake is the best proxy for Omega-3.
    # Higher frequency is protective, so we invert the score.
    omega3_map = {
        1: 5,  # Never (Highest Risk)
        2: 4,  # Less than once a week
        3: 3,  # Once a week
        4: 2,  # 2-4 times a week
        5: 1,  # 5-6 times a week
        6: 0,  # Once or more daily (Lowest Risk)
        -1: np.nan, # Do not know
        -3: np.nan  # Prefer not to answer
    }
    surrogates['risk_low_omega3'] = ukb_main_df['1329-0.0'].map(omega3_map)

    return surrogates


def generate_comorbidity_risk_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a minimal set of surrogates for key comorbidities associated with dementia risk,
    derived from UK Biobank data. This function assumes the input DataFrame contains the necessary
    UK Biobank fields.

    The selected comorbidity risk factors are:
    1.  History of Stroke
    2.  History of Ischaemic Heart Disease (Angina/Heart Attack)
    3.  Atrial Fibrillation
    4.  Diabetes
    5.  Hypertension
    6.  Depression
    7.  Chronic Kidney Disease

    Args:
        ukb_main_df: A pandas DataFrame containing the raw UK Biobank data, indexed by 'eid'.

    Returns:
        A pandas DataFrame with one row per participant and columns for each of the comorbidity risk
        factor surrogates, where a value of 1 indicates the presence of the condition.
        The DataFrame is indexed by 'eid'.
    """
    # Create a new DataFrame to hold the surrogates, using the index from the input
    surrogates = pd.DataFrame(index=ukb_main_df.index)

    # Use a dictionary to map self-reported vascular/heart problems (Field 6150)
    # 1: Heart attack, 2: Angina, 3: Stroke, 4: High blood pressure
    vascular_map = ukb_main_df.filter(regex='^6150-').stack().reset_index()
    vascular_map.columns = ['eid', 'field', 'condition_code']
    vascular_map = vascular_map.drop('field', axis=1).drop_duplicates()
    
    # --- 1. History of Stroke ---
    stroke_eids = vascular_map[vascular_map['condition_code'] == 3]['eid']
    surrogates['risk_stroke'] = surrogates.index.isin(stroke_eids).astype(int)

    # --- 2. History of Ischaemic Heart Disease (IHD) ---
    # Combines Angina (2) and Heart Attack (1)
    ihd_eids = vascular_map[vascular_map['condition_code'].isin([1, 2])]['eid']
    surrogates['risk_ihd'] = surrogates.index.isin(ihd_eids).astype(int)

    # --- 3. Hypertension ---
    hypertension_eids = vascular_map[vascular_map['condition_code'] == 4]['eid']
    surrogates['risk_hypertension'] = surrogates.index.isin(hypertension_eids).astype(int)

    # Helper function to check for a code in the non-cancer illness array fields (e.g., 20002)
    def has_illness_code(df, code):
        illness_cols = df.filter(regex=f'^20002-').columns
        if illness_cols.empty:
            return pd.Series(0, index=df.index)
        return df[illness_cols].apply(lambda row: (row == code).any(), axis=1).astype(int)

    # --- 4. Atrial Fibrillation ---
    # Field 20002: Non-cancer illness code, 1491 for Atrial Fibrillation/Flutter
    surrogates['risk_atrial_fibrillation'] = has_illness_code(ukb_main_df, 1491)

    # --- 5. Chronic Kidney Disease (CKD) ---
    # Field 20002: Non-cancer illness code, 1192 for Kidney Failure/Dialysis
    surrogates['risk_ckd'] = has_illness_code(ukb_main_df, 1192)

    # --- 6. Diabetes ---
    # Field 2443: Doctor diagnosed diabetes. 'Yes' (1) indicates higher risk.
    # Handle special values -1 (Do not know) and -3 (Prefer not to answer)
    surrogates['risk_diabetes'] = ukb_main_df['2443-0.0'].apply(lambda x: 1 if x == 1 else 0)

    # --- 7. Depression ---
    # Field 20126: Ever had prolonged feelings of sadness or depression. 'Yes' (1) indicates risk.
    surrogates['risk_depression'] = ukb_main_df['20126-0.0'].apply(lambda x: 1 if x == 1 else 0)

    return surrogates




def gen_lipid_risk_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates surrogate biomarkers for Alzheimer's-associated lipid risk using
    UK Biobank data from NMR metabolomics and clinical chemistry.

    This function calculates key ratios and a composite score related to
    atherogenic dyslipidemia, a condition linked to Alzheimer's disease risk.
    It also calculates the number of APOE4 alleles, a major genetic risk factor.

    The function operates on a pandas DataFrame containing UK Biobank data
    and adds the following new columns:
    - 'lipid_apoe4_allele_count': Number of APOE e4 alleles (0, 1, or 2).
    - 'lipid_risk_apob_apoa1_ratio': Ratio of Apolipoprotein B to Apolipoprotein A1.
    - 'lipid_risk_tg_hdl_ratio': Ratio of Triglycerides to HDL Cholesterol.
    - 'lipid_atherogenic_score_z': A composite Z-score of multiple pro-atherogenic
                                     NMR lipid measures.

    Args:
        df (pd.DataFrame): A DataFrame containing decoded UK Biobank data.
                           It must contain the relevant columns (fields) for
                           lipids, apolipoproteins, and genetics.

    Returns:
        pd.DataFrame: The input DataFrame with added columns for the lipid
                      risk surrogates.
    """
    print("Generating lipid risk surrogates...")

    # --- 1. Define UK Biobank Field Mappings ---
    # Maps readable names to their corresponding UKB field codes (instance 0, array 0)
    field_map = {
        # Clinical Chemistry (Biochemistry)
        'hdl_cholesterol': 'f.30760.0.0',
        'triglycerides': 'f.30870.0.0',
        'apolipoprotein_a': 'f.30630.0.0',
        'apolipoprotein_b': 'f.30640.0.0',

        # NMR Metabolomics - Lipoprotein Cholesterol & Lipids
        'nmr_remnant_c': 'f.23476.0.0',  # Remnant cholesterol (VLDL+IDL)
        'nmr_ldl_c': 'f.23474.0.0',      # Cholesterol in LDL
        'nmr_hdl_c': 'f.23472.0.0',      # Cholesterol in HDL
        'nmr_total_tg_in_vldl': 'f.23485.0.0', # Total triglycerides in VLDL
        'nmr_sphingomyelins': 'f.23457.0.0', # Sphingomyelins
        'nmr_dha': 'f.23446.0.0',           # Docosahexaenoic acid (DHA)

        # Genetics
        'apoe_genotype': 'f.22617.0.0' # APOE e2/e3/e4 status
    }

    # Create a copy to avoid SettingWithCopyWarning
    df_out = df.copy()
    
    # Keep track of generated columns for a final summary
    generated_cols = []

    # --- 2. Process APOE Genotype ---
    apoe_field = field_map['apoe_genotype']
    if apoe_field in df_out.columns:
        # UKB coding: e2/e2=1, e2/e3=2, e3/e3=3, e2/e4=4, e3/e4=5, e4/e4=6
        apoe4_map = {
            1: 0, 2: 0, 3: 0, # No e4 allele
            4: 1, 5: 1,       # One e4 allele
            6: 2              # Two e4 alleles
        }
        df_out['lipid_apoe4_allele_count'] = df_out[apoe_field].map(apoe4_map)
        generated_cols.append('lipid_apoe4_allele_count')
    else:
        warnings.warn(f"Column {apoe_field} (APOE genotype) not found. Skipping APOE4 count.")


    # --- 3. Calculate Clinical Ratios ---
    # Ratio of Apolipoprotein B to Apolipoprotein A1
    try:
        apo_b = df_out[field_map['apolipoprotein_b']]
        apo_a = df_out[field_map['apolipoprotein_a']]
        df_out['lipid_risk_apob_apoa1_ratio'] = apo_b / apo_a
        generated_cols.append('lipid_risk_apob_apoa1_ratio')
    except KeyError:
        warnings.warn("Apolipoprotein columns not found. Skipping ApoB/ApoA1 ratio.")

    # Ratio of Triglycerides to HDL Cholesterol
    try:
        tg = df_out[field_map['triglycerides']]
        hdl = df_out[field_map['hdl_cholesterol']]
        df_out['lipid_risk_tg_hdl_ratio'] = tg / hdl
        generated_cols.append('lipid_risk_tg_hdl_ratio')
    except KeyError:
        warnings.warn("Triglyceride or HDL columns not found. Skipping TG/HDL ratio.")

    # --- 4. Derive Composite Atherogenic Lipid Score ---
    # This score combines multiple NMR measures associated with atherogenic
    # dyslipidemia. We standardize them (Z-score) and sum them up.
    atherogenic_components = [
        'nmr_remnant_c',
        'nmr_ldl_c',
        'nmr_total_tg_in_vldl',
        'nmr_sphingomyelins'
    ]
    
    # DHA is considered protective
    protective_components = [
        'nmr_dha'
    ]

    df_out['lipid_atherogenic_score_z'] = 0
    score_components_found = 0

    # Helper function to safely calculate Z-score
    def safe_zscore(series):
        if series.notna().sum() > 0:
            return (series - series.mean()) / series.std()
        return series

    # Sum Z-scores of atherogenic components
    for comp in atherogenic_components:
        field = field_map.get(comp)
        if field and field in df_out.columns:
            df_out['lipid_atherogenic_score_z'] += safe_zscore(df_out[field].astype(float))
            score_components_found += 1
        else:
            warnings.warn(f"NMR component {comp} ({field}) not found. Skipping from score.")
            
    # Subtract Z-scores of protective components
    for comp in protective_components:
        field = field_map.get(comp)
        if field and field in df_out.columns:
            df_out['lipid_atherogenic_score_z'] -= safe_zscore(df_out[field].astype(float))
            score_components_found += 1
        else:
            warnings.warn(f"NMR component {comp} ({field}) not found. Skipping from score.")

    if score_components_found > 0:
        # Normalize the final score by the number of components used
        df_out['lipid_atherogenic_score_z'] /= score_components_found
        generated_cols.append('lipid_atherogenic_score_z')
    else:
        # Drop the column if no components were found
        df_out = df_out.drop(columns=['lipid_atherogenic_score_z'])
        warnings.warn("No NMR components found. Could not generate Atherogenic Lipid Score.")


    print("\n--- Lipid Risk Surrogate Generation Complete ---")
    if generated_cols:
        print("Added the following columns:")
        for col in generated_cols:
            print(f"  - {col}")
    else:
        print("No new columns were generated. Please check input data for required fields.")
    print("----------------------------------------------")

    return df_out