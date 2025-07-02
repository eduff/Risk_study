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
        2: 4,  # A levels/AS levels or equivalent
        3: 3,  # O levels/GCSEs or equivalent
        4: 3,  # CSEs or equivalent
        5: 2,  # NVQ or HND or HNC or equivalent
        6: 5,  # Other professional qualifications eg: nursing, teaching
        -7:0,  # None of the above
        -3: np.nan  # Prefer not to answer
    }
    ukb_main_df['risk_less_education'] = ukb_main_df['6138-0.0'].map(education_mapping)
    # Invert the scale so higher is higher risk
    ukb_main_df['risk_less_education'] = 7 - ukb_main_df['risk_less_education']


    # --- 2. Hypertension ---
    # Field 20002: Self-reported non-cancer illness
    # Coding 1071 corresponds to hypertension
    ukb_main_df['risk_hypertension'] = ukb_main_df['20002-0.0'].apply(lambda x: 1 if x == 1071 else 0)
    ukb_main_df['systolic']=ukb_main_df['4080-0.0']
    ukb_main_df['diastolic']=ukb_main_df['4079-0.0']
    
    ukb_main_df['risk_hypertension'] = (ukb_main_df['4080-0.0']>130) & (ukb_main_df['4079-0.0']>80)
    

    # --- 3. Hearing Impairment ---
    # Field 2247: Hearing difficulty/problems
    # Recoding: 'Yes' (1) indicates higher risk.
    ukb_main_df['risk_hearing_impairment'] = ukb_main_df['2247-0.0'].apply(lambda x: 1 if ((x == 1) | (x == 99))else 0)


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
    BMI=ukb_main_df['21001-0.0'] 
    ukb_main_df['risk_obesity'] = BMI

    #height = ukb_main_df['12144-0.0']  # Field 4076: Height
    waist = ukb_main_df['48-0.0']    # Field 48: Waist circumference
   # BMI
    weight = ukb_main_df['21002-0.0'] # Weight
    height = (weight / BMI ) ** 0.5  # Calculate height from weight and BMI
    ratio = waist/(height*100) 
    sex_col = ukb_main_df['31-0.0']  # Field 31
    ukb_main_df['risk_obesity']= ratio.apply(lambda x: 1 if x > 0.85 else 0) # Waist to height ratio > 0.85 indicates higher risk

    ukb_main_df['risk_obesity']= ratio

    # --- 6. Depression ---
    # Field 21063: Self-reported depression
    ukb_main_df['risk_depression'] = ukb_main_df['21063-0.0'].apply(lambda x: 1 if x == 1 else 0)


    # --- 7. Physical Inactivity ---
    # Field 22032: International Physical Activity Questionnaire (IPAQ) activity level
    # 
    physical_activity_mapping = {
        0: 3, # High
        1: 2, # Moderate
        2: 1, # Low
    }
    ukb_main_df['risk_physical_inactivity'] = ukb_main_df['22032-0.0'].map(physical_activity_mapping)


    # --- 8. Diabetes ---
    # Field 2443: Doctor diagnosed diabetes
    ukb_main_df['risk_diabetes'] = ukb_main_df['2443-0.0'].apply(lambda x: 1 if x == 1 else 0)


    # --- 9. Social Isolation ---
    # Field 1031: Frequency of friend/family visits
    # Recoding: Less frequent visits indicate higher isolation (higher risk).
    social_isolation_mapping = {
        1: 1, # Daily or almost daily
        2: 2, # 2-4 times a week
        3: 3, # Once a week
        4: 4, # 1-3 times a month
        5: 5, # Never or almost never
        6: 6, # Never or almost never
        7: 7, # Never or almost never
        -1: np.nan, # None of the above
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
        4: 0,  # One to three times a month
        5: 0,  # Special occasions only
        6: 0,  # Never
        -3: np.nan  # Prefer not to answer
    }
    ukb_main_df['risk_alcohol'] = ukb_main_df['1558-0.0'].map(alcohol_mapping)
    # # Fields 20416,29093:Frequency of consuming six or more units of alcohol

    # # Recoding: Higher frequency indicates higher risk.
    # alcohol_mapping = {
    #     5: 5,  # Daily or almost daily
    #     3: 3,  # Three or four times a week
    #     2: 2,  # Monthly or less
    #     4: 4,  # One to three times a month
    #     1:1, # Never
    #     0: 0,  # Never
    #     -3: np.nan,  # Never# Prefer not to answer
    #     -818: np.nan, # Never# Prefer not to answer
    # } 
    # ukb_main_df['risk_alcohol'] =  ukb_main_df['20416-0.0'].map(alcohol_mapping)
   


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
    ukb_main_df['risk_high_cholesterol'] = (ukb_main_df['30690-0.0']>6.5).astype(int)




    # --- 14. Uncorrected Vision Loss ---
    # Field 6142: Wears glasses or contact lenses
    # Assuming those who need glasses but don't wear them are at higher risk.
    # This is a simplification; a more detailed assessment would be needed.
    # Recoding: 1 if 'Yes', 0 if 'No'
    ukb_main_df['risk_vision_loss'] = ukb_main_df['6148-0.0'].apply(lambda x: 1 if (x>0) else 0)


    # --- Consolidate Risk Factors ---
    risk_factors_df = ukb_main_df[[
        'risk_less_education',
        'risk_hypertension',
        'systolic',
        'diastolic',
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


def gen_comorbidity_risk_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
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

import pandas as pd
import numpy as np

def gen_metabolomics_risk_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates surrogate risk markers using the specific NMR biomarker data
    fields identified from the provided field summary file.

    Args:
        ukb_main_df: A pandas DataFrame containing UK Biobank data.

    Returns:
        A new DataFrame containing the calculated surrogate risk scores from NMR data.
    """
    surrogates = pd.DataFrame(index=ukb_main_df.index)

    # --- Direct Risk Factors (Higher value = Higher risk) ---
    # Using verified Nightingale NMR metabolomics fields from the provided file.

    # Field 23439: Apolipoprotein B
    surrogates['risk_nmr_apob'] = ukb_main_df.get('23439-0.0')

    # Field 23407: Total Triglycerides
    surrogates['risk_nmr_triglycerides'] = ukb_main_df.get('23407-0.0')
    
    # Field 23405: LDL Cholesterol
    surrogates['risk_nmr_ldl_cholesterol'] = ukb_main_df.get('23405-0.0')

    # Field 23480: Glycoprotein Acetyls (GlycA)
    surrogates['risk_nmr_glyca_inflammation'] = ukb_main_df.get('23480-0.0')

    # Field 23470: Glucose
    surrogates['risk_nmr_glucose'] = ukb_main_df.get('23470-0.0')

    # --- Branched-Chain Amino Acids (BCAAs) ---
    bcaa_fields = {
        'leucine': '23466-0.0',    # Field 23466: Leucine
        'isoleucine': '23465-0.0', # Field 23465: Isoleucine
        'valine': '23467-0.0'      # Field 23467: Valine
    }
    bcaa_scores = []
    for field in bcaa_fields.values():
        if field in ukb_main_df.columns:
            # Normalize each BCAA before summing to give equal weight
            series = pd.to_numeric(ukb_main_df[field], errors='coerce')
            normalized_series = (series - series.mean()) / series.std()
            bcaa_scores.append(normalized_series)

    if bcaa_scores:
        # Sum the normalized scores to create a composite risk marker
        surrogates['risk_nmr_bcaa_composite'] = pd.concat(bcaa_scores, axis=1).sum(axis=1, skipna=False)


    # --- Protective Factors (Inverted to create risk score) ---
    # Lower levels are worse, so we multiply by -1 to make a higher value indicate higher risk.

    # Field 23406: HDL Cholesterol
    if '23406-0.0' in ukb_main_df:
        surrogates['risk_nmr_inverted_hdl'] = ukb_main_df['23406-0.0'] * -1

    # Field 23440: Apolipoprotein A1
    if '23440-0.0' in ukb_main_df:
        surrogates['risk_nmr_inverted_apoa1'] = ukb_main_df['23440-0.0'] * -1

    return surrogates

def gen_immunological_risk_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a set of surrogates for key immunological and inflammatory biomarkers associated
    with dementia risk, derived from UK Biobank blood assay and NMR data. This function assumes
    the input DataFrame contains the necessary UK Biobank fields.

    The selected immunological risk factors include:
    1.  C-reactive protein (CRP)
    2.  White blood cell (leukocyte) count
    3.  Neutrophil count
    4.  Monocyte count
    5.  Neutrophil-to-Lymphocyte Ratio (NLR)
    6.  Glycoprotein Acetyls (GlycA - inflammation marker from NMR)

    Args:
        ukb_main_df: A pandas DataFrame containing the raw UK Biobank data, indexed by 'eid'.

    Returns:
        A pandas DataFrame with one row per participant and columns for each of the immunological
        risk factor surrogates. For all columns, a higher value indicates higher dementia risk.
        The DataFrame is indexed by 'eid'.
    """
    # Create a new DataFrame to hold the surrogates, using the index from the input
    surrogates = pd.DataFrame(index=ukb_main_df.index)

    # For all these markers, a higher value is already associated with higher inflammation and risk.

    # --- 1. C-reactive protein (CRP) ---
    # Field 30710: A primary marker of systemic inflammation.
    surrogates['risk_crp'] = ukb_main_df.get('30710-0.0')

    # --- 2. White blood cell (leukocyte) count ---
    # Field 30000: A general marker of immune system activation.
    surrogates['risk_wbc_count'] = ukb_main_df.get('30000-0.0')

    # --- 3. Neutrophil count ---
    # Field 30140: A major component of the innate immune system.
    neutrophil_count = ukb_main_df.get('30140-0.0')
    surrogates['risk_neutrophil_count'] = neutrophil_count

    # --- 4. Monocyte count ---
    # Field 30130: Involved in inflammatory responses.
    surrogates['risk_monocyte_count'] = ukb_main_df.get('30130-0.0')
    
    # --- 5. Neutrophil-to-Lymphocyte Ratio (NLR) ---
    # A composite marker of systemic inflammation. Higher NLR indicates greater inflammation.
    # Calculated from Neutrophil count (30140) and Lymphocyte count (30120).
    lymphocyte_count = ukb_main_df.get('30120-0.0')
    
    # Calculate NLR, handling potential division by zero or NaN values gracefully.
    if neutrophil_count is not None and lymphocyte_count is not None:
        # Replace zeros in denominator with NaN to avoid division errors and subsequent NaNs
        lymphocytes_no_zero = lymphocyte_count.replace(0, np.nan)
        surrogates['risk_nlr'] = neutrophil_count / lymphocytes_no_zero

    # --- 6. Glycoprotein acetyls (GlycA) ---
    # Field 23454: A marker of chronic inflammation from the NMR metabolomics panel.
    surrogates['risk_glyca_inflammation'] = ukb_main_df.get('23454-0.0')
    
    return surrogates


import pandas as pd
import numpy as np
import warnings

def gen_serology_risk_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates surrogate biomarkers from the UK Biobank virus serology substudy.

    This function processes antibody measurement data for SARS-CoV-2 and a panel of
    other common viruses to create high-level summary variables. These surrogates
    can be used to model the impact of past infections on health outcomes.

    The function operates on a pandas DataFrame containing UK Biobank data
    and adds the following new columns:
    - 'serology_covid19_status': Categorical status (Seronegative, Infection-induced,
      Vaccine-induced) based on Spike and Nucleocapsid antibody results.
    - 'serology_pathogen_burden_score': A count of positive results from a panel
      of 8 common viruses, representing total pathogen exposure.
    - 'serology_herpesvirus_score_z': A continuous, standardized score of the
      antibody response magnitude to four common herpesviruses (EBV, CMV, HSV-1, VZV).

    Args:
        df (pd.DataFrame): A DataFrame containing decoded UK Biobank data.
                           It must contain the relevant columns (fields) from
                           the serology substudy (Categories 100300, 2120).

    Returns:
        pd.DataFrame: The input DataFrame with added columns for the
                      serology risk surrogates.
    """
    print("Generating serology risk surrogates...")

    # --- 1. Define UK Biobank Field Mappings ---
    # Maps readable names to their UKB field codes (instance 0, array 0)
    # These fields are primarily from Category 100300 and Category 2120
    field_map = {
        # SARS-CoV-2 (COVID-19) Assays
        'covid_s_antibody': '40118-0.0', # Ab-S (Spike protein, chemiluminescence)
        'covid_n_antibody': '40117-0.0', # Ab-N (Nucleocapsid protein, chemiluminescence)

        # Common Virus Panel (IgG measurements, chemiluminescence)
        'ebv': '30214-0.0',      # Epstein-Barr Virus
        'cmv': '30208-0.0',      # Cytomegalovirus
        'hsv1': '30220-0.0',     # Herpes Simplex Virus 1
        'vzv': '30244-0.0',      # Varicella-Zoster Virus (Chickenpox/Shingles)
        'hpv16': '30256-0.0',    # Human Papillomavirus 16 (high-risk)
        'hhv6': '30226-0.0',     # Human Herpesvirus 6
        'jcv': '30232-0.0',      # John Cunningham Virus
        'bkv': '30202-0.0'       # BK Virus
    }

    df_out = ukb_main_df.copy()
    generated_cols = []

    # --- 2. Create COVID-19 Status Surrogate ---
    s_field = field_map['covid_s_antibody']
    n_field = field_map['covid_n_antibody']

    if s_field in df_out.columns and n_field in df_out.columns:
        # Official cutoffs from UKB for the Roche Elecsys assays:
        # Positive if value >= 0.8 (Spike) or >= 1.0 (Nucleocapsid)
        s_positive = df_out[s_field] >= 0.8
        n_positive = df_out[n_field] >= 1.0

        # Logic to determine status:
        # - Infection: N-positive (as N is not induced by vaccines)
        # - Vaccine-induced: S-positive BUT N-negative
        # - Seronegative: Both S and N are negative
        conditions = [
            n_positive,
            (s_positive) & (~n_positive),
            (~s_positive) & (~n_positive)
        ]
        choices = [
            'Infection-induced',
            'Vaccine-induced',
            'Seronegative'
        ]
        df_out['serology_covid19_status'] = np.select(conditions, choices, default='Ambiguous')
        generated_cols.append('serology_covid19_status')
    else:
        warnings.warn("COVID-19 antibody columns not found. Skipping COVID-19 status surrogate.")


    # --- 3. Create Pathogen Burden Score ---
    pathogen_panel = ['ebv', 'cmv', 'hsv1', 'vzv', 'hpv16', 'hhv6', 'jcv', 'bkv']
    burden_cols_found = []

    # Binarize each virus based on a seropositivity cutoff.
    # NOTE: These cutoffs are illustrative. For research, use official assay documentation.
    # A common approach is to use a cutoff provided by the manufacturer or literature.
    # Here we assume a simple cutoff of >= 1.0 arbitrary units for positivity.
    cutoff = 1.0
    for virus in pathogen_panel:
        field = field_map.get(virus)
        if field and field in df_out.columns:
            binary_col_name = f'is_pos_{virus}'
            df_out[binary_col_name] = (df_out[field] >= cutoff).astype(int)
            burden_cols_found.append(binary_col_name)

    if burden_cols_found:
        df_out['serology_pathogen_burden_score'] = df_out[burden_cols_found].sum(axis=1)
        generated_cols.append('serology_pathogen_burden_score')
        # Clean up intermediate binary columns
        df_out.drop(columns=burden_cols_found, inplace=True)
    else:
        warnings.warn("No common virus panel columns found. Skipping pathogen burden score.")


    # --- 4. Create Herpesvirus Latency Score ---
    herpes_panel = ['ebv', 'cmv', 'hsv1', 'vzv']
    herpes_z_scores = []

    def safe_zscore(series):
        # Calculates Z-score, handling series with zero standard deviation
        if series.notna().sum() > 0 and series.std() > 0:
            return (series - series.mean()) / series.std()
        return series # Return original series if it can't be standardized

    for virus in herpes_panel:
        field = field_map.get(virus)
        if field and field in df_out.columns:
            df_out[f'{field}_z'] = safe_zscore(df_out[field].astype(float))
            herpes_z_scores.append(f'{field}_z')

    if herpes_z_scores:
        # The score is the sum of the standardized antibody levels
        df_out['serology_herpesvirus_score_z'] = df_out[herpes_z_scores].sum(axis=1)
        generated_cols.append('serology_herpesvirus_score_z')
        # Clean up intermediate z-score columns
        df_out.drop(columns=herpes_z_scores, inplace=True)
    else:
        warnings.warn("No herpesvirus panel columns found. Skipping herpesvirus score.")


    print("\n--- Serology Risk Surrogate Generation Complete ---")
    if generated_cols:
        print("Added the following columns:")
        for col in generated_cols:
            print(f"  - {col}")
    else:
        print("No new columns were generated. Please check input data for required fields.")
    print("-------------------------------------------------")

    return df_out



def gen_systemic_modulator_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a set of surrogates for systemic factors that may modulate amyloid-beta and
    plasma Tau levels, derived from UK Biobank biomarker data. This function focuses on
    physiological systems (kidney, liver) and general inflammation, excluding direct
    neuro-biomarkers like amyloid or tau.

    The selected factors include:
    1.  Cystatin C (kidney function)
    2.  Creatinine (kidney function)
    3.  Alanine Aminotransferase (ALT - liver function)
    4.  Aspartate Aminotransferase (AST - liver function)
    5.  C-reactive protein (CRP - systemic inflammation)
    6.  Inverted Albumin (general health/nutritional status)

    Args:
        ukb_main_df: A pandas DataFrame containing the raw UK Biobank data, indexed by 'eid'.

    Returns:
        A pandas DataFrame with one row per participant and columns for each of the surrogate
        factors. For all columns, a higher value indicates a state more associated with
        impaired systemic function or higher inflammation. The DataFrame is indexed by 'eid'.
    """
    # Create a new DataFrame to hold the surrogates, using the index from the input
    surrogates = pd.DataFrame(index=ukb_main_df.index)

    # --- Kidney Function Markers (Higher = worse function) ---
    # Field 30720: Cystatin C.
    surrogates['risk_cystatin_c'] = ukb_main_df.get('30720-0.0')
    # Field 30700: Creatinine.
    surrogates['risk_creatinine'] = ukb_main_df.get('30700-0.0')

    # --- Liver Function Markers (Higher = more liver stress) ---
    # Field 30620: Alanine aminotransferase (ALT).
    surrogates['risk_alt'] = ukb_main_df.get('30620-0.0')
    # Field 30650: Aspartate aminotransferase (AST).
    surrogates['risk_ast'] = ukb_main_df.get('30650-0.0')
    
    # --- Systemic Inflammation Marker (Higher = more inflammation) ---
    # Field 30710: C-reactive protein (CRP).
    surrogates['risk_crp'] = ukb_main_df.get('30710-0.0')

    # --- General Health Marker (Inverted to create risk score) ---
    # Field 30600: Albumin. Lower albumin is associated with poorer health outcomes.
    # We multiply by -1 so that a higher value in our surrogate indicates higher risk.
    if '30600-0.0' in ukb_main_df:
        surrogates['risk_inverted_albumin'] = ukb_main_df['30600-0.0'] * -1

    return surrogates



def gen_biochemistry_risk_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a set of surrogates for blood biochemistry markers associated with dementia risk,
    derived from the UK Biobank dataset.

    The selected biomarkers are established risk factors for dementia or related cardiometabolic
    conditions. They include markers for dyslipidemia, glucose control, and general metabolic health.

    The selected factors include:
    1.  Total Cholesterol
    2.  LDL Cholesterol
    3.  Apolipoprotein B
    4.  Triglycerides
    5.  Glucose
    6.  Glycated Haemoglobin (HbA1c)
    7.  Lipoprotein A
    8.  Inverted HDL Cholesterol (as low HDL is a risk factor)
    9.  Inverted Apolipoprotein A (as low ApoA is a risk factor)

    Args:
        ukb_main_df: A pandas DataFrame containing the raw UK Biobank data, indexed by 'eid'.

    Returns:
        A pandas DataFrame with one row per participant and columns for each of the surrogate
        factors. For all columns, a higher value indicates a state more associated with
        dementia risk. The DataFrame is indexed by 'eid'.
    """
    # Create a new DataFrame to hold the surrogates, using the index from the input
    surrogates = pd.DataFrame(index=ukb_main_df.index)

    # --- Direct Risk Factors (Higher value = Higher risk) ---
    
    # Field 30690: Total Cholesterol
    surrogates['risk_total_cholesterol'] = ukb_main_df.get('30690-0.0')
    
    # Field 30780: LDL direct
    surrogates['risk_ldl_direct'] = ukb_main_df.get('30780-0.0')
    
    # Field 30640: Apolipoprotein B (a key component of LDL)
    surrogates['risk_apob'] = ukb_main_df.get('30640-0.0')

    # Field 30870: Triglycerides
    surrogates['risk_triglycerides'] = ukb_main_df.get('30870-0.0')

    # Field 30740: Glucose
    surrogates['risk_glucose'] = ukb_main_df.get('30740-0.0')
    
    # Field 30750: Glycated haemoglobin (HbA1c) - a marker of long-term glucose levels
    surrogates['risk_hba1c'] = ukb_main_df.get('30750-0.0')

    # Field 30790: Lipoprotein A - an independent cardiovascular risk factor
    surrogates['risk_lipoprotein_a'] = ukb_main_df.get('30790-0.0')

    # --- Protective Factors (Inverted to create risk score) ---
    # For these, lower levels are associated with higher risk. We multiply by -1
    # so that a higher value in our surrogate indicates higher risk.

    # Field 30760: HDL cholesterol ("good" cholesterol)
    if '30760-0.0' in ukb_main_df:
        surrogates['risk_inverted_hdl'] = ukb_main_df.get('30760-0.0') * -1

    # Field 30630: Apolipoprotein A (a key component of HDL)
    if '30630-0.0' in ukb_main_df:
        surrogates['risk_inverted_apoa'] = ukb_main_df.get('30630-0.0') * -1

    return surrogates



