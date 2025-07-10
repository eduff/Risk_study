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
    #ukb_main_df['risk_hypertension'] = ukb_main_df['20002-0.0'].apply(lambda x: 1 if x == 1071 else 0)
    #ukb_main_df['systolic']=ukb_main_df['4080-0.0']
    #ukb_main_df['diastolic']=ukb_main_df['4079-0.0']
    
    #ukb_main_df['risk_hypertension'] = (ukb_main_df['4080-0.0']>130) & (ukb_main_df['4079-0.0']>80)
    
    # Use a dictionary to map self-reported vascular/heart problems (Field 6150)
    # 1: Heart attack, 2: Angina, 3: Stroke, 4: High blood pressure
    vascular_map = ukb_main_df.filter(regex='^6150-').stack().reset_index()
    vascular_map.columns = ['eid', 'field', 'condition_code']
    vascular_map = vascular_map.drop('field', axis=1).drop_duplicates()
    hypertension_eids = vascular_map[vascular_map['condition_code'] == 4]['eid']
    ukb_main_df['risk_hypertension'] = ukb_main_df.index.isin(hypertension_eids).astype(int)


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
    ukb_main_df['risk_obesity'] = BMI > 30

#     #height = ukb_main_df['12144-0.0']  # Field 4076: Height
#     waist = ukb_main_df['48-0.0']    # Field 48: Waist circumference
#    # BMI
#     weight = ukb_main_df['21002-0.0'] # Weight
#     height = (weight / BMI ) ** 0.5  # Calculate height from weight and BMI
#     ratio = waist/(height*100) 
#     sex_col = ukb_main_df['31-0.0']  # Field 31
#     ukb_main_df['risk_obesity']= ratio.apply(lambda x: 1 if x > 0.85 else 0) # Waist to height ratio > 0.85 indicates higher risk

#     ukb_main_df['risk_obesity']= ratio

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


import pandas as pd
import numpy as np

def gen_viral_reactivation_markers(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a set of markers for brain viral (re)activation, with a focus on HSV-1,
    derived from UK Biobank data.

    Args:
        ukb_main_df: A pandas DataFrame containing the raw UK Biobank data.

    Returns:
        A pandas DataFrame with one row per participant and columns for each of the
        viral reactivation markers.
    """

    # --- 1. Direct Viral and Immune Response Markers ---
    # Field 23050: HSV-1 seropositivity
    ukb_main_df['hsv1_seropositivity'] = ukb_main_df['23050-0.0'].apply(lambda x: 1 if x == 1 else 0)

    # Field 30000: White blood cell (leukocyte) count
    ukb_main_df['leukocyte_count'] = ukb_main_df['30000-0.0']

    # Field 30120: Lymphocyte count
    ukb_main_df['lymphocyte_count'] = ukb_main_df['30120-0.0']

    # Field 30130: Monocyte count
    ukb_main_df['monocyte_count'] = ukb_main_df['30130-0.0']

    # Field 30140: Neutrophill count
    ukb_main_df['neutrophil_count'] = ukb_main_df['30140-0.0']

    # Field 30710: C-reactive protein
    ukb_main_df['crp'] = ukb_main_df['30710-0.0']


    # --- 2. Olink Proteomics Markers ---
    # Note: The following are placeholders for the actual Olink data fields.
    # Replace the field IDs with the correct ones when available.
    ukb_main_df['nfl'] = ukb_main_df.get('31043-0.0', pd.Series(np.nan, index=ukb_main_df.index))
    ukb_main_df['gfap'] = ukb_main_df.get('31042-0.0', pd.Series(np.nan, index=ukb_main_df.index))
    ukb_main_df['ab40'] = ukb_main_df.get('31040-0.0', pd.Series(np.nan, index=ukb_main_df.index))
    ukb_main_df['ab42'] = ukb_main_df.get('31041-0.0', pd.Series(np.nan, index=ukb_main_df.index))
    ukb_main_df['ptau181'] = ukb_main_df.get('31044-0.0', pd.Series(np.nan, index=ukb_main_df.index))
    ukb_main_df['il6'] = ukb_main_df.get('olink_il6-0.0', pd.Series(np.nan, index=ukb_main_df.index))
    ukb_main_df['tnf'] = ukb_main_df.get('olink_tnf-0.0', pd.Series(np.nan, index=ukb_main_df.index))
    ukb_main_df['ifny'] = ukb_main_df.get('olink_ifny-0.0', pd.Series(np.nan, index=ukb_main_df.index))
    ukb_main_df['cxcl10'] = ukb_main_df.get('olink_cxcl10-0.0', pd.Series(np.nan, index=ukb_main_df.index))


    # --- 3. Clinical and Phenotypic Data ---
    # Field 41202, 41204: ICD10 Codes for neurological conditions
    neuro_codes = ['G00', 'G03', 'G04', 'F00', 'F01', 'F02', 'F03', 'G30']
    ukb_main_df['neuro_diagnosis'] = ukb_main_df['41202-0.0'].apply(lambda x: 1 if x in neuro_codes else 0)

    # Field 20002: Self-reported non-cancer illness
    ukb_main_df['self_reported_neuro'] = ukb_main_df['20002-0.0'].apply(lambda x: 1 if x in [1263, 1262, 1261, 1260, 1483] else 0)

    # Field 20544: Mental health problems diagnosed by professional
    #ukb_main_df['prof_diag_mental_health'] = ukb_main_df['20544-0.0'].apply(lambda x: 1 if x is not None and x != '' else 0)

    # Field 6145:	Illness, injury, bereavement, stress in last 2 years
    ukb_main_df['Illness_stress'] = ukb_main_df['6145-0.0'].apply(lambda x: 1 if x>0 else 0)

    # --- 4. Cognitive Function ---
    # Field 20016: Fluid intelligence score
    ukb_main_df['fluid_intelligence'] = ukb_main_df['20016-0.0']

    # Field 20023: Mean time to correctly identify matches
    ukb_main_df['processing_speed'] = ukb_main_df['20023-0.0']

    # Field 20018: Prospective memory result
    ukb_main_df['prospective_memory'] = ukb_main_df['20018-0.0']


    # --- 5. Brain Imaging ---
    # These fields indicate the presence of imaging data. The actual analysis would require processing the image files.
    #ukb_main_df['t1_image_available'] = ukb_main_df['20216-0.0'].apply(lambda x: 1 if pd.notna(x) else 0)
    #ukb_main_df['t2_flair_image_available'] = ukb_main_df['20220-0.0'].apply(lambda x: 1 if pd.notna(x) else 0)
    #ukb_main_df['wmh_volume'] = ukb_main_df['25781-0.0']
    #ukb_main_df['dti_available'] = ukb_main_df['20250-0.0'].apply(lambda x: 1 if pd.notna(x) else 0)
    #ukb_main_df['asl_available'] = ukb_main_df['26300-0.0'].apply(lambda x: 1 if pd.notna(x) else 0)


    # --- Consolidate Markers ---
    viral_markers_df = ukb_main_df[[
        'hsv1_seropositivity', 'leukocyte_count', 'lymphocyte_count', 'monocyte_count',
        'neutrophil_count', 'crp', 'nfl', 'gfap', 'ab40', 'ab42', 'ptau181', 'il6', 'tnf', 'ifny', 'cxcl10',
        'neuro_diagnosis', 'self_reported_neuro','Illness_stress', 'fluid_intelligence',
        'processing_speed', 'prospective_memory'
    ]].copy() #   'prof_diag_mental_health',,'t1_image_available', 't2_flair_image_available', 'wmh_volume', 'dti_available', 'asl_available'

    return viral_markers_df



import pandas as pd
import numpy as np

def gen_hsv1_susceptibility_markers(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a set of markers for susceptibility to HSV-1 related neurodegeneration,
    derived from UK Biobank data. These markers represent processes like immunosenescence,
    chronic inflammation, and metabolic dysfunction.

    Note: APOE-e4 status is a critical covariate and should be included in any
    downstream analysis but is not generated in this function.

    Args:
        ukb_main_df: A pandas DataFrame containing the raw UK Biobank data.

    Returns:
        A pandas DataFrame with one row per participant and columns for each of the
        susceptibility markers.
    """

    markers_df = pd.DataFrame(index=ukb_main_df.index)

    # --- 1. Immunosenescence Markers ---
    # Field 30120: Lymphocyte count
    markers_df['lymphocyte_count'] = ukb_main_df.get('30120-0.0')

    # Field 30140: Neutrophil count
    markers_df['neutrophil_count'] = ukb_main_df.get('30140-0.0')

    # Neutrophil to Lymphocyte Ratio (NLR)
    # A higher ratio indicates greater inflammation and is associated with immunosenescence.
    # Add a small epsilon to avoid division by zero
    markers_df['neutrophil_lymphocyte_ratio'] = ukb_main_df.get('30140-0.0') / (ukb_main_df.get('30120-0.0', 0) + 1e-9)


    # --- 2. Chronic Systemic Inflammation Markers ---
    # Field 30710: C-reactive protein
    markers_df['crp'] = ukb_main_df.get('30710-0.0')

    # Field 23480: Glycoprotein Acetyls (from NMR metabolomics)
    markers_df['glyca'] = ukb_main_df.get('23480-0.0')

    # Olink Proteomics (placeholders for relevant inflammatory markers)
    markers_df['olink_il6'] = ukb_main_df.get('olink_il6-0.0') # Example field ID
    markers_df['olink_tnf'] = ukb_main_df.get('olink_tnf-0.0') # Example field ID


    # --- 3. Blood-Brain Barrier (BBB) Dysfunction Proxies ---
    # Field 30600: Albumin (lower serum levels can be associated with BBB issues)
    markers_df['albumin'] = ukb_main_df.get('30600-0.0')

    # Hypertension status from blood pressure readings
    systolic_bp = ukb_main_df.get('4080-0.0')
    diastolic_bp = ukb_main_df.get('4079-0.0')
    markers_df['hypertension_bp_criteria'] = ((systolic_bp >= 140) | (diastolic_bp >= 90)).astype(int)

    # History of stroke (ICD-10 codes I60-I69)
    stroke_codes = [f'I6{i}' for i in range(10)]
    markers_df['stroke_history_icd10'] = ukb_main_df.get('41202-0.0', pd.Series(np.nan, index=ukb_main_df.index)).isin(stroke_codes).astype(int)


    # --- 4. Metabolic Dysfunction Markers ---
    # Field 2443: Doctor diagnosed diabetes
    markers_df['diabetes_diagnosis'] = ukb_main_df.get('2443-0.0', pd.Series(0, index=ukb_main_df.index)).apply(lambda x: 1 if x == 1 else 0)

    # Field 30750: Glycated haemoglobin (HbA1c)
    markers_df['hba1c'] = ukb_main_df.get('30750-0.0')

    # Field 21001: Body mass index (BMI)
    markers_df['bmi'] = ukb_main_df.get('21001-0.0')

    # Field 30870: Triglycerides
    markers_df['triglycerides'] = ukb_main_df.get('30870-0.0')

    # Field 30760: HDL cholesterol
    markers_df['hdl_cholesterol'] = ukb_main_df.get('30760-0.0')


    # --- 5. Genetic Factors ---
    # Field 22182: HLA imputation values availability
    # The presence of this data allows for specialized analysis of immune-related genetic risk.
    markers_df['hla_data_available'] = ukb_main_df.get('22182-0.0').notna().astype(int)


    return markers_df

if __name__ == '__main__':
    # This is a placeholder for loading the actual UK Biobank data.
    # Researchers with access should replace this with their data loading mechanism.
    # For demonstration, we create a dummy dataframe.
    num_records = 100
    dummy_data = {
        'eid': range(num_records),
        '30120-0.0': np.random.uniform(1, 4, num_records),         # lymphocyte_count
        '30140-0.0': np.random.uniform(2, 7.5, num_records),      # neutrophil_count
        '30710-0.0': np.random.uniform(0.1, 10, num_records),     # crp
        '23480-0.0': np.random.uniform(0.8, 2.5, num_records),    # glyca
        '30600-0.0': np.random.uniform(35, 50, num_records),      # albumin
        '4080-0.0': np.random.uniform(110, 160, num_records),     # systolic_bp
        '4079-0.0': np.random.uniform(70, 100, num_records),      # diastolic_bp
        '41202-0.0': np.random.choice(['I63', 'I10', 'E11', ''], num_records), # stroke_history_icd10
        '2443-0.0': np.random.choice([1, 0, -1, -3], num_records),# diabetes_diagnosis
        '30750-0.0': np.random.uniform(25, 60, num_records),      # hba1c
        '21001-0.0': np.random.uniform(18, 40, num_records),      # bmi
        '30870-0.0': np.random.uniform(0.5, 3, num_records),      # triglycerides
        '30760-0.0': np.random.uniform(0.8, 2.5, num_records),    # hdl_cholesterol
        '22182-0.0': np.random.choice([1, np.nan], num_records)   # hla_data_available
    }
    ukb_dummy_df = pd.DataFrame(dummy_data)
    ukb_dummy_df.set_index('eid', inplace=True)


    # Generate the susceptibility markers
    hsv1_susceptibility_markers = generate_hsv1_susceptibility_markers(ukb_dummy_df)

    # Display the first few rows of the generated markers
    print("Generated HSV-1 Susceptibility Markers:")
    print(hsv1_susceptibility_markers.head())



import pandas as pd
import numpy as np

def generate_prodromal_ad_surrogates(ukb_main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a set of surrogates for prodromal dementia/AD derived from UK Biobank data.
    This focuses on indicators of cognitive decline, neurobiomarkers, and memory issues,
    rather than general dementia risk factors.

    Args:
        ukb_main_df: A pandas DataFrame containing the raw UK Biobank data.

    Returns:
        A pandas DataFrame with one row per participant and columns for each
        prodromal AD surrogate. Higher values generally indicate a stronger
        surrogate for prodromal AD.
    """

    surrogates = pd.DataFrame(index=ukb_main_df.index)

    # --- 1. Decline in Specific Cognitive Test Scores Across Assessments ---

    # We need to access data from different assessment visits.
    # Assuming the UKB data is structured with -0.0 for initial assessment,
    # -1.0 for first repeat, -2.0 for second repeat, etc.

    # 1.1. Fluid Intelligence Score Decline (Field 20016)
    # This test was administered at initial assessment and follow-up.
    # Calculate change: (Follow-up Score - Initial Score)
    # A negative value (decline) indicates higher risk.
    if '20016-0.0' in ukb_main_df.columns and '20016-1.0' in ukb_main_df.columns:
        surrogates['cog_fluid_intelligence_decline'] = ukb_main_df['20016-1.0'] - ukb_main_df['20016-0.0']
        # Invert so higher value means higher decline
        surrogates['cog_fluid_intelligence_decline'] = -surrogates['cog_fluid_intelligence_decline']
    else:
        surrogates['cog_fluid_intelligence_decline'] = np.nan

    # 1.2. Reaction Time Decline (Field 20023)
    # Administered at initial assessment and follow-up.
    # Calculate change: (Follow-up Time - Initial Time)
    # A positive value (slower reaction time) indicates higher risk.
    if '20023-0.0' in ukb_main_df.columns and '20023-1.0' in ukb_main_df.columns:
        surrogates['cog_reaction_time_decline'] = ukb_main_df['20023-1.0'] - ukb_main_df['20023-0.0']
    else:
        surrogates['cog_reaction_time_decline'] = np.nan

    # 1.3. Pairs Matching Score Decline (Field 20026)
    # This test assesses visuospatial memory.
    # Calculate change: (Follow-up Score - Initial Score)
    # A negative value (decline) indicates higher risk.
    if '20026-0.0' in ukb_main_df.columns and '20026-1.0' in ukb_main_df.columns:
        surrogates['cog_pairs_matching_decline'] = ukb_main_df['20026-1.0'] - ukb_main_df['20026-0.0']
        # Invert so higher value means higher decline
        surrogates['cog_pairs_matching_decline'] = -surrogates['cog_pairs_matching_decline']
    else:
        surrogates['cog_pairs_matching_decline'] = np.nan

    # 1.4. Numeric Memory Decline (Field 20019)
    # A negative value (decline) indicates higher risk.
    if '20019-0.0' in ukb_main_df.columns and '20019-1.0' in ukb_main_df.columns:
        surrogates['cog_numeric_memory_decline'] = ukb_main_df['20019-1.0'] - ukb_main_df['20019-0.0']
        # Invert so higher value means higher decline
        surrogates['cog_numeric_memory_decline'] = -surrogates['cog_numeric_memory_decline']
    else:
        surrogates['cog_numeric_memory_decline'] = np.nan

    # --- 2. Neurobiomarkers ---

    # 2.1. Brain MRI Measurements (assuming processed data fields are available)
    # Example: Grey matter volume (lower is generally worse)
    # Field 25781: Volume of grey matter (normalised for head size)
    if '25781-2.0' in ukb_main_df.columns: # Assuming data is from follow-up imaging (instance 2)
        surrogates['neuro_grey_matter_volume'] = -ukb_main_df['25781-2.0'] # Invert so higher value means lower volume
    else:
        surrogates['neuro_grey_matter_volume'] = np.nan

    # 2.2. White Matter Hyperintensities (higher is generally worse)
    # Field 25829: Volume of white matter hyperintensities (normalised for head size)
    if '25829-2.0' in ukb_main_df.columns:
        surrogates['neuro_wm_hyperintensities'] = ukb_main_df['25829-2.0']
    else:
        surrogates['neuro_wm_hyperintensities'] = np.nan

    # 2.3. Hippocampal Volume (lower is generally worse)
    # Field 25792: Volume of left hippocampus (normalised for head size)
    # Field 25793: Volume of right hippocampus (normalised for head size)
    if '25792-2.0' in ukb_main_df.columns and '25793-2.0' in ukb_main_df.columns:
        surrogates['neuro_hippocampal_volume'] = -(ukb_main_df['25792-2.0'] + ukb_main_df['25793-2.0'])
    else:
        surrogates['neuro_hippocampal_volume'] = np.nan

    # --- 3. Reports of Memory Issues / Subjective Cognitive Decline (SCD) ---

    # 3.1. Self-reported memory changes (Field 20120)
    # 1: Yes, 0: No
    if '20120-0.0' in ukb_main_df.columns:
        # Recode: 1 if "Yes, has been noticeable", 0 otherwise
        surrogates['mem_self_reported_decline'] = ukb_main_df['20120-0.0'].apply(lambda x: 1 if x == 1 else 0)
    else:
        surrogates['mem_self_reported_decline'] = np.nan

    # 3.2. Frequency of forgetting things (Field 20121)
    # Recoding: Higher values indicate more frequent forgetting (higher risk).
    # 1: Never/rarely, 2: Sometimes, 3: Often, 4: Very often
    forgetting_mapping = {
        1: 0,  # Never/rarely
        2: 1,  # Sometimes
        3: 2,  # Often
        4: 3,  # Very often
        -1: np.nan, # Do not know
        -3: np.nan  # Prefer not to answer
    }
    if '20121-0.0' in ukb_main_df.columns:
        surrogates['mem_forgetting_frequency'] = ukb_main_df['20121-0.0'].map(forgetting_mapping)
    else:
        surrogates['mem_forgetting_frequency'] = np.nan

    # 3.3. Problems with memory (Field 699) - from Mental Health Questionnaire
    # Coding: 0: No, 1: Yes
    if '699-0.0' in ukb_main_df.columns:
        surrogates['mem_problems_mental_health'] = ukb_main_df['699-0.0'].apply(lambda x: 1 if x == 1 else 0)
    else:
        surrogates['mem_problems_mental_health'] = np.nan

    # --- 4. Other Potential Prodromal Indicators ---

    # 4.1. Sleep duration changes (e.g., severe short or long sleep)
    # Field 1160: Typical sleep duration
    # Extreme sleep durations (e.g., < 6 hours or > 9 hours) might be a surrogate
    if '1160-0.0' in ukb_main_df.columns:
        surrogates['other_sleep_duration_risk'] = ukb_main_df['1160-0.0'].apply(
            lambda x: 1 if (x < 6 or x > 9) else 0
        )
    else:
        surrogates['other_sleep_duration_risk'] = np.nan

    # 4.2. Falls (can be a sign of cognitive or motor decline)
    # Field 20002: Self-reported non-cancer illness
    # Coding 1459: "Falls"
    if '20002-0.0' in ukb_main_df.columns:
        falls_eids = ukb_main_df[ukb_main_df['20002-0.0'] == 1459].index
        surrogates['other_reported_falls'] = surrogates.index.isin(falls_eids).astype(int)
    else:
        surrogates['other_reported_falls'] = 0

    # 4.3. History of Delirium (often associated with underlying vulnerability)
    # This might be captured through hospitalization records or self-report.
    # UKB field 41270: ICD10 codes for primary diagnosis
    # F05: Delirium, not induced by alcohol or other psychoactive substances
    # Need to check across multiple instances of Field 41270 and other diagnostic fields.
    delirium_icd10_codes = ['F05']
    if '41270-0.0' in ukb_main_df.columns:
        # Check all instances of ICD10 primary and secondary diagnoses
        icd10_cols = [col for col in ukb_main_df.columns if col.startswith('41270-') or col.startswith('41271-')]
        delirium_indicator = ukb_main_df[icd10_cols].apply(
            lambda row: any(str(code).startswith(dc) for dc in delirium_icd10_codes for code in row.dropna()),
            axis=1
        )
        surrogates['other_history_delirium'] = delirium_indicator.astype(int)
    else:
        surrogates['other_history_delirium'] = 0

    return surrogates

