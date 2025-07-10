import pandas as pd
import re
import os
import math



def remove_biobank_fields(fields: list[str], categories_to_remove: list[str]) -> list[str]:
    """
    Removes UK Biobank fields from a list based on specified categories.
    This is a helper function used by the main data loader.
    """
    if not categories_to_remove:
        return fields
    patterns = [COMPREHENSIVE_CATEGORY_MAP [cat] for cat in categories_to_remove if cat in COMPREHENSIVE_CATEGORY_MAP ]
    if not patterns:
        return fields
    combined_regex = re.compile("|".join(f"({p})" for p in patterns))
    return [field for field in fields if not combined_regex.match(field)]


import pandas as pd
import re
from collections import defaultdict
import os


import pandas as pd
import re
import os
import math
from collections.abc import Iterable


import pandas as pd
import re
import os
import math
from collections.abc import Iterable

COMPREHENSIVE_CATEGORY_MAP = {
    # --- Imaging (Largest Group) ---
    "Imaging - Brain MRI (Structural)": r"^250[0-2][0-9]|^25[7-9][0-9]{2}|^26[5-9][0-9]{2}|^27[0-9]{3}",
    "Imaging - Brain MRI (Diffusion)": r"^25[1-6][0-9]{2}",
    "Imaging - Brain MRI (Functional)": r"^250[3-5][0-9]",
    "Imaging - Heart MRI": r"^2242[0-9]|^241[0-9]{2}|^310[6-9][0-9]",
    "Imaging - Abdominal & Composition": r"^224[0-1][0-9]|^23[1-3][0-9]{2}",
    "Imaging - Carotid Ultrasound": r"^226[7-8][0-9]|^2022[6-9]",
    "Imaging - Retinal (Fundus & OCT)": r"^2101[5-8]|^27[8-9][0-9]{2}|^28[5-9][0-9]{2}",

    # --- Health Records & Diagnoses ---
    "Health Records - ICD10": r"^41202$|^41204$|^41270$|^13[0-2][0-9]{3}",
    "Health Records - ICD9": r"^41203$|^41205$|^41271$",
    "Health Records - OPCS4 (Procedures)": r"^41200$|^41210$|^41272$",
    "Health Records - Cancer Register": r"^4000[56]$|^4001[1-3]$",
    "Health Records - Death Register": r"^4000[0-2]$|^4000[7-9]$|^40010$",
    "Health Records - GP Clinical": r"^42040$",

    # --- Genetics ---
    "Genetics - Imputed Data & Genotypes": r"^22[1-8][0-9]{2}",
    "Genetics - WGS/WES Data Files": r"^231[4-9][0-9]|^240[3-6][0-9]",
    "Genetics - QC & Metadata": r"^220[0-2][0-9]",
    
    # --- Questionnaires & Self-Report ---
    "Questionnaire - Mental Health": r"^204[0-9]{2}|^205[0-5][0-9]|^29[0-1][0-9]{2}",


def load_biobank_csv_with_chunking(
    filepath: str,
    categories_to_exclude: list[str] = None,
    categories_to_include: list[str] = None,
    datafields_to_include: list[str] = None,
    all_instances: bool = True, # Default changed to True for convenience
    chunksize: int = 10000,
    id_column: str = 'eid'
) -> pd.DataFrame:
    """
    Loads a UK Biobank CSV efficiently with powerful include/exclude logic.

    Args:
        filepath: The path to the UK Biobank .csv file.
        categories_to_exclude: List of category names to remove from the dataset.
        categories_to_include: List of category names to load.
        datafields_to_include: List of specific datafield IDs to load.
                               Handles full field names (e.g., '6138-0.0') by
                               extracting the base ID ('6138').
        all_instances: If True, includes all instances of a field. If False,
                       only includes the first instance (e.g., '-0.').
        chunksize: The number of rows to read into memory at a time.
        id_column: The primary participant identifier column, which is always kept.

    Returns:
        A pandas DataFrame containing the data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print("### Step 1: Analyzing and selecting columns... ###")
    all_columns = pd.read_csv(filepath, nrows=0, engine='c').columns.tolist()

    # --- Column Selection Logic ---
    
    # 1. Start with all columns, then apply exclusions
    if categories_to_exclude:
        print(f"Excluding categories: {categories_to_exclude}")
        exclude_patterns = [COMPREHENSIVE_CATEGORY_MAP.get(cat) for cat in categories_to_exclude if cat]
        exclude_regex = re.compile("|".join(f"({p})" for p in exclude_patterns))
        available_columns = [col for col in all_columns if not exclude_regex.match(col.split('-')[0])]
    else:
        available_columns = all_columns

    # 2. If no include lists are given, use the available columns.
    if not categories_to_include and not datafields_to_include:
        columns_to_keep = set(available_columns)
    # 3. Otherwise, build the include set from categories and specific fields.
    else:
        columns_to_keep = {id_column} # Use a set to handle unions gracefully

        # Add columns from included categories
        if categories_to_include:
            print(f"Including categories: {categories_to_include}")
            include_cat_patterns = [COMPREHENSIVE_CATEGORY_MAP.get(cat) for cat in categories_to_include if cat]
            include_cat_regex = re.compile("|".join(f"({p})" for p in include_cat_patterns))
            for col in available_columns:
                if include_cat_regex.match(col.split('-')[0]):
                    columns_to_keep.add(col)
        
        # Add specific datafields
        if datafields_to_include:
            # FLEXIBILITY CHANGE: Extract base IDs from the user's list
            base_ids_to_include = {str(f).split('-')[0] for f in datafields_to_include}
            print(f"Including specific datafields (base IDs): {sorted(list(base_ids_to_include))}")
            
            for base_id in base_ids_to_include:
                for col in available_columns:
                    if col.split('-')[0] == base_id:
                        columns_to_keep.add(col)
        
        # If all_instances is False, filter the final set to only instance 0
        if not all_instances:
            print("Filtering to include first instances only (all_instances=False).")
            # Ensure the regex matches '-0.' to correctly identify instance 0
            instance_0_cols = {col for col in columns_to_keep if '-0.' in col}
            instance_0_cols.add(id_column)
            columns_to_keep = instance_0_cols
            
    final_columns = sorted(list(columns_to_keep))
    if id_column not in final_columns and id_column in all_columns:
        final_columns.insert(0, id_column)

    print(f"Final column selection: {len(final_columns)} columns to be loaded.")

    # --- Loading Data (no changes from here) ---
    print("\n### Step 2: Loading data in chunks... ###")
    with open(filepath, 'r') as f:
        total_rows = sum(1 for line in f) - 1
    
    if total_rows <= 0:
        print("CSV file is empty or contains only a header.")
        return pd.DataFrame(columns=final_columns)

    total_chunks = math.ceil(total_rows / chunksize)
    chunk_list = []
    
    try:
        with pd.read_csv(filepath, usecols=final_columns, chunksize=chunksize, engine='c', low_memory=False) as reader:
            for i, chunk in enumerate(reader):
                chunk_list.append(chunk)
                percent_loaded = (i + 1) / total_chunks * 100
                print(f"\rLoaded chunk {i + 1}/{total_chunks} ({percent_loaded:.1f}%)", end="")
    except Exception as e:
        print(f"\nAn error occurred during chunk processing: {e}")
        return pd.DataFrame()

    print("\n\n### Step 3: Concatenating chunks... ###")
    final_df = pd.concat(chunk_list, ignore_index=True)
    print("DataFrame successfully created.")
    
    return final_df


import pandas as pd
import re
from collections import Counter
import os

def investigate_uncategorized_fields(filepath: str, top_n_prefixes: int = 10):
    """
    Analyzes a UKB CSV header to identify and summarize the largest groups
    of uncategorized fields.

    Args:
        filepath: The path to the UK Biobank .csv file.
        top_n_prefixes: The number of most common prefixes to display.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file was not found at: {filepath}")

    print(f"🔬 Investigating uncategorized fields in {os.path.basename(filepath)}...")
    all_columns = pd.read_csv(filepath, nrows=0).columns.tolist()

    uncategorized_fields = []
    compiled_patterns = {cat: re.compile(pattern) for cat, pattern in COMPREHENSIVE_CATEGORY_MAP.items()}

    # Separate categorized from uncategorized
    for col in all_columns:
        if col == 'eid':
            continue
        field_id = col.split('-')[0]
        if not any(pattern.match(field_id) for pattern in compiled_patterns.values()):
            uncategorized_fields.append(col)

    print(f"\nFound {len(uncategorized_fields)} uncategorized fields out of {len(all_columns) - 1} total fields.")

    if not uncategorized_fields:
        print("✅ No uncategorized fields found!")
        return

    # Count the prefixes of the uncategorized fields
    prefix_counts = Counter(field.split('-')[0] for field in uncategorized_fields)
    
    # --- Display the results ---
    print(f"\n--- Top {top_n_prefixes} Most Common Uncategorized Field Prefixes ---")
    print(f"{'Field Prefix':<15} {'Count':<10} {'Potential Category'}")
    print("-" * 60)

    for prefix, count in prefix_counts.most_common(top_n_prefixes):
        # Add interpretations for common non-standard prefixes
        potential_category = "Unknown"
        if prefix.startswith('420'):
            potential_category = "Primary Care (GP) - Read v2 Code"
        elif prefix.startswith('4000'):
            potential_category = "Hospital Inpatient (HES) - Diagnosis (ICD10)"
        elif prefix.startswith('41270'):
            potential_category = "Hospital Inpatient (HES) - Diagnosis (ICD10)"
        elif prefix.startswith('41271'):
            potential_category = "Hospital Inpatient (HES) - Diagnosis (ICD9)"
        elif prefix.startswith('41272'):
            potential_category = "Hospital Inpatient (HES) - Procedure (OPCS4)"
        elif prefix.startswith('204'): # Example
             potential_category = "Online Questionnaire (e.g., Mood)"

        print(f"{prefix:<15} {count:<10} {potential_category}")
        
def summarize_biobank_csv_comprehensive(filepath: str, id_column: str = 'eid') -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes a UK Biobank CSV header using a comprehensive category map
        to provide a detailed summary of datafield types and counts.

        Args:
            filepath: The path to the UK Biobank .csv file.
            id_column: The name of the primary participant identifier column (e.g., 'eid').

        Returns:
            A tuple containing two pandas DataFrames: (category_summary, instance_summary).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file was not found at: {filepath}")

        all_columns = pd.read_csv(filepath, nrows=0).columns.tolist()

        if id_column in all_columns:
            all_columns.remove(id_column)

        total_fields = len(all_columns)
        category_counts = defaultdict(int)
        instance_counts = defaultdict(int)

        compiled_patterns = {cat: re.compile(pattern) for cat, pattern in COMPREHENSIVE_CATEGORY_MAP.items()}
        instance_pattern = re.compile(r"-(\d+)\.\d+")

        for col in all_columns:
            field_id = col.split('-')[0]
            matched = False
            for cat, pattern in compiled_patterns.items():
                if pattern.match(field_id):
                    category_counts[cat] += 1
                    matched = True
                    break
            if not matched:
                category_counts["Uncategorized"] += 1

            match = instance_pattern.search(col)
            if match:
                instance_num = match.group(1)
                instance_counts[f"Instance {instance_num}"] += 1
            else:
                instance_counts["No Instance Info"] += 1

        category_summary = pd.DataFrame(
            category_counts.items(), columns=['Category', 'Field Count']
        ).sort_values('Field Count', ascending=False).reset_index(drop=True)
        category_summary['Percentage'] = (category_summary['Field Count'] / total_fields * 100).round(2)

        instance_summary = pd.DataFrame(
            instance_counts.items(), columns=['Assessment Instance', 'Field Count']
        ).sort_values('Field Count', ascending=False).reset_index(drop=True)
        instance_summary['Percentage'] = (instance_summary['Field Count'] / total_fields * 100).round(2)

        return category_summary
    
    
    
import pandas as pd
import glob
import os

def determine_apoe_genotype(row):
    """Determines APOE genotype from rs429358 and rs7412 variant counts."""
    # Allele counts are based on the number of specified alleles (C for rs429358, T for rs7412)
    c_count = row['rs429358_C']  # Count of C alleles (e4 marker)
    t_count = row['rs7412_T']    # Count of T alleles (e3/e4 marker)

    # e2 is defined by C at rs7412. So, (2 - t_count) is the number of C alleles at rs7412.
    num_e2 = 2 - t_count
    num_e4 = c_count
    num_e3 = 2 - (num_e2 + num_e4)

    alleles = sorted(['e' + str(allele) for allele in [2]*num_e2 + [3]*num_e3 + [4]*num_e4])
    return '/'.join(alleles)


def create_merged_genotype_df(file_paths):
    """
    Processes and merges multiple genotyping files with specific logic for
    column naming, APOE processing, and clean merging.
    """
    if not file_paths:
        print("Error: No file paths provided.")
        return None

    # Define columns that are common metadata and should not be renamed or duplicated.
    COMMON_COLS = ['IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    
    variant_dfs = []
    common_dfs = []

    for file_path in file_paths:
        try:
            # --- 1. Read file and extract gene name ---
            df = pd.read_csv(file_path, sep='\t')
            if 'FID' not in df.columns:
                print(f"Warning: 'FID' column not found in {file_path}. Skipping.")
                continue
            
            df.set_index('FID', inplace=True)
            gene_name = os.path.basename(file_path).split('.')[0].upper()

            # --- 2. Handle variant data based on gene name ---
            if 'APOE' in gene_name:
                required_snps = ['rs429358_C', 'rs7412_T']
                if all(snp in df.columns for snp in required_snps):
                    # Create a new DataFrame with just the APOE genotype
                    variant_df = pd.DataFrame(index=df.index)
                    variant_df['APOE_Genotype'] = df.apply(determine_apoe_genotype, axis=1)
                else:
                    print(f"Warning: APOE file {file_path} missing required SNPs. Skipping variants.")
                    continue
            else:
                # Identify variant columns (not FID index, not common metadata)
                variant_cols = [col for col in df.columns if col not in COMMON_COLS]
                # Create a new DataFrame with only the namespaced variant columns
                variant_df = df[variant_cols].rename(columns={
                    col: f"{gene_name}_{col}" for col in variant_cols
                })
            
            variant_dfs.append(variant_df)
            
            # --- 3. Handle common metadata ---
            # Store the common columns separately for later consolidation
            present_common_cols = [col for col in COMMON_COLS if col in df.columns]
            common_dfs.append(df[present_common_cols])

        except Exception as e:
            print(f"An error occurred processing {file_path}: {e}")

    if not variant_dfs:
        return None

    # --- 4. Merge all variant data together ---
    merged_variants = pd.concat(variant_dfs, axis=1, join='outer')

    # --- 5. Consolidate and merge common data ---
    if common_dfs:
        # Combine all common data and keep the first entry for each individual
        all_common = pd.concat(common_dfs)
        master_common = all_common[~all_common.index.duplicated(keep='first')]
        
        # Join the consolidated metadata to the main variant data
        final_df = master_common.join(merged_variants, how='outer')
    else:
        final_df = merged_variants

    return final_df


