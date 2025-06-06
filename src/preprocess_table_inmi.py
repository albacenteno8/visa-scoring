import pandas as pd
import re
from collections import defaultdict

def unify_exclusive_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    groups = defaultdict(list)

    for col in df.columns:
        base = re.sub(r'_[a-z]{2,5}$', '', col)
        groups[base].append(col)

    for base, cols in groups.items():
        if len(cols) < 2:
            continue

        sub_df = df[cols]

        if (sub_df.notna().sum(axis=1) == 1).all():
            unified_col = base + "_x"
            df[unified_col] = sub_df.bfill(axis=1).iloc[:, 0]
            df.drop(columns=cols, inplace=True)
            print(f"Unified: {cols} -> {unified_col}")

    return df

def print_columns_with_mixed_dtypes(df: pd.DataFrame):
    """
    Prints columns in the DataFrame that contain mixed data types (e.g., strings and numbers).
    """
    mixed_cols = []

    for col in df.columns:
        types_in_col = set(type(val) for val in df[col].dropna())
        if len(types_in_col) > 1:
            mixed_cols.append((col, types_in_col))

    if mixed_cols:
        print("Columns with mixed data types:")
        for col, types_found in mixed_cols:
            print(f"  - {col}: {types_found}")
    else:
        print("No columns with mixed data types found.")


def preprocess_applications_df(df_cases: pd.DataFrame) -> pd.DataFrame:
    print("Step 1: Dropping all-null rows...")
    cleaned_df = df_cases.dropna(how='all')
    print(f"Remaining rows: {cleaned_df.shape[0]}")

    print("Step 2: Dropping rows where all key counts are zero...")
    remove_index = cleaned_df[(cleaned_df['other_family_members_count']==0) &
                              (cleaned_df['country_visited_count']==0) &
                              (cleaned_df['employment_history_details_count']==0)].index
    cleaned_df_2 = cleaned_df.drop(index=remove_index.tolist())
    print(f"Remaining rows after drop: {cleaned_df_2.shape[0]}")

    print("Step 3: Keeping rows with non-null 'current_location_ac_cl'...")
    cleaned_df_2 = cleaned_df_2[cleaned_df_2.current_location_ac_cl.notnull()]
    print(f"Remaining rows: {cleaned_df_2.shape[0]}")

    print("Step 4: Dropping columns that are fully NaN...")
    df_nans = pd.DataFrame(cleaned_df_2.isna().sum()).reset_index()
    df_nans.columns = ['cols_n','numbers']
    delete_cols = df_nans[df_nans.numbers >= cleaned_df_2.shape[0]].cols_n.unique().tolist()
    cleaned_df_3 = cleaned_df_2.drop(columns=delete_cols + [''], errors='ignore')
    print(f"Remaining columns: {cleaned_df_3.shape[1]}")

    print("Step 5: Unifying mutually exclusive columns...")
    cleaned_df_4 = unify_exclusive_columns(cleaned_df_3)

    print("Step 6: Summary of unique values and NaNs...")
    df = cleaned_df_4.copy()
    summary = pd.DataFrame({
        "number_unique": df.nunique(),
        "number_nan": df.isna().sum(),
        "dtype_df": df.dtypes
    })

    print("Step 7: Cleaning and converting 'generated' field...")
    df['generated'] = df['generated'].apply(lambda x: x.split(',')[1] if isinstance(x, str) and ',' in x else x)
    df['generated_date'] = pd.to_datetime(df['generated'], dayfirst=True, errors='coerce')

    print("Step 8: Dropping object columns with >80% unique values...")
    list_columns_remove = summary[(summary.number_unique > df.shape[0] * 0.8) & 
                                  (summary.dtype_df == 'object')].index.tolist()
    list_columns_remove_f = [i for i in list_columns_remove if 'date' not in i]
    df_f = df.drop(columns=list_columns_remove_f)
    print(f"Remaining columns: {df_f.shape[1]}")

    print("Step 9: Dropping date columns with >80% missing values...")
    date_cols = [c for c in df_f.columns if 'date' in c.lower()]
    miss_pct = df_f[date_cols].isna().mean()
    to_drop = miss_pct[miss_pct > 0.8].index
    df_f = df_f.drop(columns=to_drop)
    print(f"Remaining columns after dropping sparse dates: {df_f.shape[1]}")

    print("Step 10: Filtering incorrect date entries...")
    df_f = df_f[df_f['date_of_issue_nic'] != 'Place of birth']
    df_f = df_f[df_f['date_of_expiry_nic'] != 'Place of birth']

    print("Step 11: Dropping explicit date ranges...")
    df_f = df_f.drop(columns=['date_from_e_pd','date_to_e_pd','date_from_cv','date_to_cv',
                              'date_from_ehd','date_to_ehd','date_from_e_scd','date_to_e_scd'], errors='ignore')

    print("Step 12: Converting remaining date columns to time deltas...")
    date_cols = [c for c in df_f.columns if 'date' in c.lower() and c != 'generated_date']
    for col in date_cols:
        df_f[col] = pd.to_datetime(df_f[col], dayfirst=True, errors='coerce')
        df_f[col] = (df_f['generated_date'] - df_f[col]).dt.days

    print("Step 13: Adding 'days_from_today' field...")
    today = pd.Timestamp.today().normalize()
    df_f['days_from_today'] = (today - df_f['generated_date']).dt.days

    df_f = df_f.drop(columns=['generated_date'], errors='ignore')

    print("Step 15: Removing postal code, mobile phone and passport number columns...")
    mobile_phone = [i for i in df_f.columns if 'phone' in i]
    postal_code = [i for i in df_f.columns if 'postal' in i]
    passport = [i for i in df_f.columns.tolist() if 'passport_number' in i]
    other_columns = ['identification_number_oid','give_details_nic_he']
    address_cols = [i for i in df_f.columns if 'address' in i]
    df_f = df_f.drop(columns = postal_code + mobile_phone + passport + address_cols + other_columns)

    print_columns_with_mixed_dtypes(df_f)

    print(f"Final shape of the preprocessed dataset: {df_f.shape}")
    
    return df_f
