""" Module with helper functions for HPC batch processing. """

import os
from pathlib import Path
import pandas as pd
import numpy as np

def get_input_df_for_rerun(df_in_full_path:str, df_out_1_path:str, id_col:str='id') -> pd.DataFrame:
    """ Splits the full input dataframe into two parts:
    - df_in_1: Contains all rows, which are already found in df_out_1_path
    - df_in_2: All other rows, which are not found in df_out_1_path

    Args:
        df_in_full_path (str): _path to full input dataframe
        df_out_1_path (str): _path to first output dataframe
        id_col (str): name of the id column. Defaults to 'id'

    Returns:
        pd.DataFrame: df_in_2: DataFrame containing all rows not found in df_out_1_path
    """
    # Load df_in_full:
    df_in_full = pd.read_csv(df_in_full_path)

    # If no output file, nothing to exclude
    if not os.path.isfile(df_out_1_path):
        print(f"Output file '{df_out_1_path}' not found. Returning full input.", flush=True)
        return df_in_full

    df_out_1 = pd.read_csv(df_out_1_path)
    if id_col not in df_out_1.columns:
        raise ValueError(f"Column '{id_col}' missing in {df_out_1_path}")

    # Normalize id dtype to avoid 1 vs "1" mismatches
    df_in_full[id_col] = df_in_full[id_col].astype("int")
    df_out_1[id_col] = df_out_1[id_col].astype("int")

    # Check that all columns in input are present in output
    missing_cols = set(df_in_full.columns) - set(df_out_1.columns)
    if missing_cols:
        raise ValueError(
            f"Columns missing in '{df_out_1_path}' that are present in '{df_in_full_path}': "
            f"{sorted(missing_cols)}")

    # Guard against duplicate ids in the output file (ambiguous)
    dup_ids = df_out_1[df_out_1.duplicated(id_col, keep=False)][id_col].unique().tolist()
    if dup_ids:
        raise ValueError(
            f"Duplicate ids in {df_out_1_path}: {dup_ids[:5]}{'...' if len(dup_ids) > 5 else ''}")

    # Ensure all output ids are present in the full input
    ids_in = set(df_in_full[id_col].dropna())
    ids_out = set(df_out_1[id_col].dropna())
    missing = ids_out - ids_in
    if missing:
        raise ValueError(
            f"{len(missing)} id(s) in {df_out_1_path} not present in {df_in_full_path}. "
            f"Examples: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    # Compute the “to-rerun” set
    df_in_2 = df_in_full[~df_in_full[id_col].isin(ids_out)].copy()

    # Validate output:
    # Align by id
    take_ids_sorted = sorted(ids_out)

    # Compare only columns that are shared AND not the id_col
    common_cols = sorted(set(df_out_1.columns).intersection(df_in_full.columns) - {id_col})

    left = (
        df_in_full.set_index(id_col)
        .loc[take_ids_sorted, common_cols]
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    right = (
        df_out_1.set_index(id_col)
        .loc[take_ids_sorted, common_cols]
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    # Index (ids) should match exactly
    if not left.index.equals(right.index):
        raise ValueError("Aligned id indexes differ between input and output.")

    # Strict equality on values (NaNs in same positions count as equal)
    try:
        validate_aligned_equal_by_id(df_in_full, df_out_1, id_col="id", rtol=1e-6, atol=6e-6)
    except AssertionError as e:
        raise ValueError(f"Output rows diverge from input for matched ids: {e}") from None

    print(
        f"Rerun input:{len(df_in_2)} rows (out of {len(df_in_full)}) not found in {df_out_1_path}.",
        flush=True
    )
    return df_in_2

def validate_aligned_equal_by_id(
        df_in_full: pd.DataFrame, df_out_1: pd.DataFrame, id_col: str = "id", rtol: float = 1e-6,
        atol: float = 1e-9,) -> None:
    """ Validate that two dataframes are equal on all columns except id_col,
    after aligning them by id_col.

    Args:
        df_in_full (pd.DataFrame): full input dataframe
        df_out_1 (pd.DataFrame): output dataframe to compare against input
        id_col (str, optional): name of the id column. Defaults to "id".
        rtol (float, optional): _relative tolerance for numeric comparisons. Defaults to 1e-6.
        atol (float, optional): _absolute tolerance for numeric comparisons. Defaults to 1e-9.
    """
    # Figure out common, non-id columns
    common_cols = sorted(set(df_out_1.columns).intersection(df_in_full.columns) - {id_col})

    # Align both frames by id on the same rows/cols
    ids = sorted(set(df_out_1[id_col]).intersection(df_in_full[id_col]))
    left = (df_in_full.set_index(id_col).loc[ids,common_cols]).sort_index(axis=0).sort_index(axis=1)
    right = (df_out_1 .set_index(id_col).loc[ids,common_cols]).sort_index(axis=0).sort_index(axis=1)

    # 1) Exact check for non-numeric columns
    nonnum = left.select_dtypes(exclude="number").columns.tolist()
    for c in nonnum:
        try:
            pd.testing.assert_series_equal(left[c], right[c], check_dtype=True, check_exact=True)
        except AssertionError as e:
            raise ValueError(f"Non-numeric column {c!r} differs: {e}") from None

    # 2) Tolerant check for numeric columns
    num = left.select_dtypes(include="number").columns.tolist()
    if num:
        l = left[num].to_numpy()
        r = right[num].to_numpy()
        close = np.isclose(l, r, rtol=rtol, atol=atol, equal_nan=True)
        if not np.all(close):
            # Build a concise diagnostic
            bad_rows, bad_cols = np.where(~close)
            first_r, first_c = bad_rows[0], bad_cols[0]
            col_name = num[first_c]
            row_id = left.index[first_r]
            max_abs = np.nanmax(np.abs(l - r))
            max_rel = np.nanmax(np.abs((l - r) / np.maximum(np.abs(r), np.finfo(float).eps)))
            raise ValueError(
                f"Numeric differences beyond tolerances (rtol={rtol}, atol={atol}). "
                f"Example at id={row_id!r}, column {col_name!r}: "
                f"left={l[first_r, first_c]!r}, right={r[first_r, first_c]!r}. "
                f"max_abs_diff={max_abs:.6g}, max_rel_diff={max_rel:.6g}"
            )

def combine_batch_csv_files(design_df:pd.DataFrame, output_path:str, batch_size:int,
                            data_export_path:str=None) -> pd.DataFrame:
    """ Combine the batch csv files into a single dataframe.

    Args:
        design_df (pd.DataFrame): The original design dataframe without the results.
        output_path (str): The path to the output csv files.
        batch_size (int): The batch size used for the optimization.
        data_export_path (str, optional): The path to export the combined dataframe.
            Defaults to None.
    
    Raises:
        FileNotFoundError: If the output path does not exist.
        FileNotFoundError: If a batch file does not exist.
        ValueError: If the dataframes do not have the same columns.
    """

    output_dir = Path(output_path).resolve()

    if not output_dir.exists():
        raise FileNotFoundError(f"Output path {output_dir} does not exist.")

    batch_df_list = []

    for idx_start in range(0, len(design_df), batch_size):
        idx_stop = min(idx_start + batch_size, len(design_df)) - 1

        # Construct the file path for each batch
        output_file = output_dir / f"idx{idx_start}-{idx_stop}.csv"

        if not output_file.exists():
            raise FileNotFoundError(f"Batch file {output_file} does not exist.")

        batch_df_list.append(pd.read_csv(output_file, index_col=0))

    # Ensure all dataframes have the same columns
    columns = batch_df_list[0].columns
    for df in batch_df_list:
        if not df.columns.equals(columns):
            raise ValueError("DataFrames do not have the same columns.")

    # Concatenate all batch dataframes
    df_out = pd.concat(batch_df_list, axis=0, ignore_index=False)

    # Ensure all indices are unique
    if not df_out.index.is_unique:
        raise ValueError("Something went wrong, the indices are not unique.")

    if data_export_path:
        export_file = Path(data_export_path).with_suffix('.csv')
        print(f"Exporting combined DataFrame to {export_file}")
        df_out.to_csv(export_file, index=False)

    return df_out
