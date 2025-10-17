#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd


def parse_year_from_virus(virus_name: str):
    """Extract year as int from strings like 'A/Brisbane/24/2008'.

    Returns None if parsing fails.
    """
    if not isinstance(virus_name, str):
        return None
    parts = virus_name.strip().split('/')
    if not parts:
        return None
    last = parts[-1].strip()
    # Some entries might have trailing annotations; keep only digits at end
    # but primarily expect a 4-digit year
    try:
        year = int(last)
        if 1000 <= year <= 9999:
            return year
        return None
    except ValueError:
        # Try to extract last 4-digit number if present
        import re
        m = re.search(r"(\d{4})$", last)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None


def generate_splits(df: pd.DataFrame, window: int = 5):
    """Yield tuples of (train_years_list, test_year) based on available years.

    df must contain a 'year' column with integers.
    """
    years = sorted(y for y in df['year'].dropna().unique().tolist() if isinstance(y, (int, float)))
    years = [int(y) for y in years]
    for i in range(window, len(years)):
        train_years = years[i - window:i]
        test_year = years[i]
        yield train_years, test_year


def main():
    # 直接在脚本中指定输入/输出与窗口大小
    input_path = Path("data/prd/2003-2025/final_sequences.csv")
    out_base = Path("time_series_data/data/")
    window = 10

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_base.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)

    if 'short_name' not in df.columns:
        print("Error: input CSV must contain a 'short_name' column.", file=sys.stderr)
        sys.exit(1)

    # 从 Virus 列解析年份（CSV 仅包含 Virus 列，无 year 列）
    df['year'] = df['short_name'].apply(parse_year_from_virus)

    # Drop rows where year could not be parsed
    before = len(df)
    df = df.dropna(subset=['year']).copy()
    df['year'] = df['year'].astype(int)
    after = len(df)
    if after < before:
        print(f"Warning: dropped {before - after} rows with unparseable year.")

    if df.empty:
        print("No data with valid year available after parsing.", file=sys.stderr)
        sys.exit(1)

    # Generate and write splits
    total_folds = 0
    for train_years, test_year in generate_splits(df, window=window):
        train_mask = df['year'].isin(train_years)
        test_mask = df['year'] == test_year

        df_train = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()

        if df_train.empty or df_test.empty:
            # Skip folds without data
            print(f"Skip fold train {train_years[0]}-{train_years[-1]} -> test {test_year} due to empty split.")
            continue

        # Ensure training set covers full `window` distinct years; otherwise skip
        train_years_present = df_train['year'].nunique()
        if train_years_present < window:
            print(
                f"Skip fold train {train_years[0]}-{train_years[-1]} -> test {test_year} "
                f"due to insufficient train years: {train_years_present} < {window}."
            )
            continue

        fold_dir = out_base / f"{test_year}"
        train_dir = fold_dir / "train"
        test_dir = fold_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        df_train = df_train[['short_name','HA1_sequence']]
        df_test = df_test[['short_name','HA1_sequence']]

        # Write CSVs
        train_path = train_dir / "train.csv"
        test_path = test_dir / "test.csv"
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        total_folds += 1
        print(
            f"Wrote fold: train {train_years[0]}-{train_years[-1]} -> test {test_year} | "
            f"{len(df_train)} train rows, {len(df_test)} test rows \n"
            f"  - {train_path}\n  - {test_path}"
        )

    if total_folds == 0:
        print("No valid folds produced. Check year coverage and window size.", file=sys.stderr)
        sys.exit(2)
    else:
        print(f"Done. Produced {total_folds} folds under: {out_base}")


if __name__ == "__main__":
    main()
