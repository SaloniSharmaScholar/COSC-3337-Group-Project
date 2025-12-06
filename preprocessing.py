import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def load_raw_df() -> pd.DataFrame:
    data = []
    for i in range(4):
        file_path = DATA_DIR / f"dblp-ref-{i}.json"
        print(f"Loading {file_path}...")
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    print("Loaded papers:", len(df))
    return df


def main():
    df = load_raw_df()

    #basic cleaning
    df = df[df["title"]. notna() & df["venue"].notna() & df["year"].notna()]
    df["Abstract"] = df["abstract"].fillna("")
    df["author_count"] = df["authors"].apply(
        lambda x: len(x) if isinstance(x,list) else 0
    )
    df["text"] = df["title"] + " " + df["Abstract"]
    print("After cleaning:", len(df))

#Classification dataset
    venue_counts = df["venue"].value_counts()
    top_venues = venue_counts[venue_counts >= 1000].index  # threshold can be changed
    df_class = df[df["venue"].isin(top_venues)].copy()
    df_class.to_parquet(DATA_DIR / "classification_df.parquet")
    print("Classification rows:", len(df_class), " | venues:", len(top_venues))

#Regression dataset
    df_reg = df[df["n_citation"].notna()].copy()
    df_reg["n_citation"] = df_reg["n_citation"].clip(upper=df_reg["n_citation"].quantile(0.99))
    df_reg.to_parquet(DATA_DIR / "regression_df.parquet")
    print("Regression rows:", len(df_reg))

if __name__ == "__main__":
    main()