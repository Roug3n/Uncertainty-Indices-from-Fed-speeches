'''
This script compares how our indices, created from Fed speeches, compare to other uncertainty indices (EPU, MPU, VIX)
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#######################################################
### 1. Load EPU data
def load_epu(path="data/epu.xlsx"):
    df = pd.read_excel(path)

    # This eliminates non-data rows that appear
    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["Month"], errors="coerce").notna()]
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    # Creates a proper date column
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))

    # Keeps only what we need
    df = df[["date", "News_Based_Policy_Uncert_Index"]]

    # Rename for clarity
    df = df.rename(columns={
        "News_Based_Policy_Uncert_Index": "epu"
    })
    df = df.sort_values("date")

    return df


#######################################################
### 2. Load MPU data
def load_mpu(path="data/mpu.xlsx"):

    df = pd.read_excel(path)

    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    df = df[pd.to_numeric(df["Month"], errors="coerce").notna()]
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    df["date"] = pd.to_datetime(dict(year=df["Year"],month=df["Month"],day=1))

    df = df[["date","BBD MPU Index Based on Access World News"]]

    df = df.rename(
        columns={
            "BBD MPU Index Based on Access World News":
            "mpu"
        }
    )
    return df.sort_values("date")


#######################################################
### 3. Load and clean VIX data
def load_vix(path="data/vix.xlsx"):
    df = pd.read_excel(path, sheet_name="Daily, Close")

    # Rename columns for convenience
    df = df.rename(columns={
        "observation_date": "date",
        "VIXCLS": "vix"
    })

    # Converts date
    df["date"] = pd.to_datetime(df["date"])

    # Fixes decimal format
    df["vix"] = df["vix"].astype(str).str.replace(",", ".")
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.dropna(subset=["vix"])

    return df

# Agregate VIX data by month

def monthly_vix(df):
    df["month"] = df["date"].dt.to_period("M")

    vix_monthly = (
        df.groupby("month")["vix"]
        .mean()
        .reset_index()
    )

    vix_monthly["date"] = vix_monthly["month"].dt.to_timestamp()

    vix_monthly = vix_monthly[["date", "vix"]]

    return vix_monthly


#######################################################
### 4. Load our Fed indeces

def load_fed_index(path, column_name):

    df = pd.read_pickle(path)

    if "month" in df.columns:

        df["date"] = pd.PeriodIndex(
            df["month"],
            freq="M"
        ).to_timestamp()

    else:

        df["date"] = df.index.to_timestamp()

    df = (
        df[["date", "uncertainty_index"]]
        .rename(
            columns={
                "uncertainty_index": column_name
            }
        )
        .sort_values("date")
    )

    return df


#######################################################
### 5. Merge indices

'''
Creates a single monthly dataset containing:
 - External benchmarks
 - All Fed-based uncertainty indices

This dataset becomes the foundation for the
benchmarking and validation exercises.
'''

def merge_all(indices, epu, mpu, vix):

    df = epu.copy()

    df = df.merge(
        mpu,
        on="date",
        how="left"
    )

    df = df.merge(
        vix,
        on="date",
        how="left"
    )

    for index_df in indices:
        df = df.merge(
            index_df,
            on="date",
            how="left"
        )

    return df.sort_values("date")


#######################################################
### 6. Standarize external indices

def standardize(df):
    for col in ["epu", "mpu", "vix"]:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()

    return df


#######################################################
### 7. Store all fed indices names

def fed_columns():

    return [

        "fed_base",
        "fed_chair",
        "fed_tfidf",

        "fed_embedding",
        "fed_embedding_chair",

        "fed_embedding_pure",
        "fed_embedding_chair_pure",

        "fed_finbert",
        "fed_finbert_chair",

        "fed_finbert_pure",
        "fed_finbert_chair_pure",

        "fed_finbert_A",
        "fed_finbert_chair_A",

        "fed_finbert_B",
        "fed_finbert_chair_B",

        "fed_finbert_C",
        "fed_finbert_chair_C",

        "fed_finbert_D",
        "fed_finbert_chair_D"
    ]


#######################################################
### 8. Correlation benchmark function

'''
Computes simple Pearson correlations between each
Fed uncertainty index and a selected benchmark.

Indices are ranked according to the absolute value
of the correlation coefficient.
'''

def benchmark_against(
    df,
    benchmark_col
):

    results = []

    for col in fed_columns():

        corr = (
            df[
                [
                    col,
                    benchmark_col
                ]
            ]
            .corr()
            .iloc[0,1]
        )

        results.append(
            {
                "index": col,
                "corr": corr,
            }
        )

    results = pd.DataFrame(results)

    results = results.sort_values(
        "corr",
        key=abs,
        ascending=False
    )

    results["rank"] = range(
        1,
        len(results) + 1
    )

    return results
    


#######################################################
### 9. Combined ranking

'''
Aggregates rankings obtained against EPU, MPU and
VIX into a single average ranking.

This provides an overall measure of how closely
each Fed uncertainty index tracks established
uncertainty proxies
'''

def combined_ranking(
    epu_table,
    mpu_table,
    vix_table
):

    combined = (
        epu_table[["index", "corr", "rank"]]
        .rename(
            columns={
                "corr": "corr_epu",
                "rank": "rank_epu"
            }
        )
        .merge(
            mpu_table[["index", "corr", "rank"]].rename(
                columns={
                    "corr": "corr_mpu",
                    "rank": "rank_mpu"
                }
            ),
            on="index"
        )
        .merge(
            vix_table[["index", "corr", "rank"]].rename(
                columns={
                    "corr": "corr_vix",
                    "rank": "rank_vix"
                }
            ),
            on="index"
        )
    )

    combined["average_rank"] = (
        combined[
            ["rank_epu",
             "rank_mpu",
             "rank_vix"]
        ]
        .mean(axis=1)
    )

    return (
        combined
        .sort_values(
            "average_rank",
            ascending=True
        )
    )


#######################################################
### 10. Correlation matrix for Fed indices and Heatmap

def fed_correlation_matrix(df):

    corr = (
        df[
            fed_columns()
        ]
        .corr()
    )

    return corr


def plot_heatmap(corr):

    plt.figure(
        figsize=(14,12)
    )

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Correlation Between Fed Indices"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/fed_correlations_heatmap.png",
        dpi=300
    )

    plt.show()


#######################################################
### MAIN

if __name__ == "__main__":

    # Load datasets
    fed_base = load_fed_index(
        "data/uncertainty_index.pkl",
        "fed_base"
    )

    fed_chair = load_fed_index(
        "data/uncertainty_index_chair.pkl",
        "fed_chair"
    )


    fed_tfidf = load_fed_index(
        "data/tfidf_uncertainty_index.pkl",
        "fed_tfidf"
    )

    fed_embedding = load_fed_index(
        "data/embedding_index.pkl",
        "fed_embedding"
    )

    fed_embedding_chair = load_fed_index(
        "data/embedding_chair_index.pkl",
        "fed_embedding_chair"
    )

    fed_embedding_pure = load_fed_index(
        "data/embedding_pure_index.pkl",
        "fed_embedding_pure"
    )

    fed_embedding_chair_pure = load_fed_index(
        "data/embedding_chair_pure_index.pkl",
        "fed_embedding_chair_pure"
    )

    fed_finbert = load_fed_index(
        "data/finbert_encoder_index.pkl",
        "fed_finbert"
    )

    fed_finbert_chair = load_fed_index(
        "data/finbert_encoder_chair_index.pkl",
        "fed_finbert_chair"
    )

    fed_finbert_pure = load_fed_index(
        "data/finbert_encoder_pure_index.pkl",
        "fed_finbert_pure"
    )

    fed_finbert_chair_pure = load_fed_index(
        "data/finbert_encoder_chair_pure_index.pkl",
        "fed_finbert_chair_pure"
    )
    
    fed_finbert_A = load_fed_index(
        "data/finbert_index_A.pkl",
        "fed_finbert_A"
    )

    fed_finbert_chair_A = load_fed_index(
        "data/finbert_chair_index_A.pkl",
        "fed_finbert_chair_A"
    )
    
    fed_finbert_B = load_fed_index(
        "data/finbert_index_B.pkl",
        "fed_finbert_B"
    )

    fed_finbert_chair_B = load_fed_index(
        "data/finbert_chair_index_B.pkl",
        "fed_finbert_chair_B"
    )

    fed_finbert_C = load_fed_index(
        "data/finbert_index_C.pkl",
        "fed_finbert_C"
    )

    fed_finbert_chair_C = load_fed_index(
        "data/finbert_chair_index_C.pkl",
        "fed_finbert_chair_C"
    )

    fed_finbert_D = load_fed_index(
        "data/finbert_index_D.pkl",
        "fed_finbert_D"
    )

    fed_finbert_chair_D = load_fed_index(
        "data/finbert_chair_index_D.pkl",
        "fed_finbert_chair_D"
    )

    epu = load_epu()

    mpu = load_mpu()

    vix = monthly_vix(
        load_vix()
    )

    df = merge_all(
        [
            fed_base,
            fed_chair,
            fed_tfidf,
            fed_embedding,
            fed_embedding_chair,
            fed_embedding_pure,
            fed_embedding_chair_pure,
            fed_finbert,
            fed_finbert_chair,
            fed_finbert_pure,
            fed_finbert_chair_pure,
            fed_finbert_A,
            fed_finbert_chair_A,
            fed_finbert_B,
            fed_finbert_chair_B,
            fed_finbert_C,
            fed_finbert_chair_C,
            fed_finbert_D,
            fed_finbert_chair_D
        ],
        epu,
        mpu,
        vix
    )

    df = df[
        df["date"] >= "2006-01-01"
    ].copy()

    df = standardize(df)

    df.to_pickle(
        "data/benchmark_dataset.pkl"
    )


    ## EPU

    epu_table = benchmark_against(
        df,
        "epu_z"
    )

    print("\nEPU RANKING\n")
    print(epu_table)


    ## MPU

    mpu_table = benchmark_against(
        df,
        "mpu_z"
    )

    print("\nMPU RANKING\n")
    print(mpu_table)


    ## VIX

    vix_table = benchmark_against(
        df,
        "vix_z"
    )

    print("\nVIX RANKING\n")
    print(vix_table)


    ## COMBINED

    combined = combined_ranking(
        epu_table,
        mpu_table,
        vix_table
    )

    print("\nCOMBINED RANKING\n")
    print(combined)

    ###################################################
    ### FED MATRIX

    corr = fed_correlation_matrix(df)

    print("\nFED CORRELATION MATRIX\n")
    print(corr.round(3))

    plot_heatmap(corr)

    print("\nDone.")