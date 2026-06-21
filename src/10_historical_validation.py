'''
This script identifies the months classified as the highest-uncertainty periods by each 
index and studies the degree of agreement across methodologies.

The underlying idea is that genuinely important uncertainty episodes 
should emerge consistently across multiple independent approaches.
'''

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Number of highest-uncertainty months retained for each index.
TOP_N = 10

#######################################################
### 1. Load index

def load_index(path):

    df = pd.read_pickle(path)

    if "month" in df.columns:

        df = df.set_index("month")

    return df


#######################################################
### 2. Build top-month matrix

'''
Extracts top uncertainty months

For each index, select the TOP_N months with the
highest uncertainty values.

These rankings constitute the basis of the
historical validation exercise
'''

def build_top_months(indices):

    rows = []

    for name, df in indices.items():

        top_months = (
            df.sort_values(
                "uncertainty_index",
                ascending=False
            )
            .head(TOP_N)
            .index.astype(str)
            .tolist()
        )

        row = {"index": name}

        for i, month in enumerate(top_months, start=1):

            row[f"top{i}"] = month

        rows.append(row)

    return pd.DataFrame(rows)


#######################################################
### 3. Frequency table

def build_frequency_table(top_months_table):

    all_months = []

    for i in range(1, TOP_N + 1):

        all_months.extend(
            top_months_table[f"top{i}"]
            .dropna()
            .tolist()
        )

    freq = Counter(all_months)

    freq_table = (
        pd.DataFrame(
            freq.items(),
            columns=[
                "month",
                "frequency"
            ]
        )
        .sort_values(
            "frequency",
            ascending=False
        )
        .reset_index(drop=True)
    )

    n_indices = len(top_months_table)

    freq_table["share"] = (
        100
        * freq_table["frequency"]
        / n_indices
    )

    return freq_table.sort_values(
        "frequency",
        ascending=False
    ).reset_index(drop=True)


#######################################################
### 4. Consensus frequency chart

def plot_month_frequency(frequency):

    plot_df = frequency.head(12)

    plt.figure(figsize=(12,6))

    plt.bar(
        plot_df["month"].astype(str),
        plot_df["share"]
    )

    plt.title(
        "Months Appearing Most Frequently in Top-10 Lists"
    )

    plt.ylabel(
        "Percentage of Indices"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        "figures/top_month_frequency.png",
        dpi=300
    )

    plt.show()


#######################################################
### 4. Consensus frequency chart time sorted

def plot_sorted_month_frequency(frequency):

    freq_time = frequency.copy()

    freq_time["month"] = pd.to_datetime(
        freq_time["month"]
    )

    freq_time = freq_time.sort_values(
        "month"
    )

    plt.figure(figsize=(14,6))

    plt.bar(
        freq_time["month"],
        freq_time["share"],
        width=25
    )

    plt.title(
        "Frequency of Appearance in Top-10 Lists"
    )

    plt.ylabel(
        "Percentage of Indices"
    )

    plt.xlabel(
        "Month"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        "figures/top_month_frequency_timeline.png",
        dpi=300
    )

    plt.show()


#######################################################
### 5. Heatmap of top months

def plot_top_month_heatmap(top_months):

    heatmap_df = top_months.copy()

    month_cols = [
        col for col in heatmap_df.columns
        if col.startswith("top")
    ]

    for col in month_cols:

        heatmap_df[col] = (
            pd.to_datetime(
                heatmap_df[col]
            )
            .dt.year
        )

    heatmap_df = heatmap_df.set_index(
        "index"
    )

    plt.figure(figsize=(14,10))

    sns.heatmap(
        heatmap_df,
        cmap="YlOrRd",
        linewidths=0.5
    )

    plt.title(
        "Top Months Selected by Each Index (Color = Year)"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/top_months_heatmap.png",
        dpi=300
    )

    plt.show()


#######################################################
### 6. Binary consensus heatmap

def plot_binary_consensus_heatmap(
        top_months,
        start="2006-01",
        end="2026-12"
):

    timeline = pd.period_range(
        start=start,
        end=end,
        freq="M"
    ).astype(str)

    heatmap_df = pd.DataFrame(
        0,
        index=top_months["index"],
        columns=timeline
    )

    month_cols = [
        col for col in top_months.columns
        if col.startswith("top")
    ]

    for _, row in top_months.iterrows():

        idx_name = row["index"]

        for col in month_cols:

            month = row[col]

            if pd.notna(month):

                heatmap_df.loc[
                    idx_name,
                    month
                ] = 1

    plt.figure(figsize=(20,8))

    sns.heatmap(
        heatmap_df,
        cmap="Reds",
        cbar=False,
        linewidths=0
    )

    plt.title(
        "Consensus Map of Top-10 Uncertainty Months"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Index"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/binary_consensus_heatmap.png",
        dpi=300
    )

    plt.show()


#######################################################
### MAIN

if __name__ == "__main__":

    indices = {

        "fed_base":
            load_index(
                "data/uncertainty_index.pkl"
            ),

        "fed_chair":
            load_index(
                "data/uncertainty_index_chair.pkl"
            ),

        "fed_tfidf":
            load_index(
                "data/tfidf_uncertainty_index.pkl"
            ),

        "fed_embedding":
            load_index(
                "data/embedding_index.pkl"
            ),

        "fed_embedding_chair":
            load_index(
                "data/embedding_chair_index.pkl"
            ),

        "fed_embedding_pure":
            load_index(
                "data/embedding_pure_index.pkl"
            ),

        "fed_embedding_chair_pure":
            load_index(
                "data/embedding_chair_pure_index.pkl"
            ),

        "fed_finbert":
            load_index(
                "data/finbert_encoder_index.pkl"
            ),

        "fed_finbert_chair":
            load_index(
                "data/finbert_encoder_chair_index.pkl"
            ),

        "fed_finbert_pure":
            load_index(
                "data/finbert_encoder_pure_index.pkl"
            ),

        "fed_finbert_chair_pure":
            load_index(
                "data/finbert_encoder_chair_pure_index.pkl"
            ),

        "fed_finbert_A":
            load_index(
                "data/finbert_index_A.pkl"
            ),

        "fed_finbert_chair_A":
            load_index(
                "data/finbert_chair_index_A.pkl"
            ),

        "fed_finbert_B":
            load_index(
                "data/finbert_index_B.pkl"
            ),

        "fed_finbert_chair_B":
            load_index(
                "data/finbert_chair_index_B.pkl"
            ),

        "fed_finbert_C":
            load_index(
                "data/finbert_index_C.pkl"
            ),

        "fed_finbert_chair_C":
            load_index(
                "data/finbert_chair_index_C.pkl"
            ),

        "fed_finbert_D":
            load_index(
                "data/finbert_index_D.pkl"
            ),

        "fed_finbert_chair_D":
            load_index(
                "data/finbert_chair_index_D.pkl"
            )
    }

    ### Top months matrix

    top_months = build_top_months(indices)

    print("\nTOP MONTHS BY INDEX\n")

    print(
        top_months.to_string(
            index=False
        )
    )

    ### Frequency table

    frequency_table = build_frequency_table(
        top_months
    )

    print(
        "\nMONTHS APPEARING MOST OFTEN IN TOP-10 LISTS\n"
    )

    print(
        frequency_table.head(12)
        .round({"share":1})
        .to_string(index=False)
    )

    ### Save

    top_months.to_pickle(
        "data/historical_validation_top_months.pkl"
    )

    frequency_table.to_pickle(
        "data/historical_validation_frequency.pkl"
    )

    ### Plots

    plot_month_frequency(frequency_table)

    plot_sorted_month_frequency(frequency_table)

    plot_top_month_heatmap(top_months)

    plot_binary_consensus_heatmap(top_months)

    print("\nDone.")