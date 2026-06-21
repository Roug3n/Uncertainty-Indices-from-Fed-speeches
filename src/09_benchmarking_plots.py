'''
This script creates all visualizations used to evaluate and compare uncertainty indices.

The visualizations cover:
 - Methodological comparisons across NLP approaches
 - Chair vs full-sample differences
 - Benchmark comparisons (EPU, MPU, VIX)
 - Standardized (z-score) comparisons
'''


import pandas as pd
import matplotlib.pyplot as plt


#######################################################
### 1. Load data

def load_data():

    df = pd.read_pickle(
        "data/benchmark_dataset.pkl"
    )

    return df



#######################################################
### 2. Methodological comparison

def plot_methodology(df):

    plt.figure(figsize=(14,8))

    cols = [
        "fed_base",
        "fed_tfidf",
        "fed_embedding",
        "fed_finbert",
    ]

    for col in cols:
        plt.plot(
            df["date"],
            df[col],
            label=col
        )

    plt.title(
        "Fed Uncertainty Indices: Methodological Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "figures/methodology_comparison.png",
        dpi=300
    )

    plt.show()


#######################################################
### 4. Chair vs full sample

def plot_chair_vs_full(
        df,
        full_col,
        chair_col,
        title,
        filename
):

    plt.figure(figsize=(12,6))

    plt.plot(
        df["date"],
        df[full_col],
        label=full_col
    )

    plt.plot(
        df["date"],
        df[chair_col],
        label=chair_col
    )

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"figures/{filename}.png",
        dpi=300
    )

    plt.show()


#######################################################
### 5. Dual axis benchmark plot

def benchmark_plot(
        df,
        benchmark_col,
        fed_col,
        benchmark_name
):

    fig, ax1 = plt.subplots(
        figsize=(12,6)
    )

    ax1.plot(
        df["date"],
        df[benchmark_col],
        color="black",
        linewidth=2.5,
        label=benchmark_col
    )

    ax1.set_ylabel(
        benchmark_name
    )

    fed_colors = {
        "fed_base": "tab:blue",
        "fed_tfidf": "tab:orange",
        "fed_embedding": "tab:green",
        "fed_embedding_pure": "tab:red",
        "fed_embedding_chair": "tab:purple",
        "fed_finbert": "tab:brown",
        "fed_finbert_chair": "tab:pink",
        "fed_finbert_pure": "tab:gray",
        "fed_finbert_A": "tab:blue",
        "fed_finbert_B": "tab:red"
    }

    ax2 = ax1.twinx()

    ax2.plot(
        df["date"],
        df[fed_col],
        color=fed_colors[fed_col],
        alpha=0.8,
        label=fed_col
    )

    ax2.set_ylabel(
        fed_col
    )

    plt.title(
        f"{benchmark_name} vs {fed_col}"
    )

    plt.tight_layout()

    plt.savefig(
        f"figures/{benchmark_name}_{fed_col}.png",
        dpi=300
    )

    plt.show()

#######################################################
### 5. Z-score benchmark plot

def plot_standardized_benchmark(
    df,
    benchmark_col,
    fed_col,
    benchmark_name
):
    
    plot_df = df[
        ["date", benchmark_col, fed_col]
    ].copy()

    # Standardize both indices
    for col in [benchmark_col, fed_col]:

        plot_df[col] = (
            plot_df[col] - plot_df[col].mean()
        ) / plot_df[col].std()

    fed_colors = {
        "fed_base": "tab:blue",
        "fed_tfidf": "tab:orange",
        "fed_embedding": "tab:green",
        "fed_embedding_pure": "tab:red",
        "fed_embedding_chair": "tab:purple",
        "fed_finbert": "tab:brown",
        "fed_finbert_chair": "tab:pink",
        "fed_finbert_pure": "tab:blue",
        "fed_finbert_A": "tab:blue",
        "fed_finbert_B": "tab:red"
    }

    plt.figure(figsize=(12,6))
    plt.plot(
        plot_df["date"],
        plot_df[benchmark_col],
        color="black",
        linewidth=2.5,
        label=benchmark_name
    )

    plt.plot(
        plot_df["date"],
        plot_df[fed_col],
        color=fed_colors.get(fed_col, "tab:blue"),
        alpha=0.8,
        label=fed_col
    )

    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        alpha=0.5
    )

    plt.ylabel("Standardized value (z-score)")

    plt.title(
        f"{benchmark_name} vs {fed_col} (standardized)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"figures/{benchmark_name}_{fed_col}_zscore.png",
        dpi=300
    )

    plt.show()



#######################################################
### MAIN

if __name__ == "__main__":


    df = load_data()

    ### Methodology

    plot_methodology(df)

    ### Chair vs Full

    plot_chair_vs_full(
        df,
        "fed_base",
        "fed_chair",
        "Base vs Chair Base",
        "chair_base"
    )

    plot_chair_vs_full(
        df,
        "fed_embedding",
        "fed_embedding_chair",
        "Embedding vs Chair Embedding",
        "chair_embedding"
    )

    plot_chair_vs_full(
        df,
        "fed_embedding_pure",
        "fed_embedding_chair_pure",
        "Embedding Pure vs Chair",
        "chair_embedding_pure"
    )

    plot_chair_vs_full(
        df,
        "fed_finbert_C",
        "fed_finbert_chair_C",
        "FinBERT C vs Chair",
        "chair_finbert_C"
    )

    ### EPU

    epu_indices = [
        "fed_base",
        "fed_tfidf",
        "fed_embedding",
        "fed_embedding_pure",
        "fed_embedding_chair",
        "fed_finbert"
    ]

    for idx in epu_indices:

        benchmark_plot(
            df,
            "epu",
            idx,
            "EPU"
        )
    

    ### MPU

    mpu_indices = [
        "fed_base",
        "fed_tfidf",
        "fed_embedding",
        "fed_embedding_pure",
        "fed_embedding_chair",
        "fed_finbert"
    ]

    for idx in mpu_indices:

        benchmark_plot(
            df,
            "mpu",
            idx,
            "MPU"
        )

    ### VIX

    vix_indices = [
        "fed_embedding_pure",
        "fed_finbert_pure",
        "fed_finbert_A",
        "fed_finbert_B"
    ]

    for idx in vix_indices:

        benchmark_plot(
            df,
            "vix",
            idx,
            "VIX"
        )

    ### Standarized plots

    for idx in epu_indices:

        plot_standardized_benchmark(
            df,
            "epu",
            idx,
            "EPU"
        )

    for idx in mpu_indices:

        plot_standardized_benchmark(
            df,
            "mpu",
            idx,
            "MPU"
        )


    for idx in vix_indices:

        plot_standardized_benchmark(
            df,
            "vix",
            idx,
            "VIX"
        )

    print("\nAll figures generated.")