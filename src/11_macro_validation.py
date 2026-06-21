'''
This script compares how our indices, created from Fed speeches, relate to macroeconomic measures
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CORE_INDICES = [
    "fed_base",
    "fed_tfidf",
    "fed_embedding_pure",
    "fed_embedding_chair",
    "fed_finbert",
    "fed_finbert_pure"
]

#######################################################
### 1. Load all indices

def load_indices():

    indices = {

        "fed_base":
            pd.read_pickle(
                "data/uncertainty_index.pkl"
            ),

        "fed_chair":
            pd.read_pickle(
                "data/uncertainty_index_chair.pkl"
            ),

        "fed_tfidf":
            pd.read_pickle(
                "data/tfidf_uncertainty_index.pkl"
            ),

        "fed_embedding":
            pd.read_pickle(
                "data/embedding_index.pkl"
            ),

        "fed_embedding_chair":
            pd.read_pickle(
                "data/embedding_chair_index.pkl"
            ),

        "fed_embedding_pure":
            pd.read_pickle(
                "data/embedding_pure_index.pkl"
            ),

        "fed_embedding_chair_pure":
            pd.read_pickle(
                "data/embedding_chair_pure_index.pkl"
            ),

        "fed_finbert":
            pd.read_pickle(
                "data/finbert_encoder_index.pkl"
            ),

        "fed_finbert_chair":
            pd.read_pickle(
                "data/finbert_encoder_chair_index.pkl"
            ),

        "fed_finbert_pure":
            pd.read_pickle(
                "data/finbert_encoder_pure_index.pkl"
            ),

        "fed_finbert_chair_pure":
            pd.read_pickle(
                "data/finbert_encoder_chair_pure_index.pkl"
            ),

        "fed_finbert_A":
            pd.read_pickle(
                "data/finbert_index_A.pkl"
            ),

        "fed_finbert_chair_A":
            pd.read_pickle(
                "data/finbert_chair_index_A.pkl"
            ),

        "fed_finbert_B":
            pd.read_pickle(
                "data/finbert_index_B.pkl"
            ),

        "fed_finbert_chair_B":
            pd.read_pickle(
                "data/finbert_chair_index_B.pkl"
            ),

        "fed_finbert_C":
            pd.read_pickle(
                "data/finbert_index_C.pkl"
            ),

        "fed_finbert_chair_C":
            pd.read_pickle(
                "data/finbert_chair_index_C.pkl"
            ),

        "fed_finbert_D":
            pd.read_pickle(
                "data/finbert_index_D.pkl"
            ),

        "fed_finbert_chair_D":
            pd.read_pickle(
                "data/finbert_chair_index_D.pkl"
            )
    }

    for name, df in indices.items():

        if "month" in df.columns:

            df.index = (
                pd.PeriodIndex(
                    df["month"],
                    freq="M"
                )
                .to_timestamp()
            )

        elif isinstance(
            df.index,
            pd.PeriodIndex
        ):

            df.index = (
                df.index
                .to_timestamp()
            )

        elif "date" in df.columns:

            df.index = pd.to_datetime(
                df["date"]
            )

        else:

            raise ValueError(
                f"Cannot determine dates for {name}"
            )

        df.index = (
            df.index
            + pd.offsets.MonthEnd(0)
        )

        indices[name] = df

    return indices


#######################################################
### 2. Download macroeconomic data

def fred_series(series_id):

    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    )

    df = pd.read_csv(url)

    date_col = df.columns[0]

    df[date_col] = pd.to_datetime(
        df[date_col]
    )

    df = df.set_index(date_col)

    return df

def load_macro_data():

    start = "2006-01-01"

    # GDP
    gdp = fred_series("GDPC1")
    gdp = gdp.loc[start:]
    gdp["gdp_growth"] = (gdp["GDPC1"].pct_change(4)*100)
    gdp = gdp.resample("ME").ffill()

    # Unemployment
    unrate = fred_series("UNRATE")
    unrate = unrate.loc[start:]

    #CPI inflation
    cpi = fred_series("CPIAUCSL")
    cpi = cpi.loc[start:]
    cpi["inflation"] = (cpi["CPIAUCSL"].pct_change(12)* 100)

    # Recession indicator
    recession = fred_series("USREC")
    recession = recession.loc[start:]
    recession = recession.rename(
        columns={
            "USREC": "recession"
        }
    )

    gdp = gdp[["gdp_growth"]]
    unrate = unrate.rename(
        columns={"UNRATE": "unemployment"}
    )
    cpi = cpi[["inflation"]]

    # Industrial production
    indpro = fred_series("INDPRO")
    indpro = indpro.loc[start:]

    indpro = indpro.rename(
        columns={
            "INDPRO": "industrial_production"
        }
    )

    # Consumer Sentiment
    sentiment = fred_series("UMCSENT")
    sentiment = sentiment.loc[start:]

    sentiment = sentiment.rename(
        columns={
            "UMCSENT": "consumer_sentiment"
        }
    )

    # SP500 return
    sp500 = fred_series("SP500")

    sp500 = (
        sp500
        .resample("ME")
        .last()
    )

    sp500 = sp500.loc[start:]

    sp500["sp500_return"] = (
        sp500["SP500"]
        .pct_change()
        * 100
    )

    sp500 = sp500[["sp500_return"]]
        

    for df in [gdp, unrate, cpi, recession, indpro, sentiment, sp500]:

        df.index = pd.to_datetime(df.index)

        df.index = (
            df.index
            .to_period("M")
            .to_timestamp("M")
        )

    return {
        "GDP Growth": gdp,
        "Unemployment": unrate,
        "Inflation": cpi,
        "Recession": recession,
        "Industrial Production": indpro,
        "Consumer Sentiment": sentiment,
        "SP500 Returns": sp500
    }


#######################################################
### 3. Create benchmark

def benchmark_index(index, macro, macro_column):

    df = index.join(
        macro,
        how="inner"
    )

    lag12 = (
        df["uncertainty_index"]
        .corr(
            df[macro_column]
            .shift(12)
        )
    )

    lag6 = (
        df["uncertainty_index"]
        .corr(
            df[macro_column]
            .shift(6)
        )
    )

    lag3 = (
        df["uncertainty_index"]
        .corr(
            df[macro_column]
            .shift(3)
        )
    )

    lag1 = (
        df["uncertainty_index"]
        .corr(
            df[macro_column]
            .shift(1)
        )
    )

    current = (
        df["uncertainty_index"]
        .corr(df[macro_column])
    )

    lead1 = (
        df["uncertainty_index"]
        .corr(
            df[macro_column]
            .shift(-1)
        )
    )

    lead3 = (
        df["uncertainty_index"]
        .corr(
            df[macro_column]
            .shift(-3)
        )
    )

    lead6 = (
        df["uncertainty_index"]
        .corr(
            df[macro_column]
            .shift(-6)
        )
    )

    lead12 = (
        df["uncertainty_index"]
        .corr(
            df[macro_column]
            .shift(-12)
        )
    )

    return (
        lag12,
        lag6,
        lag3,
        lag1,
        current,
        lead1,
        lead3,
        lead6,
        lead12
    )


#######################################################
### 4. Run benchmark

def run_benchmarks(indices,macro_dict):

    results = []

    for index_name, index in indices.items():

        for macro_name, macro in macro_dict.items():

            macro_column = macro.columns[0]

            lag12, lag6, lag3, lag1, current, lead1, lead3, lead6, lead12 = (
                benchmark_index(
                    index,
                    macro,
                    macro_column
                )
            )

            results.append({
        "index": index_name,
        "macro": macro_name,

        "lag12": lag12,
        "lag6": lag6,
        "lag3": lag3,
        "lag1": lag1,
        "current": current,
        "lead1": lead1,
        "lead3": lead3,
        "lead6": lead6,
        "lead12": lead12
})

    results = pd.DataFrame(results)

    return results


#######################################################
### 5. Macro ranking table

def create_macro_rankings(results):

    horizon_cols = [
        "lead3",
        "lead6",
        "lead12"
    ]

    rankings = []

    for macro in results["macro"].unique():

        temp = (
            results[
                results["macro"] == macro
            ]
            .copy()
        )

        temp["predictive_score"] = (
            temp[horizon_cols]
            .abs()
            .mean(axis=1)
        )

        temp = temp.sort_values(
            "predictive_score",
            ascending=False
        )

        rankings.append(temp)

        print(
            f"\n{'='*60}"
        )

        print(
            f"\n{macro.upper()} RANKING\n"
        )

        print(
            temp[
                [
                    "index",
                    "predictive_score",
                    "lead3",
                    "lead6",
                    "lead12"
                ]
            ]
            .to_string(index=False)
        )

    rankings = pd.concat(
        rankings,
        ignore_index=True
    )

    return rankings

#######################################################
### 6. Macro heatmap

def plot_macro_heatmap(results, overall_ranking):

    horizon_cols = [
        "lead3",
        "lead6",
        "lead12"
    ]

    temp = results.copy()

    temp["predictive_score"] = (
        temp[horizon_cols]
        .mean(axis=1)
    )

    heatmap_df = (
        temp.pivot(
            index="index",
            columns="macro",
            values="predictive_score"
        )
    )

    family_order = [
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

    heatmap_df = heatmap_df.reindex(family_order)

    plt.figure(
        figsize=(12,10)
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".2f"
    )

    plt.title(
        "Predictive Strength Across Macroeconomic Variables"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/macro_validation_heatmap.png",
        dpi=300
    )

    plt.show()

#######################################################
### 7. Overall ranking

def create_overall_macro_ranking(results):

    horizon_cols = [
        "lead3",
        "lead6",
        "lead12"
    ]

    temp = results.copy()

    temp["predictive_score"] = (
        temp[horizon_cols]
        .abs()
        .mean(axis=1)
    )

    ranking = (
        temp
        .groupby("index")
        ["predictive_score"]
        .mean()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    print(
        "\nOVERALL MACRO RANKING\n"
    )

    print(
        ranking.to_string(index=False)
    )

    return ranking

#######################################################
### 8. Plot index against macro

def plot_index_vs_macro(
    index,
    macro,
    macro_column,
    title,
    lead=0
):

    df = (
        index.join(
            macro,
            how="inner"
        )
        .dropna()
    )

    if lead != 0:

        df[macro_column] = (
            df[macro_column]
            .shift(-lead)
        )

    df = df.dropna()

    df["unc_z"] = (
        df["uncertainty_index"]
        .rolling(3, center=True)
        .mean()
    )

    df["unc_z"] = (
        df["unc_z"]
        - df["unc_z"].mean()
    ) / df["unc_z"].std()

    df["unc_smooth"] = (
        df["uncertainty_index"]
        .rolling(6, center=True)
        .mean()
    )
    
    df["unc_smooth_z"] = (
        df["unc_smooth"]
        - df["unc_smooth"].mean()
    ) / df["unc_smooth"].std()

    if macro_column == "recession":

        df["macro_z"] = df[macro_column]

    else:

        df["macro_z"] = (
            df[macro_column]
            - df[macro_column].mean()
        ) / df[macro_column].std()

    corr = (
        df["uncertainty_index"]
        .corr(df[macro_column])
    )

    plt.figure(
        figsize=(12,6)
    )

    plt.plot(
        df.index,
        df["unc_z"],
        label="3-month MA",
        linewidth=2,

    )

    plt.plot(
        df.index,
        df["unc_smooth_z"],
        linewidth=2.5,
        color="gray",
        label="6-month MA"
    )

    plt.plot(
        df.index,
        df["macro_z"],
        label=macro_column,
        linewidth=3,
        color="black"
    )

    plt.axhline(
        0,
        color="black",
        alpha=0.4
    )

    plt.title(
        f"{title}\n"
        f"Macro shifted {lead} months forward | Corr = {corr:.3f}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"figures/{title.replace(' ','_')}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()



#######################################################
### 10. Cross-correlation function

def cross_correlation_profile(index,macro,macro_column,max_lag=24):

    df = index.join(
        macro,
        how="inner"
    ).dropna()

    correlations = []

    for lag in range(0, max_lag + 1):

        corr = (
            df["uncertainty_index"]
            .corr(
                df[macro_column]
                .shift(-lag)
            )
        )

        correlations.append({
            "lag": lag,
            "corr": corr
        })

    return pd.DataFrame(correlations)


#######################################################
### 11. Plot Cross-correlation

def plot_multiple_cross_correlations(indices, macro,macro_column,index_names=None,max_lag=24):

    if index_names is None:

        index_names = list(
            indices.keys()
        )

    plt.figure(
        figsize=(12, 6)
    )

    for name in index_names:

        profile = (
            cross_correlation_profile(
                indices[name],
                macro,
                macro_column,
                max_lag
            )
        )

        plt.plot(
            profile["lag"],
            profile["corr"],
            label=name
        )

    plt.axvline(
        0,
        linestyle="--",
        color="black"
    )

    plt.xlabel(
        "Lag (months)"
    )

    plt.ylabel(
        "Correlation"
    )

    plt.title(
        f"Cross-Correlation Profile: {macro_column}"
    )

    plt.legend()

    plt.grid(True)

    plt.axhline(
        0,
        color="black",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        f"figures/crosscorr_{macro_column}.png",
        dpi=300
    )

    plt.show()


#######################################################
### MAIN

if __name__ == "__main__":

    indices = load_indices()

    macro_dict = load_macro_data()
    gdp = macro_dict["GDP Growth"]
    unemployment = macro_dict["Unemployment"]
    recession = macro_dict["Recession"]

    # Correlation benchmark and tables of correlation
    results = run_benchmarks(
        indices,
        macro_dict
    )
    results.to_csv(
        "data/macro_benchmark_results.csv",
        index=False
    )

    macro_rankings = create_macro_rankings(
        results
    )

    overall_ranking = (
        create_overall_macro_ranking(
            results
        )
    )

    plot_macro_heatmap(
        results, overall_ranking
    )

    core_indices = {
        k:v
        for k,v in indices.items()
        if k in CORE_INDICES
    }

    plot_multiple_cross_correlations(
        core_indices,
        gdp,
        "gdp_growth"
    )

    plot_multiple_cross_correlations(
        core_indices,
        macro_dict["Industrial Production"],
        "industrial_production"
    )

    plot_multiple_cross_correlations(
        core_indices,
        unemployment,
        "unemployment"
    )

    plot_multiple_cross_correlations(
        core_indices,
        recession,
        "recession"
    )

    ### Selected visual inspections

    # GDP

    plot_index_vs_macro(
        indices["fed_embedding"],
        macro_dict["GDP Growth"],
        "gdp_growth",
        "GDP Growth vs Fed Embedding",
        lead=3
    )

    plot_index_vs_macro(
        indices["fed_tfidf"],
        macro_dict["GDP Growth"],
        "gdp_growth",
        "GDP Growth vs Fed TFIDF",
        lead=12
    )

    plot_index_vs_macro(
        indices["fed_embedding_pure"],
        macro_dict["GDP Growth"],
        "gdp_growth",
        "GDP Growth vs Fed Embedding Pure",
        lead=12
    )

    # Unemployment

    plot_index_vs_macro(
        indices["fed_finbert_B"],
        macro_dict["Unemployment"],
        "unemployment",
        "Unemployment vs FinBERT B",
        lead=3
    )

    plot_index_vs_macro(
        indices["fed_embedding"],
        macro_dict["Unemployment"],
        "unemployment",
        "Unemployment vs Fed Embedding",
        lead=12
    )

    # Inflation

    plot_index_vs_macro(
        indices["fed_embedding"],
        macro_dict["Inflation"],
        "inflation",
        "Inflation vs Fed Embedding",
        lead=12
    )

    plot_index_vs_macro(
        indices["fed_tfidf"],
        macro_dict["Inflation"],
        "inflation",
        "Inflation vs Fed TFIDF",
        lead=12
    )

    plot_index_vs_macro(
        indices["fed_finbert"],
        macro_dict["Inflation"],
        "inflation",
        "Inflation vs FinBERT",
        lead=12
    )

    # Recession

    plot_index_vs_macro(
        indices["fed_embedding_pure"],
        macro_dict["Recession"],
        "recession",
        "Recession vs Fed Embedding Pure",
        lead=12
    )

    plot_index_vs_macro(
        indices["fed_tfidf"],
        macro_dict["Recession"],
        "recession",
        "Recession vs Fed TFIDF",
        lead=6
    )

    # Industrial Production

    plot_index_vs_macro(
        indices["fed_finbert_B"],
        macro_dict["Industrial Production"],
        "industrial_production",
        "Industrial Production vs FinBERT B",
        lead=3
    )

    plot_index_vs_macro(
        indices["fed_embedding"],
        macro_dict["Industrial Production"],
        "industrial_production",
        "Industrial Production vs Fed Embedding",
        lead=6
    )

    # Consumer Sentiment

    plot_index_vs_macro(
        indices["fed_finbert_B"],
        macro_dict["Consumer Sentiment"],
        "consumer_sentiment",
        "Consumer Sentiment vs FinBERT B",
        lead=3
    )

    plot_index_vs_macro(
        indices["fed_chair"],
        macro_dict["Consumer Sentiment"],
        "consumer_sentiment",
        "Consumer Sentiment vs Chair Index",
        lead=6
    )

    # SP500

    plot_index_vs_macro(
        indices["fed_finbert_chair_D"],
        macro_dict["SP500 Returns"],
        "sp500_return",
        "SP500 Returns vs FinBERT Chair D",
        lead=6
    )

    plot_index_vs_macro(
        indices["fed_chair"],
        macro_dict["SP500 Returns"],
        "sp500_return",
        "SP500 Returns vs Chair Index",
        lead=6
    )

    print("\nDone.")




