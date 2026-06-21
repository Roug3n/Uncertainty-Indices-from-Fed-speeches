import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    roc_curve
)

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


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
### 3. Prediction benchmark
def binary_prediction_metrics(index,target):

    df = pd.concat(
        [index["uncertainty_index"], target],
        axis=1,
        join="inner"
    )

    df = df.dropna()

    if len(df) == 0:
        return np.nan, np.nan, np.nan

    y_true = df.iloc[:, 1]

    scores = df["uncertainty_index"]

    try:

        auc = roc_auc_score(
            y_true,
            scores
        )

    except:

        auc = np.nan

    threshold = scores.quantile(0.8)

    y_pred = (
        scores > threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return auc, precision, recall


#######################################################
### 4. Run prediciton benchmark

def run_prediction_benchmark(indices,target,target_name):

    results = []

    for name, index in indices.items():

        auc, precision, recall = (
            binary_prediction_metrics(
                index,
                target
            )
        )

        results.append({

            "index": name,

            "target": target_name,

            "auc": auc,

            "precision": precision,

            "recall": recall

        })

    return pd.DataFrame(results)


#######################################################
### 5. Forecast regression
def forecast_regression(
    index,
    macro,
    macro_column,
    horizon=6
):

    df = index.join(
        macro,
        how="inner"
    )

    df["future_macro"] = (
        df[macro_column]
        .shift(-horizon)
    )

    df = df.dropna()

    X = sm.add_constant(
        df["uncertainty_index"]
    )

    y = df["future_macro"]

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags":6}
    )

    return {

        "beta":
            model.params[
                "uncertainty_index"
            ],

        "pvalue":
            model.pvalues[
                "uncertainty_index"
            ],

        "r2":
            model.rsquared
    }


#######################################################
### 6. Controlled forecast regression
def controlled_forecast_regression(
    index,
    macro,
    macro_column,
    horizon=6
):

    df = index.join(
        macro,
        how="inner"
    )

    df["future_macro"] = (
        df[macro_column]
        .shift(-horizon)
    )

    df = df.dropna()

    X = df[
        [
            "uncertainty_index",
            macro_column
        ]
    ]

    X = sm.add_constant(X)

    y = df["future_macro"]

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags":6}
    )

    return {

        "beta":
            model.params[
                "uncertainty_index"
            ],

        "pvalue":
            model.pvalues[
                "uncertainty_index"
            ],

        "r2":
            model.rsquared
    }


#######################################################
### 7. Granger causality

def granger_test(
    index,
    macro,
    macro_column,
    max_lag=12
):

    df = index.join(
        macro,
        how="inner"
    )

    df = df[
        [
            macro_column,
            "uncertainty_index"
        ]
    ].dropna()

    results = grangercausalitytests(
        df,
        maxlag=max_lag,
        verbose=False
    )

    best_p = 1.0
    best_lag = None

    for lag in range(
        1,
        max_lag + 1
    ):

        p = results[lag][0]["ssr_ftest"][1]

        if p < best_p:

            best_p = p
            best_lag = lag

    return best_p, best_lag


#######################################################
### 8. Forecast ranking plot

def plot_forecast_ranking(
    forecast_df,
    macro_name
):

    df = (
        forecast_df[
            forecast_df["macro"] == macro_name
        ]
        .sort_values(
            "ctrl_r2",
            ascending=False
        )
        .copy()
    )

    colors = []

    for p in df["ctrl_pvalue"]:

        if p < 0.05:
            colors.append("navy")

        elif p < 0.10:
            colors.append("cornflowerblue")

        else:
            colors.append("lightgray")

    plt.figure(figsize=(9,6))

    plt.barh(
        df["index"],
        df["ctrl_r2"],
        color=colors
    )

    plt.xlabel("Controlled Regression R²")

    plt.title(
        f"{macro_name}: Forecast Performance"
    )

    plt.gca().invert_yaxis()

    from matplotlib.patches import Patch

    legend_elements = [

        Patch(
            facecolor="navy",
            label="p < 0.05"
        ),

        Patch(
            facecolor="cornflowerblue",
            label="0.05 ≤ p < 0.10"
        ),

        Patch(
            facecolor="lightgray",
            label="p ≥ 0.10"
        )
    ]

    plt.legend(
        handles=legend_elements,
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        f"figures/forecast_ranking_{macro_name.replace(' ','_')}.png",
        dpi=300
    )

    plt.show()


#######################################################
### 9. Prediction heatmap

def plot_prediction_heatmap(prediction_results):

    heatmap_df = (
        prediction_results
        .pivot(
            index="index",
            columns="target",
            values="auc"
        )
    )

    plt.figure(
        figsize=(8,8)
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="RdYlGn",
        center=0.5,
        fmt=".2f"
    )

    plt.title(
        "Prediction Performance (ROC-AUC)"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/prediction_heatmap.png",
        dpi=300
    )

    plt.show()


#######################################################
### 10. Granger heatmap

def plot_granger_heatmap(granger_results):

    heatmap_df = (
        granger_results
        .pivot(
            index="index",
            columns="macro",
            values="best_pvalue"
        )
    )

    heatmap_df = (
        -np.log10(
            heatmap_df
        )
    )

    plt.figure(
        figsize=(10,8)
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="YlGnBu"
    )

    plt.title(
        "Granger Causality Strength\n(-log10 p-value)"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/granger_heatmap.png",
        dpi=300
    )

    plt.show()


#######################################################
### 11. ROC curves

def plot_roc_curves(
    indices,
    target,
    selected_indices,
    title
):

    plt.figure(
        figsize=(8,6)
    )

    for name in selected_indices:

        df = pd.concat(
            [
                indices[name]["uncertainty_index"],
                target
            ],
            axis=1,
            join="inner"
        ).dropna()

        y_true = df.iloc[:,1]

        scores = df["uncertainty_index"]

        fpr, tpr, _ = roc_curve(
            y_true,
            scores
        )

        auc = roc_auc_score(
            y_true,
            scores
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.2f})"
        )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="black"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"figures/{title.replace(' ','_')}.png",
        dpi=300
    )

    plt.show()


#######################################################
### 12. Overall ranking table

def build_summary_table(
    prediction_results,
    forecast_results,
    granger_results
):

    recession_auc = (
        prediction_results[
            prediction_results["target"] == "Recession"
        ][["index","auc"]]
        .rename(
            columns={
                "auc":"recession_auc"
            }
        )
    )

    gdp_auc = (
        prediction_results[
            prediction_results["target"] == "GDP Contraction"
        ][["index","auc"]]
        .rename(
            columns={
                "auc":"gdp_auc"
            }
        )
    )

    forecast_best = (
        forecast_results
        .groupby("index")
        ["ctrl_pvalue"]
        .min()
        .reset_index()
        .rename(
            columns={
                "ctrl_pvalue":"best_forecast_p"
            }
        )
    )

    granger_best = (
        granger_results
        .groupby("index")
        ["best_pvalue"]
        .min()
        .reset_index()
        .rename(
            columns={
                "best_pvalue":"best_granger_p"
            }
        )
    )

    summary = recession_auc.merge(
        gdp_auc,
        on="index"
    )

    summary = summary.merge(
        forecast_best,
        on="index"
    )

    summary = summary.merge(
        granger_best,
        on="index"
    )

    summary["rank_recession_auc"] = (
        summary["recession_auc"]
        .rank(ascending=False)
    )

    summary["rank_gdp_auc"] = (
        summary["gdp_auc"]
        .rank(ascending=False)
    )

    summary["rank_forecast"] = (
        summary["best_forecast_p"]
        .rank(ascending=True)
    )

    summary["rank_granger"] = (
        summary["best_granger_p"]
        .rank(ascending=True)
    )

    summary["overall_score"] = (

        summary["rank_recession_auc"]

        + summary["rank_gdp_auc"]

        + summary["rank_forecast"]

        + summary["rank_granger"]

    )

    summary = summary.sort_values(
        "overall_score"
    )

    return summary


#######################################################
### 13. Overall ranking figure

def plot_overall_ranking(summary):

    plt.figure(figsize=(10,7))

    plt.barh(
        summary["index"],
        summary["overall_score"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Overall Performance Score\n(lower is better)"
    )

    plt.title(
        "Overall Predictive Performance Ranking"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/overall_ranking.png",
        dpi=300
    )

    plt.show()


#######################################################
### MAIN
if __name__ == "__main__":


    indices = load_indices()

    macro_dict = load_macro_data()

    gdp = macro_dict["GDP Growth"]
    recession = macro_dict["Recession"]
    unemployment = macro_dict["Unemployment"]
    industrial = macro_dict["Industrial Production"]

    # Prediction results
    future_recession = (recession["recession"].shift(-6)> 0).astype(int)
    future_negative_gdp = (gdp["gdp_growth"].shift(-6)< 0).astype(int)

    prediction_results = pd.concat([

        run_prediction_benchmark(
            indices,
            future_recession,
            "Recession"
        ),

        run_prediction_benchmark(
            indices,
            future_negative_gdp,
            "GDP Contraction"
        )
    ])

    for target in prediction_results["target"].unique():

        print(f"\n{target}\n")

        print(
            prediction_results[
                prediction_results["target"] == target
            ]
            .sort_values(
                "auc",
                ascending=False
            )
        )

    # Forecast results
    forecast_results = []

    # Granger Causality
    granger_results = []

    for index_name,index in indices.items():

        for macro_name in [

            "GDP Growth",
            "Unemployment",
            "Industrial Production",
            "Consumer Sentiment"
        ]:

            macro = macro_dict[macro_name]

            macro_column = macro.columns[0]

            uni = forecast_regression(
                index,
                macro,
                macro_column,
                horizon=6
            )

            ctrl = controlled_forecast_regression(
                index,
                macro,
                macro_column,
                horizon=6
            )

            forecast_results.append({

                "index": index_name,
                "macro": macro_name,

                "uni_beta": uni["beta"],
                "uni_pvalue": uni["pvalue"],
                "uni_r2": uni["r2"],

                "ctrl_beta": ctrl["beta"],
                "ctrl_pvalue": ctrl["pvalue"],
                "ctrl_r2": ctrl["r2"]

            })

            pvalue, lag = granger_test(
                index,
                macro,
                macro_column
            )

            granger_results.append({

                "index": index_name,
                "macro": macro_name,

                "best_pvalue": pvalue,
                "best_lag": lag

            })

    forecast_results = pd.DataFrame(
        forecast_results
    )    

    granger_results = pd.DataFrame(
        granger_results
    )

    for macro in forecast_results["macro"].unique():

        print(f"\n{'='*60}")
        print(f"\n{macro.upper()} FORECAST REGRESSIONS\n")

        temp = (
            forecast_results[
                forecast_results["macro"] == macro
            ]
            .sort_values(
                ["ctrl_pvalue", "ctrl_r2"],
                ascending=[True, False]
            )
        )

        print(
            temp[
                [
                    "index",
                    "ctrl_beta",
                    "ctrl_pvalue",
                    "ctrl_r2"
                ]
            ]
        )    
    
    for macro in granger_results["macro"].unique():

        print(f"\n{'='*60}")
        print(f"\n{macro.upper()} GRANGER CAUSALITY\n")

        temp = (
            granger_results[
                granger_results["macro"] == macro
            ]
            .sort_values(
                "best_pvalue"
            )
        )

        print(temp)

    
    summary_table = build_summary_table(
        prediction_results,
        forecast_results,
        granger_results
    )

    print("\n")
    print("="*60)
    print("OVERALL INDEX RANKING")
    print("="*60)

    print(summary_table)
    plot_overall_ranking(
        summary_table
    )

    for macro in [

        "GDP Growth",
        "Unemployment",
        "Industrial Production",
        "Consumer Sentiment"

    ]:

        plot_forecast_ranking(
            forecast_results,
            macro
        )

    plot_prediction_heatmap(
        prediction_results
    )

    plot_granger_heatmap(
        granger_results
    )

    plot_roc_curves(

        indices,

        future_recession,

        [
            "fed_finbert_chair",
            "fed_finbert",
            "fed_tfidf",
            "fed_base",
            "fed_embedding"
        ],

        "ROC Recession Prediction"
    )

    plot_roc_curves(

        indices,

        future_negative_gdp,

        [
            "fed_finbert_pure",
            "fed_finbert_chair_A",
            "fed_finbert_B",
            "fed_base",
            "fed_embedding"
        ],

        "ROC GDP Contraction Prediction"
    )
