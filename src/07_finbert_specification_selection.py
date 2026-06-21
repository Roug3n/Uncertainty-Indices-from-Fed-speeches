'''
This script performs specification selection for the FinBERT-based
uncertainty index.

Different weighting schemes are applied to the three similarity
components:
    sim_u = uncertainty similarity
    sim_b = bad-news similarity
    sim_d = discussion-of-uncertainty similarity

The objective is not statistical optimization but economic
validation. Specifications are evaluated according to whether
their peaks coincide with historically recognized periods of
elevated uncertainty.

The selected specifications are later used in benchmarking and
macroeconomic validation.
'''

import pandas as pd


#######################################################
### 1. Load data

def load_data(path="data/finbert_similarity_components.pkl"):

    df = pd.read_pickle(path)

    print("Loaded corpus:", df.shape)

    return df


#######################################################
### 2. Calculate uncertainty
def calculate_uncertainty(df,weight_u,weight_b,weight_d):

    return (
        weight_u * df["sim_u"]
        - weight_b * df["sim_b"]
        - weight_d * df["sim_d"]
    )


#######################################################
### 3. Monthly aggregation
def build_index(df,column_name):

    index = (
        df.groupby("month")[column_name]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    return index


#######################################################
### 4. Normalize
def normalize_index(df,column_name):

    mean = df[column_name].mean()

    std = df[column_name].std()

    df["uncertainty_index"] = (
        df[column_name] - mean
    ) / std

    return df


#######################################################
### 5. Create and store the indices
def build_and_store_index(
    df,
    weight_u,
    weight_b,
    weight_d,
    label
):

    df_temp = df.copy()

    df_temp["finbert_uncertainty"] = (
        calculate_uncertainty(
            df_temp,
            weight_u,
            weight_b,
            weight_d
        )
    )

    ''' 
    Full sample
    '''

    index = build_index(
        df_temp,
        "finbert_uncertainty"
    )

    index = normalize_index(
        index,
        "finbert_uncertainty"
    )

    index.to_pickle(
        f"data/finbert_index_{label}.pkl"
    )

    '''
    Chair only
    '''

    df_chair = (
        df_temp[df_temp["is_chair"]]
        .copy()
    )

    chair_index = build_index(
        df_chair,
        "finbert_uncertainty"
    )

    chair_index = normalize_index(
        chair_index,
        "finbert_uncertainty"
    )

    chair_index.to_pickle(
        f"data/finbert_chair_index_{label}.pkl"
    )

    print(
        f"Stored specification {label}"
    )

    return index, chair_index


#######################################################
### MAIN

if __name__ == "__main__":

    df = load_data()

    print(df[["sim_u","sim_b","sim_d"]].describe())

    print(df[["sim_u","sim_b","sim_d"]].corr())

    '''
    Candidate weights explored around economically reasonable values.
    The objective is interpretability and historical plausibility,
    not predictive optimization.
    '''
    weight_u_val=(0.75, 1, 1.25, 1.5)
    weight_b_val=(0.25, 0.5, 0.75, 1)
    weight_d_val=(0.25, 0.5, 0.75, 1)

    '''
    The following loop valuates all combinations of candidate weights.
    Each specification generates a monthly uncertainty index whose 
    peaks are inspected against known uncertainty episodes.
    '''
    for weight_u in weight_u_val:
        for weight_b in weight_b_val:
                for weight_d in weight_d_val:
                    
                    df_temp = df.copy()

                    df_temp["finbert_uncertainty"] = calculate_uncertainty(
                        df_temp,
                        weight_u,
                        weight_b,
                        weight_d
                    )

                    df_chair = df_temp[df_temp["is_chair"]].copy()

                    # Index with the designated weights
                    index = build_index(
                        df_temp,
                        "finbert_uncertainty"
                    )
                    index = normalize_index(
                        index,
                        "finbert_uncertainty"
                    )
                    index = index.set_index("month")

                    print("\nTop 10 months finbert encoder:")
                    print("weight_u",weight_u,"weight_b",weight_b,"weight_d", weight_d)
                    print(
                        index.sort_values(
                            "uncertainty_index",
                            ascending=False
                        ).head(10)
                    )

                    # Same but with chair only speeches
                    chair_index = build_index(
                        df_chair,
                        "finbert_uncertainty"
                    )
                    chair_index = normalize_index(
                        chair_index,
                        "finbert_uncertainty"
                    )
                    chair_index = chair_index.set_index("month")

                    print("\nTop 10 months chair finbert encoder:")
                    print("weight_u",weight_u,"weight_b",weight_b,"weight_d", weight_d)
                    print(
                        chair_index.sort_values(
                            "uncertainty_index",
                            ascending=False
                        ).head(10)
                    )   
    
    '''
    After these iterations we conclude that the bests outcomes come from assigning weights similar to:
    A = (1.25, 0.25, 0.50)
    B = (1.25, 0.25, 0.75)
    C = (1.25, 0.50, 0.50)
    D = (1.00, 0.25, 0.25)
    These produced the most economically plausible rankings of uncertainty 
    episodes while maintaining diversity in weighting structures

    Multiple specifications are retained rather than selecting
    a single winner in order to test robustness during the
    benchmarking stage
    '''

    ## Selected specifications

    specifications = {
        "A": (1.25, 0.25, 0.50),
        "B": (1.25, 0.25, 0.75),
        "C": (1.25, 0.50, 0.50),
        "D": (1.00, 0.25, 0.25)
    }

    for label, weights in specifications.items():

        w_u, w_b, w_d = weights

        build_and_store_index(
            df,
            w_u,
            w_b,
            w_d,
            label
        )

    print(
        "\nStored all selected specifications."
    )

    print("\nDone.")
    

'''
Strong prior candidates (2006–2026)

Very high uncertainty

2008-09 to 2009-03: global financial crisis, Lehman aftermath
2010-05 to 2010-08: Eurozone sovereign debt crisis
2011-08 to 2011-10: US debt ceiling crisis + euro crisis escalation
2015-08 to 2016-02: China slowdown, RMB devaluation, global growth fears
2016-06 to 2016-08: Brexit
2020-03 to 2020-06: COVID shock
2022-03 to 2022-11: inflation regime shift, Ukraine war, aggressive tightening
2023-03 to 2023-04: regional banking crisis

Moderately high uncertainty

2007-07 to 2007-12: early subprime stress
2012-06: euro breakup concerns
2018 Q4: trade war + growth slowdown + market selloff
2024-2026: hard to judge because we're evaluating with hindsight
'''