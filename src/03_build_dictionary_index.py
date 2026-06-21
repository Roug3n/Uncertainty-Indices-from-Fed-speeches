'''
 In this script we create the first index based on a specialized uncertainty dictionary. 
 The script stores the index as well as a new df with the uncertainty value for each speech 
 which is equal to the nº of words in the speech that match a word in the uncertainty dictionary
 divided by the nº of words from that speech.
 We also and additional index, restricted to chairman speeches. That way we study the effects of limiting it to chairman speeches 
'''

'''
Reason for choosing the Dictionary used:

Loughran–McDonald Financial Sentiment Dictionary

It is:
    Specifically designed for financial and economic texts
    Has a dedicated “uncertainty” category
    Corrects common misclassifications (e.g., “liability” is neutral in finance, not negative)

Pros

High signal in macro/financial contexts
Widely used in empirical finance and central bank research
Interpretable and easy to justify in a paper

Cons

Built mainly on 10-K filings, not speeches
May miss some forward-guidance phrasing typical of central banks
'''


import pandas as pd
import spacy

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

#######################################################
### 1. Load cleaned data

def load_data(path="data/corpus.pkl"):
    df = pd.read_pickle(path)
    print("Loaded cleaned data:", df.shape)
    return df


#######################################################
### 2. Create and load dictionary

'''
The following code creates a .txt from the words classified as Uncertain by the Loughran-McDonald uncertainty dictionary
It will be left in this comment as the .txt is already created and stored in data/lm_uncertainty.txt

df_dict = pd.read_excel("data/Loughran-McDonald_MasterDictionary_1993-2025.xlsx")
uncertainty_words = df_dict[df_dict["Uncertainty"] > 0]["Word"]
uncertainty_words = uncertainty_words.str.lower()
uncertainty_words.to_csv("data/lm_uncertainty.txt", index=False, header=False)
'''

def load_dictionary(path="data/lm_uncertainty.txt"):
    with open(path, "r") as f:
        words = f.read().splitlines()

    '''
    To ensure that lexical variations did not affect the measurement of uncertainty,
    both the corpus and the dictionary were lemmatized 
    '''
    dictionary = {
        nlp(word)[0].lemma_.lower()
        for word in words
    }

    print("Dictionary size:", len(dictionary))

    return dictionary


#######################################################
### 3. Compute uncertainty score for a speech

'''
Uncertainty is measured as the proportion of words in a speech
that belong to the Loughran-McDonald uncertainty category.
    uncertainty = (# uncertainty words) / (# total words)
'''

def compute_uncertainty(tokens, dictionary):

    total_words = len(tokens)

    if total_words == 0:
        return pd.Series({
            "uncertainty_words": 0,
            "uncertainty": 0
        })

    count = sum(
        1 for token in tokens
        if token in dictionary
    )

    return pd.Series({
        "uncertainty_words": count,
        "uncertainty": count / total_words
    })


#######################################################
### 4. Apply to all speeches

def apply_uncertainty(df, dictionary):

    scores = df["tokens_lemma"].apply(lambda x: compute_uncertainty(x, dictionary))
    df = pd.concat([df, scores], axis=1)

    return df


#######################################################
### 5. Aggregate to monthly index

'''
Monthly aggregation is used to match the frequency of most
macroeconomic variables employed later in the validation stage.
When multiple speeches occur within the same month, their
uncertainty scores are averaged to obtain a single monthly measure.
'''

def build_index(df):

    index = (
        df.groupby("month")["uncertainty"]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    return index

#######################################################
### 6. Build chair-only index

def build_chair_index(df):

    chair_df = df[df["is_chair"]].copy()

    index = (
        chair_df.groupby("month")["uncertainty"]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    return index


#######################################################
### 8. Normalize index (z-score)

'''
Standardization facilitates comparison across uncertainty measures
developed later in the project (TF-IDF, embeddings and FinBERT),
placing all indices on a common scale with mean 0 and variance 1.
'''

def normalize_index(df):
    mean = df["uncertainty"].mean()
    std = df["uncertainty"].std()

    df["uncertainty_index"] = (df["uncertainty"] - mean) / std

    return df


#######################################################
### 9. Save results

def save_results(df_speeches, df_index):
    df_speeches.to_pickle("data/speeches_with_uncertainty.pkl")
    df_index.to_pickle("data/uncertainty_index.pkl")

    print("Saved results.")


#######################################################
### MAIN

if __name__ == "__main__":

    df = load_data()

    dictionary = load_dictionary()

    df = apply_uncertainty(df, dictionary)

    index = build_index(df)

    index = normalize_index(index)

    index = index.set_index("month")
    index = index.sort_index()

    chair_index = build_chair_index(df)

    chair_index = normalize_index(chair_index)
    chair_index = (
        chair_index
        .set_index("month")
        .sort_index()
    )

    save_results(df, index)
    chair_index.to_pickle("data/uncertainty_index_chair.pkl")

    print("\nTop 10 uncertainty months base index:")
    print(
        index.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    print("\nTop 10 uncertainty months chair index:")
    print(
        chair_index.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )
    print("\nDone.")