'''
In this script we create the an index applying the TF-IDF methodology

TF-IDF extends the dictionary approach by weighting uncertainty words
according to their importance within each speech.

Words that appear frequently in a particular speech receive a higher weight,
while words that appear in almost every speech receive a lower weight.

This allows the index to capture not only the presence of uncertainty-related
language, but also its relative emphasis within each document.
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)


#######################################################
### 1. Load data
def load_data(path="data/corpus.pkl"):

    df = pd.read_pickle(path)

    print("Loaded corpus:", df.shape)

    return df


#######################################################
### 2. Create and load dictionary (same as in 03_build_dictionary_index.py)

def load_dictionary(path="data/lm_uncertainty.txt"):

    with open(path, "r") as f:
        words = f.read().splitlines()

    dictionary = {
        nlp(word)[0].lemma_.lower()
        for word in words
    }

    print("Dictionary size:", len(dictionary))

    return dictionary


#######################################################
### 3. Build document strings

def build_documents(df):

    '''
    Converts back tokens into strings to later fit TF-IDF
    '''
    return df["tokens_lemma"].apply(
        lambda x: " ".join(x)
    )


#######################################################
### 4. Fitting TF-IDF

'''
Builds a document-term matrix where each element represents
the TF-IDF weight of a given word in a given speech.
'''

def fit_tfidf(documents):

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None
    )
    '''
    The corpus has already been cleaned, tokenized and lemmatized in
    Script 02, therefore no additional preprocessing is required here.
    '''

    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names


#######################################################
### 5. Compute uncertainty score

'''
The uncertainty score of a speech is defined as the sum of TF-IDF
weights associated with words belonging to the uncertainty dictionary.

Unlike the dictionary index, words contribute according to their
TF-IDF importance rather than simple frequency.
'''

def compute_tfidf_uncertainty(tfidf_matrix,feature_names,dictionary):

    feature_to_idx = {
        word: idx
        for idx, word in enumerate(feature_names)
    }

    uncertainty_cols = [
        feature_to_idx[word]
        for word in dictionary
        if word in feature_to_idx
    ]

    scores = (
        tfidf_matrix[:, uncertainty_cols]
        .sum(axis=1)
        .A1
    )

    return scores

# Attaches scores
def apply_scores(df, scores):

    df["tfidf_uncertainty"] = scores

    return df


#######################################################
### 6. Build monthly index

'''
Aggregate speech-level uncertainty scores to monthly frequency
by averaging all speeches delivered within each month.
'''

def build_index(df):

    index = (
        df.groupby("month")["tfidf_uncertainty"]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    return index


#######################################################
### 7. Normalize index

'''
Standardize the monthly series to facilitate comparison with
the other uncertainty indices constructed in the project.
'''

def normalize_index(df):

    df["uncertainty_index"] = (
        df["tfidf_uncertainty"]
        - df["tfidf_uncertainty"].mean()
    ) / df["tfidf_uncertainty"].std()

    return df


#######################################################
### 8. Save results
def save_results(df, index):

    df.to_pickle(
        "data/tfidf_speeches.pkl"
    )

    index.to_pickle(
        "data/tfidf_uncertainty_index.pkl"
    )

    print("Saved results.")


#######################################################
### MAIN

if __name__ == "__main__":

    df = load_data()

    dictionary = load_dictionary()

    documents = build_documents(df)

    tfidf_matrix, feature_names = fit_tfidf(documents)

    scores = compute_tfidf_uncertainty(
        tfidf_matrix,
        feature_names,
        dictionary
    )

    df = apply_scores(df, scores)

    index = build_index(df)

    index = normalize_index(index)

    index = index.set_index("month")

    save_results(df, index)

    print("\nTop 10 TF-IDF months:")
    print(
        index.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    print("\nDone.")