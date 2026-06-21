'''
In this script we load and make small adjustments to the scraped data for the NPL analysis
 This script:
 1. Loads raw scraped speeches
 2. Cleans and normalizes text
 3. Creates metadata features
 4. Saves a reusable NLP corpus
'''
import pandas as pd
import re
import nltk
import spacy

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

from nltk.corpus import stopwords

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))


#######################################################
### 1. Load data

def load_data(path="data/all_fed_speeches"):
    df = pd.read_pickle(path)
    print("Initial shape:", df.shape)
    return df

#######################################################
### 2. Feature engineering

def add_features(df):

    # Word count
    df["word_count"] = df["text"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

    # Year / Month 
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M")

    # Chairman dummy 
    df["speaker_clean"] = (
        df["speaker"]
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    df["is_chair"] = df["speaker_clean"].str.contains("chair", case=False, na=False) & \
                    ~df["speaker_clean"].str.contains("vice", case=False, na=False)

    return df


#######################################################
### 3. Remove bad observations

def filter_data(df):

    # Remove empty or very short speeches
    df = df[df["word_count"] > 100]

    # Remove missing text
    df = df[df["text"].notnull()]

    df = df.reset_index(drop=True)

    print("After filtering:", df.shape)

    return df

#######################################################
### 4. NLP preprocessing

# Convert to lowercase

def lowercase_text(text):
    return text.lower()

# Remove punctuation
def remove_punctuation(text):
    return re.sub(r"[^\w\s]", " ", text)

# Tokenization
def tokenize(text):
    return text.split()


'''
Lemmatization reduces inflected forms to a common base form
(e.g. "markets", "marketed" -> "market")
'''
# Lemmatization
def lemmatize(tokens):

    doc = nlp(" ".join(tokens))

    return [
        token.lemma_
        for token in doc
    ]


'''
Stopwords are removed to reduce noise and focus the analysis 
on economically meaningful terms. Common functional words
(e.g. "the", "and", "of") generally carry little information.
However, this is only used for dictionary approaches, not semantic ones
'''
# Remove Stopwords
def remove_stopwords(tokens):
    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

def build_corpus(df):
   
    '''
    We create three new columns, to preserve each stage of the text for different analysis
    '''
    df["text_clean"] = (
        df["text"]
        .apply(lowercase_text)
        .apply(remove_punctuation)
    )

    df["tokens"] = (
        df["text_clean"]
        .apply(tokenize)
    )

    df["tokens_lemma"] = (
        df["tokens"]
        .apply(lemmatize)
    )

    df["tokens_lemma"] = (
        df["tokens_lemma"]
        .apply(remove_stopwords)
    )

    return df


#######################################################
### 4. Save clean dataset

def save_data(df, path="data/corpus.pkl"):
    df.to_pickle(path)
    print("Saved cleaned dataset to:", path)


#######################################################
### MAIN

if __name__ == "__main__":

    df = load_data()

    df = add_features(df)

    df = filter_data(df)

    df = build_corpus(df)

    save_data(df)

    print("\nDone")

    print("\nNumber of speeches:")
    print(len(df))

    print("\nDate range:")
    print(df["date"].min())
    print(df["date"].max())

