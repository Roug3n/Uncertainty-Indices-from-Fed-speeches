'''
In this script we create an index applying embedding metholody with the model all-MiniLM-L6-v2

Instead of counting uncertainty-related words, this approach measures
semantic similarity between speech content and a set of prototype
sentences representing speaker uncertainty.

Each speech is embedded into a semantic vector space using MiniLM,
and uncertainty is measured according to its proximity to an
uncertainty centroid.

Additional centroids are introduced to distinguish genuine speaker
uncertainty from:

1. Negative economic news
2. Discussions about uncertainty as a topic

This helps isolate uncertainty expressed by the policymaker rather
than uncertainty merely being mentioned.
'''

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import sent_tokenize

# Required for sentence segmentation used during chunking
nltk.download("punkt")
nltk.download("punkt_tab")

#######################################################
### 1. Load data

def load_data(path="data/corpus.pkl"):

    df = pd.read_pickle(path)

    print("Loaded corpus:", df.shape)

    return df


#######################################################
### 2. Load embedding model

def load_model():

    print("Loading MiniLM model")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


#######################################################
### 3. Build uncertainty centroid


def build_uncertainty_centroid(model):

    '''
    The following sentences depict uncertainty, ambiguity and difficulty to asess, which relates to our speaker uncertainty
    '''

    speaker_uncertainty_sentences = [

        "The outlook remains highly uncertain.",
        "Future developments are difficult to predict.",
        "Forecasting economic conditions is particularly challenging.",
        "We face unusually large forecasting uncertainty.",
        "The economic outlook is unusually difficult to assess.",
        "There is a wide range of possible outcomes.",
        "Several materially different scenarios remain possible.",
        "We cannot rule out several plausible outcomes.",
        "The balance of risks is difficult to evaluate.",
        "The economy may evolve in unexpected ways.",
        "The probability distribution of future outcomes is highly uncertain.",
        "Historical relationships may not provide reliable guidance.",
        "Existing models may not adequately capture current conditions.",
        "Recent developments have increased uncertainty about the outlook.",
        "The effects of recent shocks remain unclear.",
        "We have limited visibility into future economic conditions.",
        "The available information does not allow a confident forecast.",
        "The future path of inflation is especially uncertain.",
        "The future path of economic activity is difficult to assess.",
        "Unexpected shocks could materially alter the outlook.",
        "The appropriate policy response depends on uncertain future developments.",
        "The outlook is clouded by substantial uncertainty.",
        "Incoming information has increased uncertainty about future conditions.",
        "The economy is operating in an environment of elevated uncertainty.",
        "We must be prepared for outcomes that differ substantially from expectations."

    ]

    '''
    The following sentences create an opposing view, with pessimistic yet not uncertain economic statements
    The objective is to prevent the index from confusing bad economic news with uncertainty about future outcomes. 
    '''

    bad_news_sentences = [

        "GDP growth has slowed substantially.",
        "Economic activity has weakened.",
        "Inflation remains elevated.",
        "Unemployment has increased.",
        "Credit conditions have tightened.",
        "Financial markets are under stress.",
        "Consumer spending has weakened.",
        "Business investment has declined.",
        "The economy has entered recession.",
        "Labor market conditions have deteriorated.",
        "Growth remains weak.",
        "The economy is contracting.",
        "Inflationary pressures remain strong.",
        "Financial conditions have worsened.",
        "Employment growth has slowed.",
        "Economic indicators remain weak.",
        "Demand has fallen significantly.",
        "The housing market remains weak.",
        "Business confidence has declined.",
        "Economic conditions remain challenging.",
        "Systemic risks have increased.",
        "Financial market functioning has deteriorated.",
        "Financial stability risks have risen.",
        "Credit markets are under strain.",
        "Market participants face elevated stress.",
        "Risk spreads have widened substantially.",
        "The financial system faces significant vulnerabilities.",
        "Funding markets are experiencing disruptions.",
        "Downside risks to growth have increased.",
        "Financial conditions have tightened materially."
    ]

    '''
    The following sentences create an complimentary opposition, with statements about discussing uncertainty rather than being uncertain themselves
    This centroid helps separate talking about uncertainty from actually communicating uncertainty.
    '''

    discussion_uncertainty_sentences = [

        "Today I would like to discuss uncertainty.",
        "The topic of this speech is uncertainty.",
        "This symposium focuses on uncertainty.",
        "This conference examines uncertainty.",
        "Researchers have studied uncertainty extensively.",
        "The academic literature on uncertainty is extensive.",
        "Economic models attempt to incorporate uncertainty.",
        "Theory provides insights into uncertainty.",
        "Managing uncertainty is a central topic in economics.",
        "Uncertainty is an important concept in policymaking.",
        "We consider how economists think about uncertainty.",
        "This paper examines uncertainty.",
        "The literature provides different approaches to uncertainty.",
        "Forecasting models incorporate uncertainty in various ways.",
        "Economists have long debated how to measure uncertainty.",
        "This discussion concerns uncertainty as a general concept.",
        "The purpose of these remarks is to discuss forecasting.",
        "Today I will discuss the forecasting process.",
        "Forecasting is inherently difficult.",
        "This speech focuses on the challenges of forecasting."

    ]

    uncertainty_embeddings = model.encode(
        speaker_uncertainty_sentences,
        show_progress_bar=False
    )

    uncertainty_centroid = uncertainty_embeddings.mean(axis=0)

    bad_news_embeddings = model.encode(
        bad_news_sentences,
        show_progress_bar=False
    )

    bad_news_centroid = bad_news_embeddings.mean(axis=0)

    discussion_embeddings = model.encode(
        discussion_uncertainty_sentences,
        show_progress_bar=False
    )

    discussion_centroid = discussion_embeddings.mean(axis=0)

    return uncertainty_centroid, bad_news_centroid, discussion_centroid


#######################################################
### 4. Chunking speeches maintaining sentence structure

'''
Long speeches exceed the optimal input size for sentence embeddings.
Therefore speeches are divided into coherent sentence-based chunks
of approximately 100 words and scored separately.
'''

def chunk_sentences(text, target_words=100):

    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:

        n_words = len(sentence.split())

        if current_words + n_words > target_words and current_chunk:

            chunks.append(" ".join(current_chunk))

            current_chunk = [sentence]
            current_words = n_words

        else:

            current_chunk.append(sentence)
            current_words += n_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


#######################################################
### 5. Compute speech similarity

'''
Speech uncertainty score:

score = similarity(uncertainty centroid) - α * similarity(bad news centroid) - β * similarity(discussion centroid)

where α and β are weighting parameters
'''

def compute_similarity(text, model, uncertainty_centroid, bad_news_centroid, discussion_centroid, bad_weight=0.5, discussion_weight=0.5):

    chunks = chunk_sentences(
        text,
        target_words=100
    )

    if len(chunks) == 0:
        return np.nan

    chunk_embeddings = model.encode(
        chunks,
        show_progress_bar=False
    )

    sim_uncertainty = cosine_similarity(
        chunk_embeddings,
        uncertainty_centroid.reshape(1,-1)
    ).flatten()

    sim_badnews = cosine_similarity(
        chunk_embeddings,
        bad_news_centroid.reshape(1,-1)
    ).flatten()

    sim_discussion = cosine_similarity(
        chunk_embeddings,
        discussion_centroid.reshape(1,-1)
    ).flatten()

    score = sim_uncertainty - bad_weight*sim_badnews - discussion_weight*sim_discussion

    '''
    The final speech score is the average uncertainty score
    across all chunks belonging to the speech.
    '''

    return score.mean()



#######################################################
### 6. Apply to all speeches

def apply_embedding_index(df, model, uncertainty_centroid, bad_news_centroid, discussion_centroid, bad_weight, discussion_weight):

    print("Embedding speeches")

    df["embedding_uncertainty"] = df["text"].apply(
        lambda x: compute_similarity(
            x,
            model,
            uncertainty_centroid,
            bad_news_centroid,
            discussion_centroid,
            bad_weight,
            discussion_weight,
        )
    )

    return df


#######################################################
### 7. Monthly aggregation

def build_index(df):

    index = (
        df.groupby("month")["embedding_uncertainty"]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    return index


#######################################################
### 8. Normalize index

def normalize_index(df):

    mean = df["embedding_uncertainty"].mean()

    std = df["embedding_uncertainty"].std()

    df["uncertainty_index"] = (
        df["embedding_uncertainty"] - mean
    ) / std

    return df



#######################################################
### MAIN

if __name__ == "__main__":

    df = load_data()

    model = load_model()

    uncertainty_centroid, bad_news_centroid, discussion_centroid= build_uncertainty_centroid(
        model
    )

    # Indices with weight 0.5 for bad_cetroid and 0.5 discussion_centroid
    df = apply_embedding_index(
        df,
        model,
        uncertainty_centroid,
        bad_news_centroid,
        discussion_centroid,
        bad_weight=0.5,
        discussion_weight=0.5
    )

    df_chair = df[df["is_chair"]].copy()

    index = build_index(df)
    index = normalize_index(index)
    index = index.set_index("month")

    chair_index = build_index(df_chair)
    chair_index = normalize_index(chair_index)
    chair_index = chair_index.set_index("month")

    index.to_pickle("data/embedding_index.pkl")
    chair_index.to_pickle("data/embedding_chair_index.pkl")


    # Indices with weight 0.25 for bad_cetroid and 1 discussion_centroid
    df = apply_embedding_index(
        df,
        model,
        uncertainty_centroid,
        bad_news_centroid,
        discussion_centroid,
        bad_weight=0.25,
        discussion_weight=1
    )

    df_chair = df[df["is_chair"]].copy()

    index_pure = build_index(df)
    index_pure = normalize_index(index_pure)
    index_pure = index_pure.set_index("month")

    chair_index_pure = build_index(df_chair)
    chair_index_pure = normalize_index(chair_index_pure)
    chair_index_pure = chair_index_pure.set_index("month")

    index_pure.to_pickle("data/embedding_pure_index.pkl")
    chair_index_pure.to_pickle("data/embedding_chair_pure_index.pkl")


    print("\nTop 10 months embedding 0.5/0.5:")

    print(
        index.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    print("\nTop 10 months embedding chair 0.5/0.5:")

    print(
        chair_index.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    print("\nTop 10 months embedding 0.25/1:")

    print(
        index_pure.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    print("\nTop 10 months embedding chair 0.25/1:")

    print(
        chair_index_pure.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    print("\nDone.")