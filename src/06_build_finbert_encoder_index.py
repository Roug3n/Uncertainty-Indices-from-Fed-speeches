'''
In this script we create an index applying FinBERT encoding methodology 
with the model finbert-tone developed by Yang et al.(2020)

This index follows the same semantic-centroid methodology used in
05_build_embedding_index.py but replaces the general-purpose MiniLM
encoder with FinBERT.

Because FinBERT was trained on financial text, it may capture
economic and monetary-policy language more accurately than
general-purpose sentence embeddings.
'''

import pandas as pd
import numpy as np
import torch
import os

# Maximizes computation capacity
torch.set_num_threads(
    max(1, os.cpu_count() - 1)
)

print(
    "PyTorch threads:",
    torch.get_num_threads()
)

from tqdm import tqdm

tqdm.pandas()

from transformers import BertTokenizer, BertModel

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
### 2. Load FinBERT encoder

'''
FinBERT is used only as an encoder.
We do not use its sentiment classifier.

Instead, hidden-state embeddings are extracted and used
to construct semantic uncertainty measures.
'''

def load_model():

    print("Loading FinBERT encoder")

    tokenizer = BertTokenizer.from_pretrained(
        "yiyanghkust/finbert-tone"
    )

    model = BertModel.from_pretrained(
        "yiyanghkust/finbert-tone"
    )

    model.eval()

    return tokenizer, model


#######################################################
### 3. Mean pooling

'''
Converts token-level embeddings into a single vector
representation for the entire text sequence.

Mean pooling is commonly used when FinBERT is employed
as a sentence/document encoder.
'''

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    pooled = (
        token_embeddings * mask
    ).sum(dim=1) / torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return pooled


#######################################################
### 4. Batched FinBERT encoding

'''
Texts are encoded in batches to reduce memory usage
and improve computational efficiency
'''

def encode_texts(
    texts,
    tokenizer,
    model,
    batch_size=64
):

    all_embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        '''
        FinBERT inherits BERT's maximum context length.
        Longer inputs are truncated, which motivates the
        chunking procedure used later
        '''

        with torch.no_grad():

            outputs = model(**inputs)

        embeddings = mean_pooling(
            outputs,
            inputs["attention_mask"]
        )

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

    return np.vstack(all_embeddings)


#######################################################
### 5. Build uncertainty centroids

'''
The same semantic reference sentences used in the
MiniLM index are employed here to ensure that any
differences between indices arise from the encoder
rather than from changes in the uncertainty definition
'''

def build_uncertainty_centroid(
    tokenizer,
    model
):

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

    uncertainty_centroid = encode_texts(
        speaker_uncertainty_sentences,
        tokenizer,
        model
    ).mean(axis=0)

    bad_news_centroid = encode_texts(
        bad_news_sentences,
        tokenizer,
        model
    ).mean(axis=0)

    discussion_centroid = encode_texts(
        discussion_uncertainty_sentences,
        tokenizer,
        model
    ).mean(axis=0)

    return (
        uncertainty_centroid,
        bad_news_centroid,
        discussion_centroid
    )


#######################################################
### 6. Chunk speeches

def chunk_sentences(
    text,
    target_words=100
):

    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:

        n_words = len(sentence.split())

        if current_words + n_words > target_words and current_chunk:

            chunks.append(
                " ".join(current_chunk)
            )

            current_chunk = [sentence]
            current_words = n_words

        else:

            current_chunk.append(sentence)
            current_words += n_words

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


#######################################################
### 7. Speech score

def compute_similarities(
    text,
    tokenizer,
    model,
    uncertainty_centroid,
    bad_news_centroid,
    discussion_centroid,
):

    chunks = chunk_sentences(
        text,
        target_words=100
    )

    if len(chunks) == 0:

        return {
            "sim_u": np.nan,
            "sim_b": np.nan,
            "sim_d": np.nan
        }

    chunk_embeddings = encode_texts(
        chunks,
        tokenizer,
        model
    )

    sim_u = cosine_similarity(
        chunk_embeddings,
        uncertainty_centroid.reshape(1, -1)
    ).flatten()

    sim_b = cosine_similarity(
        chunk_embeddings,
        bad_news_centroid.reshape(1, -1)
    ).flatten()

    sim_d = cosine_similarity(
        chunk_embeddings,
        discussion_centroid.reshape(1, -1)
    ).flatten()

    '''
    Stores similarities separately so the final uncertainty
    score can be constructed using alternative weighting schemes.
    '''

    return {
        "sim_u": sim_u.mean(),
        "sim_b": sim_b.mean(),
        "sim_d": sim_d.mean()
    }


#######################################################
### 8. Apply index

def compute_all_similarities(
    df,
    tokenizer,
    model,
    uncertainty_centroid,
    bad_news_centroid,
    discussion_centroid
):

    print("Computing similarities")

    sims = df["text"].progress_apply(
        lambda x: compute_similarities(
            x,
            tokenizer,
            model,
            uncertainty_centroid,
            bad_news_centroid,
            discussion_centroid
        )
    )

    sims = pd.DataFrame(list(sims))

    df["sim_u"] = sims["sim_u"]
    df["sim_b"] = sims["sim_b"]
    df["sim_d"] = sims["sim_d"]

    return df


#######################################################
### 9. Monthly aggregation

def build_index(
    df,
    column_name
):

    index = (
        df.groupby("month")[column_name]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    return index


#######################################################
### 10. Normalize

def normalize_index(
    df,
    column_name
):

    mean = df[column_name].mean()

    std = df[column_name].std()

    df["uncertainty_index"] = (
        df[column_name] - mean
    ) / std

    return df


#######################################################
### MAIN

if __name__ == "__main__":

    df = load_data()

    tokenizer, model = load_model()

    (
        uncertainty_centroid,
        bad_news_centroid,
        discussion_centroid
    ) = build_uncertainty_centroid(
        tokenizer,
        model
    )

    ###################################################
    ### Version 1: 0.5 / 0.5
    # Baseline specification
    # uncertainty - 0.5 * bad news - 0.5 * discussion

    df = compute_all_similarities(
        df,
        tokenizer,
        model,
        uncertainty_centroid,
        bad_news_centroid,
        discussion_centroid
    )

    df[
        [
            "date",
            "month",
            "speaker",
            "is_chair",
            "sim_u",
            "sim_b",
            "sim_d"
        ]
    ].to_pickle(
        "data/finbert_similarity_components.pkl"
    )


    df["finbert_uncertainty"] = (
        df["sim_u"]
        - 0.5 * df["sim_b"]
        - 0.5 * df["sim_d"]
    )

    df[
        [
            "date",
            "month",
            "speaker",
            "is_chair",
            "finbert_uncertainty"
        ]
    ].to_pickle(
        "data/finbert_encoder_speech_scores.pkl"
    )

    df_chair = df[df["is_chair"]].copy()

    index = build_index(
        df,
        "finbert_uncertainty"
    )
    index = normalize_index(
        index,
        "finbert_uncertainty"
    )
    index = index.set_index("month")

    print("\nTop 10 months finbert encoder 0.5/0.5:")

    print(
        index.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    chair_index = build_index(
        df_chair,
        "finbert_uncertainty"
    )
    chair_index = normalize_index(
        chair_index,
        "finbert_uncertainty"
    )
    chair_index = chair_index.set_index("month")

    print("\nTop 10 months chair finbert encoder 0.5/0.5:")
    print(
        chair_index.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    index.to_pickle(
        "data/finbert_encoder_index.pkl"
    )

    chair_index.to_pickle(
        "data/finbert_encoder_chair_index.pkl"
    )

    ###################################################
    ### Version 2: 0.25 / 1
    # Pure uncertainty specification
    # Places stronger emphasis on removing speeches that merely discuss uncertainty as a topic

    df["finbert_uncertainty_pure"] = (
        df["sim_u"]
        - 0.25 * df["sim_b"]
        - 1.0 * df["sim_d"]
    )

    df[
        [
            "date",
            "month",
            "speaker",
            "is_chair",
            "finbert_uncertainty_pure"
        ]
    ].to_pickle(
        "data/finbert_encoder_speech_scores_pure.pkl"
    )

    df_chair = df[df["is_chair"]].copy()

    index_pure = build_index(
        df,
        "finbert_uncertainty_pure"
    )

    index_pure = normalize_index(
        index_pure,
        "finbert_uncertainty_pure"
    )

    index_pure = index_pure.set_index("month")

    print("\nTop 10 months finbert encoder 0.25/1:")

    print(
        index_pure.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    chair_index_pure = build_index(
        df_chair,
        "finbert_uncertainty_pure"
    )

    chair_index_pure = normalize_index(
        chair_index_pure,
        "finbert_uncertainty_pure"
    )

    chair_index_pure = chair_index_pure.set_index("month")

    print("\nTop 10 months chair finbert encoder 0.25/1:")

    print(
        chair_index_pure.sort_values(
            "uncertainty_index",
            ascending=False
        ).head(10)
    )

    index_pure.to_pickle(
        "data/finbert_encoder_pure_index.pkl"
    )

    chair_index_pure.to_pickle(
        "data/finbert_encoder_chair_pure_index.pkl"
    )

    print("\nDone.")

