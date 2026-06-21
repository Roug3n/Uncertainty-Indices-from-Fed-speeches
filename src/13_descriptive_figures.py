"""
Creates descriptive tables and figures for the thesis.
"""

import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from collections import Counter


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
# LOAD DATA

raw_df = pd.read_pickle(
    "data/all_fed_speeches"
)

corpus = pd.read_pickle(
    "data/corpus.pkl"
)

raw_df["date"] = pd.to_datetime(raw_df["date"])
corpus["date"] = pd.to_datetime(corpus["date"])


#######################################################
# FIGURE 1
# Raw dataset sample

fig1 = raw_df[
    [
        "date",
        "speaker",
        "title",
        "link",
        "text"
    ]
].head(3)

print("Figure 1 saved")
print(fig1)


print("Figure 1 saved")


#######################################################
# FIGURE 2
# Processed corpus sample

temp = corpus.copy()

temp["text_clean"] = (
    temp["text_clean"]
    .astype(str)
    .str[:120]
    + "..."
)

temp["tokens_lemma"] = (
    temp["tokens_lemma"]
    .astype(str)
    .str[:120]
    + "..."
)

fig2 = temp[
    [
        "date",
        "speaker",
        "is_chair",
        "month",
        "text_clean",
        "tokens_lemma"
    ]
].head(3)

print("Figure 2")
print(fig2)

print("Figure 2 saved")


#######################################################
# FIGURE 3
# Speeches by quarter

quarterly = (
    corpus
    .groupby(
        pd.Grouper(
            key="date",
            freq="QE"
        )
    )
    .size()
)

fig, ax = plt.subplots(figsize=(14,6))

quarterly.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Number of Fed Speeches per Quarter"
)

ax.set_ylabel(
    "Number of Speeches"
)

ax.set_xlabel("")

# Show only one label per year (Q1 positions)
positions = [
    i for i, d in enumerate(quarterly.index)
    if d.quarter == 1
]

labels = [
    str(d.year)
    for d in quarterly.index
    if d.quarter == 1
]

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45)

plt.tight_layout()

plt.savefig(
    "figures/figure3_speeches_per_quarter.png",
    dpi=300
)

plt.close()

print("Figure 3 saved")


#######################################################
# FIGURE 3.2
# Chairman speeches by quarter

chair = corpus[
    corpus["is_chair"]
].copy()

chair["chairman_group"] = "Other"

chair.loc[
    chair["speaker"]
    .str.contains("Bernanke", case=False, na=False),
    "chairman_group"
] = "Bernanke"

chair.loc[
    chair["speaker"]
    .str.contains("Yellen", case=False, na=False),
    "chairman_group"
] = "Yellen"

chair.loc[
    chair["speaker"]
    .str.contains("Powell", case=False, na=False),
    "chairman_group"
] = "Powell"

quarter_chair = (
    chair
    .groupby(
        [
            pd.Grouper(
                key="date",
                freq="QE"
            ),
            "chairman_group"
        ]
    )
    .size()
    .unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(14,6))

quarter_chair.plot(
    kind="bar",
    stacked=True,
    ax=ax
)

ax.set_title(
    "Chairman Speeches per Quarter"
)

ax.set_ylabel(
    "Number of Speeches"
)

ax.set_xlabel("")

# Show only Q1 labels
positions = [
    i for i, d in enumerate(quarter_chair.index)
    if d.quarter == 1
]

labels = [
    str(d.year)
    for d in quarter_chair.index
    if d.quarter == 1
]

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45)

plt.tight_layout()

plt.savefig(
    "figures/figure3_2_chair_speeches.png",
    dpi=300
)

plt.close()

print("Figure 3.2 saved")

#######################################################
# FIGURE 5
# Wordcloud BEFORE preprocessing

raw_text = " ".join(
    raw_df["text"]
    .dropna()
    .astype(str)
)

wc = WordCloud(
    width=1800,
    height=900,
    background_color="white"
)

wc.generate(raw_text)

plt.figure(figsize=(14,7))

plt.imshow(
    wc,
    interpolation="bilinear"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "figures/figure5_wordcloud_before.png",
    dpi=300
)

plt.close()

print("Figure 5 saved")


#######################################################
# FIGURE 6
# Wordcloud AFTER preprocessing

clean_text = " ".join(
    [
        word
        for doc in corpus["tokens_lemma"].dropna()
        for word in doc
    ]
)

wc = WordCloud(
    width=1800,
    height=900,
    background_color="white"
)

wc.generate(clean_text)

plt.figure(figsize=(14,7))

plt.imshow(
    wc,
    interpolation="bilinear"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "figures/figure6_wordcloud_after.png",
    dpi=300
)

plt.close()

print("Figure 6 saved")


#######################################################
# FIGURE 7
# Powell wordcloud + top words

powell = corpus[
    corpus["speaker"]
    .str.contains(
        "Powell",
        case=False,
        na=False
    )
]

powell_text = " ".join(
    [
        word
        for doc in powell["tokens_lemma"].dropna()
        for word in doc
    ]
)

# Wordcloud

wc = WordCloud(
    width=1800,
    height=900,
    background_color="white"
)

wc.generate(powell_text)

plt.figure(figsize=(14,7))

plt.imshow(
    wc,
    interpolation="bilinear"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "figures/figure7_powell_wordcloud.png",
    dpi=300
)

plt.close()

# Top frequencies

powell_words = powell_text.split()

powell_freq = pd.DataFrame(
    Counter(powell_words).most_common(20),
    columns=["word","frequency"]
)

print("\n")
print("=" * 60)
print("FIGURE 7A - POWELL TOP 20 WORDS")
print("=" * 60)
print(powell_freq.to_string(index=False))

print("Figure 7 saved")

#######################################################
# FIGURE 8
# Bernanke wordcloud + top words

bernanke = corpus[
    corpus["speaker"]
    .str.contains(
        "Bernanke",
        case=False,
        na=False
    )
]

bernanke_text = " ".join(
    [
        word
        for doc in bernanke["tokens_lemma"].dropna()
        for word in doc
    ]
)

# Wordcloud

wc = WordCloud(
    width=1800,
    height=900,
    background_color="white"
)

wc.generate(bernanke_text)

plt.figure(figsize=(14,7))

plt.imshow(
    wc,
    interpolation="bilinear"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "figures/figure8_bernanke_wordcloud.png",
    dpi=300
)

plt.close()

# Top frequencies

bernanke_words = bernanke_text.split()

bernanke_freq = pd.DataFrame(
    Counter(bernanke_words).most_common(20),
    columns=["word","frequency"]
)

print("\n")
print("=" * 60)
print("FIGURE 8A - BERNANKE TOP 20 WORDS")
print("=" * 60)
print(bernanke_freq.to_string(index=False))

print("Figure 8 saved")

#######################################################
# FIGURE 9
# NLP preprocessing example

def lowercase_text(text):
    return text.lower()

# Remove punctuation
def remove_punctuation(text):
    return re.sub(r"[^\w\s]", " ", text)

# Tokenization
def tokenize(text):
    return text.split()

# Lemmatization
def lemmatize(tokens):

    doc = nlp(" ".join(tokens))

    return [
        token.lemma_
        for token in doc
    ]

# Remove Stopwords
def remove_stopwords(tokens):
    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]




example_raw = raw_df.iloc[0]["text"][:500]

example_lower = lowercase_text(
    example_raw
)

example_nopunct = remove_punctuation(
    example_lower
)

example_tokens = tokenize(
    example_nopunct
)

example_lemma = lemmatize(
    example_tokens
)

example_stop = remove_stopwords(
    example_lemma
)

fig9 = pd.DataFrame(
    {
        "step":[
            "Raw text",
            "Lowercase",
            "Remove punctuation",
            "Tokenization",
            "Lemmatization",
            "Stopword removal"
        ],
        "output":[
            example_raw,
            example_lower[:270] ,
            example_nopunct[:270] ,
            str(example_tokens),
            str(example_lemma),
            str(example_stop)
        ]
    }
)

print("\n")
print("=" * 60)
print("FIGURE 9 - NLP PREPROCESSING PIPELINE")
print("=" * 60)
print(fig9.to_string(index=False))




all_words = clean_text.split()

all_freq = pd.DataFrame(
    Counter(all_words).most_common(20),
    columns=["word","frequency"]
)

print("\n")
print("=" * 60)
print("FIGURE 6A - CORPUS TOP 20 WORDS")
print("=" * 60)
print(all_freq.to_string(index=False))