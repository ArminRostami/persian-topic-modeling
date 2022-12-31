import pandas as pd
import re
from hazm import (
    Lemmatizer,
    InformalNormalizer,
)
import numpy as np
import time

inormalizer = InformalNormalizer()


def remove_hashtags(text):
    text = re.sub("#مهسا_امینی", "", text)
    text = re.sub("\n#\S+", "", text)
    return text


def replace_newline(text):
    table = str.maketrans({"\n": " ", "\u200C": " "})
    return text.translate(table)


def remove_non_farsi(text):
    text = re.sub(r"[^\u0620-\u064A\u0674-\u06D5 ]+", "", text)
    return text


def remove_mentions(text):
    text = re.sub("@\S+", "", text)
    return text


def remove_punctuation(text):
    punctuation = r"""!"#$%&'()*+,-./:;<=>?…“”،٫؟@[\]^_`{|}~"""
    table = str.maketrans(dict.fromkeys(punctuation, " "))
    return text.translate(table)


def clean_text(df):
    df["clean_text"] = (
        df["text"]
        .apply(remove_hashtags)
        .apply(remove_mentions)
        .apply(replace_newline)
        .apply(remove_punctuation)
        .apply(remove_non_farsi)
    )
    return df


def tokenize(text):
    tokens = re.findall("\w+", text)
    return np.array(tokens)


def formalize(word):

    if word in inormalizer.lemmatizer.words or word in inormalizer.lemmatizer.verbs:
        return word

    elif word in inormalizer.iverb_map:
        return inormalizer.iverb_map[word]

    elif word in inormalizer.iword_map:
        return inormalizer.iword_map[word]

    elif word[:-2] in inormalizer.ilemmatizer.verbs and word.endswith("ین"):
        return word[:-1] + "د"

    elif word.endswith("ن") and word[:-1] in inormalizer.ilemmatizer.verbs:
        return word + "د"

    elif (
        word[:-1] in inormalizer.ilemmatizer.verbs
        and word.endswith("ه")
        and word[:-1] not in inormalizer.lemmatizer.words
    ):
        return inormalizer.iword_map.get(word[:-1], word[:-1]) + "د"

    elif (
        word not in inormalizer.ilemmatizer.verbs
        and word.endswith("ه")
        and word[:-1] in inormalizer.ilemmatizer.words
    ):
        return inormalizer.iword_map.get(word[:-1], word[:-1])

    elif (
        word not in inormalizer.ilemmatizer.verbs
        and word.endswith("ون")
        and inormalizer.lemmatizer.lemmatize(word[:-2] + "ان")
        in inormalizer.ilemmatizer.words
    ):
        return word[:-2] + "ان"

    elif inormalizer.seperation_flag:
        return inormalizer.split_token_words(word)

    return word


def main():
    cols = ["tweet_id", "text"]
    df = pd.read_csv("./tweets.csv", usecols=cols, nrows=1000)

    df = clean_text(df)

    df["tokens"] = df["clean_text"].apply(tokenize)
    df["len"] = df["tokens"].apply(len)
    df = df[df["len"] > 5]

    start = time.time()
    vnorm = np.vectorize(formalize)
    df["tokens"] = df["tokens"].apply(vnorm)
    mid = time.time()

    vlemma = np.vectorize(Lemmatizer().lemmatize)
    df["tokens"] = df["tokens"].apply(vlemma)
    end = time.time()

    df[["tweet_id", "text", "tokens"]].to_csv("./clean.csv")

    print("normalization time: ", mid - start)
    print("lemmatization time: ", end - mid)


if __name__ == "__main__":
    main()
