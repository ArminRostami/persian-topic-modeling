import pandas as pd
import re
from hazm import (
    Lemmatizer,
    InformalNormalizer,
)
import numpy as np
import time
import arabic_reshaper
from bidi.algorithm import get_display
import gensim.corpora as corpora
import gensim

inormalizer = InformalNormalizer()
stop_list = [
    "و",
    "از",
    "با",
    "که",
    "را",
    "به",
    "در",
    "این",
    "تو",
    "برای",
    "آن",
    "می",
    "تا",
    "ما",
    "ها",
    "های",
    "روز",
    "من",
    "هم",
    "بی",
    "یک",
    "ای",
    "رو",
    "هر",
    "یا",
    "شده",
    "باید",
    "بر",
    "ی",
    "کرد#کن",
    "شد#شو",
    "بود#باش",
    "داشت#دار",
    "خواست#خواه",
    "گفت#گو",
    "#است",
    "#هست",
    "داد#ده",
    "دیگر",
    "همه",
    "چه",
    "ولی",
    "شنبه",
    "بازنشر",
    "اگر",
    "دی",
    "ماه",
    "کردن",
    "یلدا",
    "سه",
    "محمد",
    "بعد",
    "کیان",
    "خود",
    "نیست",
    "آذر",
    "دست",
    "انقلاب",
    "ایران",
    "جمهوری",
    "اسلام",
    "مردم",
    "آزاد",
    "زن",
    "زندگی",
    "شب",
    # "تهران",
    # "شهر",
]


def remove_hashtags(text):
    text = re.sub("#مهسا_امینی", "", text)
    text = re.sub("\n#\S+", "", text)
    return text


def replace_newline(text):
    table = str.maketrans({"\n": " "})
    return text.translate(table)


def remove_non_farsi(text):
    text = re.sub(r"[^\u0620-\u064A\u0674-\u06D5\u200C ]+", "", text)
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
    abbrs = {"ج": "جمهوری", "ا": "اسلام"}
    if word in abbrs:
        return abbrs[word]

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


def delete_stop_words(tokens):
    criteria = np.where(np.isin(tokens, stop_list))
    return np.delete(tokens, criteria)


def create_model(df):
    id2word = corpora.Dictionary(df["tokens"])
    num_topics = 5
    corpus = df["tokens"].apply(id2word.doc2bow)

    # TODO: optimize func params
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus, id2word=id2word, num_topics=num_topics, passes=10
    )
    # TODO: write topic number to df

    topics = lda_model.print_topics()
    for i, topic in enumerate(topics):
        print(f"topic {i+1}: ", get_display(arabic_reshaper.reshape(topic[1])), "\n")


def main():
    cols = ["tweet_id", "text"]
    df = pd.read_csv("./tweets.csv", usecols=cols)

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

    df["tokens"] = df["tokens"].apply(delete_stop_words)
    df["len"] = df["tokens"].apply(len)
    df = df[df["len"] > 5]

    create_model(df)

    df[["tweet_id", "tokens", "text"]].head(100000).to_csv("./clean.csv")

    # print("normalization time: ", mid - start)
    # print("lemmatization time: ", end - mid)


if __name__ == "__main__":
    main()
#
