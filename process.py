from hazm import Normalizer, Lemmatizer, InformalNormalizer
import re
import numpy as np
import time


stop_list = []
inormalizer = InformalNormalizer()


def read_stop_words():
    with open("stop_list.txt", "r") as list:
        for word in list:
            if not word.startswith("-"):
                stop_list.append(word.strip("\n"))


def remove_hashtags(text):
    text = re.sub(r"#مهسا_امینی", "", text)
    # text = re.sub("\n\s*#.*", "", text)
    text = re.sub(r"\n\s*#[^\n]*", "", text)
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
    df.loc[:, "clean_text"] = (
        df.loc[:, "text"]
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
    abbrs = {"ج": "جمهوری", "ا": "اسلام", "خونه": "خانه"}
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


def preprocess(df):
    start = time.time()

    has_likes = df["likes"] > 0
    df = df[has_likes]

    df = clean_text(df)
    vnorm = np.vectorize(Normalizer().normalize)
    df["clean_text"] = df["clean_text"].apply(vnorm)

    df["tokens"] = df["clean_text"].apply(tokenize)
    df["len"] = df["tokens"].apply(len)
    df = df[df["len"] > 5]

    vinorm = np.vectorize(formalize)
    df["tokens"] = df["tokens"].apply(vinorm)

    vlemma = np.vectorize(Lemmatizer().lemmatize)
    df["tokens"] = df["tokens"].apply(vlemma)

    df["tokens"] = df["tokens"].apply(np.unique)

    # read_stop_words()
    # df["tokens"] = df["tokens"].apply(delete_stop_words)

    df["len"] = df["tokens"].apply(len)
    df = df[df["len"] > 5]
    print("preprocessing time: ", time.time() - start)

    return df


if __name__ == "__main__":
    read_stop_words()
    print(stop_list)
