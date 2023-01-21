from hazm import Normalizer, Lemmatizer, InformalNormalizer
import re
import numpy as np
import time
import gensim


def read_stop_words():
    stop_list = []

    with open("stop_list.txt", "r") as list:
        for word in list:
            if not word.startswith("-"):
                stop_list.append(word.strip("\n"))

    return stop_list


def remove_hashtags(text):
    text = re.sub(r"#مهسا_امینی", "", text)
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


def formalize(word, inormalizer):
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


def create_dictionary(df):
    dictionary = gensim.corpora.Dictionary(df["tokens"])
    stop_list = read_stop_words()
    bad_ids = [dictionary.token2id[word] for word in stop_list]
    dictionary.filter_tokens(bad_ids)
    dictionary.filter_extremes()

    # dictionary.save_as_text("./dict.txt")
    return dictionary


def apply_dict_to_docs(df, dictionary):
    df["tokens"] = df["tokens"].apply(dictionary.doc2idx)

    def del_nones(list):
        arr = np.array(list)
        return arr[arr != -1]

    df["tokens"] = df["tokens"].apply(del_nones)

    vget = np.vectorize(dictionary.get, otypes="U")
    df["tokens"] = df["tokens"].apply(vget)

    df["len"] = df["tokens"].apply(len)
    df = df[df["len"] > 0]

    return df


def preprocess(df):
    start = time.time()

    has_likes = df["likes"] > 0
    df = df[has_likes]

    df = clean_text(df)
    vnorm = np.vectorize(Normalizer().normalize)
    df["clean_text"] = df["clean_text"].apply(vnorm)

    df["tokens"] = df["clean_text"].apply(tokenize)
    df["len"] = df["tokens"].apply(len)
    df = df[df["len"] > 0]

    inormalizer = InformalNormalizer()
    vinorm = np.vectorize(lambda x: formalize(x, inormalizer))
    df["tokens"] = df["tokens"].apply(vinorm)

    vlemma = np.vectorize(Lemmatizer().lemmatize)
    df["tokens"] = df["tokens"].apply(vlemma)

    df["tokens"] = df["tokens"].apply(np.unique)

    df["len"] = df["tokens"].apply(len)
    df = df[df["len"] > 0]
    print("preprocessing time: ", time.time() - start)

    return df
