import pandas as pd
import re
from hazm import Normalizer, Lemmatizer, InformalNormalizer
import numpy as np
import time
from random import choice, randint
from multiprocessing import Pool, cpu_count
import os.path as path

import gensim
from gsdmm import MovieGroupProcess

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
    "ویروس",
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
    "خیلی",
    "بازنشر",
    "اگر",
    "نیست",
    "کردن",
    "بعد",
    "خود",
    "دست",
    "روز",
    "دی",
    "آذر",
    "شنبه",
    "یکشنبه",
    "ماه",
    "انقلاب",
    # "جمهوری",
    # "اسلام",
    # "سه",
    #
    # "محمد",
    # "کیان",
    # "یلدا",
    # "ایران",
    # "مردم",
    # "آزاد",
    # "زن",
    # "زندگی",
    # "شب",
    # "تهران",
    # "شهر",
]


def remove_hashtags(text):
    text = re.sub("#مهسا_امینی", "", text)
    # text = re.sub("\n\s*#.*", "", text)
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


def get_topics_lists(model, top_clusters, n_words):
    # create empty list to contain topics
    topics = []

    # iterate over top n clusters
    for cluster in top_clusters:
        # create sorted dictionary of word distributions
        sorted_dict = sorted(
            model.cluster_word_distribution[cluster].items(),
            key=lambda k: k[1],
            reverse=True,
        )[:n_words]

        # create empty list to contain words
        topic = []

        # iterate over top n words in topic
        for k, _ in sorted_dict:
            # append words to topic list
            topic.append(k)

        # append topics to topics list
        topics.append(topic)

    return topics


def get_topic_top_words(cluster_word_distribution, top_cluster, values):
    res = ""
    for cluster in top_cluster:
        sort_dicts = sorted(
            cluster_word_distribution[cluster].items(),
            key=lambda k: k[1],
            reverse=True,
        )[:values]
        res += f"\nCluster {cluster} : {sort_dicts}"
    return res


def create_dictionary(docs, remove_n):

    dictionary = gensim.corpora.Dictionary(docs)

    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
    dictionary.filter_n_most_frequent(remove_n)

    # dictionary.save_as_text("./dict.txt")

    return dictionary


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

    df["tokens"] = df["tokens"].apply(delete_stop_words)

    df["len"] = df["tokens"].apply(len)
    df = df[df["len"] > 5]
    print("preprocessing time: ", time.time() - start)

    return df


def create_gsdmm_model(args):
    start = time.time()
    df, hp = args
    np.random.seed(hp["random_state"])

    docs = df["tokens"]

    dictionary = create_dictionary(docs, hp["remove_n"])

    vocab_length = len(dictionary)

    gsdmm = MovieGroupProcess(
        K=hp["num_topics"], alpha=hp["alpha"], beta=hp["beta"], n_iters=hp["iters"]
    )

    topic_model = gsdmm.fit(docs, vocab_length)
    print("modelling time: ", time.time() - start)

    doc_count = np.array(gsdmm.cluster_doc_count)
    print("Number of documents per topic :", doc_count)

    # Topics sorted by the number of document they are allocated to
    top_index = doc_count.argsort()[-15:][::-1]

    # print("Most important clusters (by number of docs inside):", top_index)
    # print(get_topic_top_words(gsdmm.cluster_word_distribution, top_index, 20))

    topics = get_topics_lists(gsdmm, top_index, 20)

    return topic_model, topics, dictionary, hp


def get_coherence(topics, dictionary, docs):
    coherence = -1
    try:
        coherence = gensim.models.CoherenceModel(
            topics=topics,
            dictionary=dictionary,
            texts=docs,
            coherence="c_v",
        ).get_coherence()
    except:
        print("\ncould not calculate coherence.")

    return coherence


def random_search(df, num_runs):
    topic_count = range(4, 30, 2)
    iter_count = range(5, 30, 5)
    hprange = [0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5]
    remove_range = range(0, 80, 10)

    tasks = []
    for _ in range(num_runs):
        hparams = {
            "random_state": randint(0, 100000),
            "num_topics": choice(topic_count),
            "iters": choice(iter_count),
            "alpha": choice(hprange),
            "beta": choice(hprange),
            "remove_n": choice(remove_range),
        }
        tasks.append((df, hparams))

    NUM_WORKERS = cpu_count()
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(create_gsdmm_model, tasks)

    results_df = pd.DataFrame()
    if path.exists("./hparams.csv"):
        results_df = pd.read_csv("./hparams.csv", index_col=0)

    max_score = 0
    best_params = None

    for res in results:
        topic_model, topics, dictionary, hp = res
        cs = get_coherence(topics, dictionary, df["tokens"])
        hp["coherence"] = cs
        results_df = pd.concat([results_df, pd.DataFrame([hp])], ignore_index=True)

        if cs > max_score:
            max_score = cs
            best_params = hp

    results_df.to_csv("./hparams.csv")

    return best_params


def main():

    cols = ["tweet_id", "text", "likes"]
    df = pd.read_csv("./tweets.csv", usecols=cols)

    df = preprocess(df)
    best_params = random_search(df, 20)

    print("best run params: ", best_params)


if __name__ == "__main__":
    main()
