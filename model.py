import pandas as pd
from random import choice, randint
from multiprocessing import Pool, cpu_count
import os.path as path
import time
import numpy as np
from process import preprocess

import gensim
from gsdmm import MovieGroupProcess


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
        if len(topic) > 0:
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
    top_index = doc_count.argsort()[::-1]

    # print("Most important clusters (by number of docs inside):", top_index)
    print(get_topic_top_words(gsdmm.cluster_word_distribution, top_index, 20))

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
    hparam_file = "./hparams4.csv"
    topic_count = range(10, 20, 2)
    iter_count = range(5, 25, 5)
    hprange = [0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4]
    remove_range = range(80, 130, 10)

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
    if path.exists(hparam_file):
        results_df = pd.read_csv(hparam_file, index_col=0)

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

    results_df.to_csv(hparam_file)

    return best_params


def run_with_params(df, hp):
    topic_model, topics, dictionary, hp = create_gsdmm_model((df, hp))
    cs = get_coherence(topics, dictionary, df["tokens"])
    print("coherence: ", cs)
    df["topic"] = topic_model
    df.head(50000).to_csv("clean.csv")


def main():

    cols = ["tweet_id", "text", "likes"]
    df = pd.read_csv("./tweets.csv", usecols=cols)

    df = preprocess(df)
    hparams = {
        "random_state": 25056,
        "num_topics": 14,
        "iters": 5,
        "alpha": 0.1,
        "beta": 0.4,
        "remove_n": 180,
    }

    run_with_params(df, hparams)

    # best_params = random_search(df, 20)
    # print("best run params: ", best_params)


if __name__ == "__main__":
    main()
