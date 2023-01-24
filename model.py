import pandas as pd
from random import choice, randint
from multiprocessing import Pool, cpu_count
import os.path as path
import time
import numpy as np
from process import preprocess, create_dictionary, apply_dict_to_docs

from gensim.models import CoherenceModel
from gsdmm import MovieGroupProcess

HPARAM_FILE = "hparams.csv"


def get_topics_lists(model, top_clusters, n_words):
    topics = []

    for cluster in top_clusters:
        # create sorted dictionary of word distributions
        sorted_dict = sorted(
            model.cluster_word_distribution[cluster].items(),
            key=lambda k: k[1],
            reverse=True,
        )[:n_words]

        topic = []

        for k, _ in sorted_dict:
            topic.append(k)

        topics.append(topic)

    return topics


def get_topic_top_words(cluster_word_distribution, top_cluster, count):
    res = ""
    for cluster in top_cluster:
        sort_dicts = sorted(
            cluster_word_distribution[cluster].items(),
            key=lambda k: k[1],
            reverse=True,
        )[:count]
        res += f"\nCluster {cluster} : {sort_dicts}"
    return res


def create_gsdmm_model(args):
    start = time.time()
    docs, dictionary, hp = args
    np.random.seed(hp["random_state"])

    vocab_length = len(dictionary)

    gsdmm = MovieGroupProcess(
        K=hp["num_topics"], alpha=hp["alpha"], beta=hp["beta"], n_iters=hp["iters"]
    )

    topic_model = gsdmm.fit(docs, vocab_length)
    print("modelling time: ", time.time() - start)

    doc_count = np.array(gsdmm.cluster_doc_count)
    print("Number of documents per topic :", doc_count)

    top_index = doc_count.argsort()[::-1]

    print(get_topic_top_words(gsdmm.cluster_word_distribution, top_index, 40))

    topics = get_topics_lists(gsdmm, top_index, 20)

    return topic_model, topics, hp


def get_coherence(topics, dictionary, docs):
    coherence = -1
    try:
        coherence = CoherenceModel(
            topics=topics,
            dictionary=dictionary,
            texts=docs,
            coherence="c_v",
        ).get_coherence()
    except:
        print("\ncould not calculate coherence.")

    return coherence


def random_search(docs, dictionary, num_runs):
    topic_count = range(10, 18, 2)
    iter_count = range(5, 25, 5)
    brange = [0.2, 0.3, 0.4, 0.5]
    arange = [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4]

    tasks = []
    for _ in range(num_runs):
        hparams = {
            "random_state": randint(0, 100000),
            "num_topics": choice(topic_count),
            "iters": choice(iter_count),
            "alpha": choice(arange),
            "beta": choice(brange),
        }
        tasks.append((docs, dictionary, hparams))

    NUM_WORKERS = cpu_count()
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(create_gsdmm_model, tasks)

    results_df = pd.DataFrame()
    if path.exists(HPARAM_FILE):
        results_df = pd.read_csv(HPARAM_FILE, index_col=0)

    max_score = 0
    best_params = None

    for res in results:
        _, topics, hp = res
        cs = get_coherence(topics, dictionary, docs)
        hp["coherence"] = cs
        results_df = pd.concat([results_df, pd.DataFrame([hp])], ignore_index=True)

        if cs > max_score:
            max_score = cs
            best_params = hp

    results_df.to_csv(HPARAM_FILE)

    return best_params


def run_with_params(docs, dictionary, hp):
    topic_model, topics, hp = create_gsdmm_model((docs, dictionary, hp))
    cs = get_coherence(topics, dictionary, docs)
    print("coherence: ", cs)
    return topic_model


def main():
    cols = ["tweet_id", "text", "likes"]
    df = pd.read_csv("./tweets.csv")

    df = preprocess(df)

    dictionary = create_dictionary(df)

    df = apply_dict_to_docs(df, dictionary)

    hparams = {
        "random_state": 16448,
        "num_topics": 16,
        "iters": 20,
        "alpha": 0.06,
        "beta": 0.5,
    }

    topic_model = run_with_params(df["tokens"], dictionary, hparams)
    df["topic"] = topic_model
    df.to_csv("clean.csv")

    # best_params = random_search(df["tokens"], dictionary, 40)
    # print("best run params: ", best_params)


if __name__ == "__main__":
    main()
