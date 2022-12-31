import tweepy
import csv
import datetime
import time
import os


def get_writer():
    dir = "./tweets.csv"
    exists = os.path.exists(dir)
    file = open(dir, "a", encoding="utf-8")
    headers = [
        "tweet_id",
        "text",
        "creation_date",
        "likes",
        "retweets",
        "replies",
        "quotes",
        "author_id",
        "username",
        "user_since",
        "followers",
        "following",
        "tweet_count",
        "timestamp",
    ]
    writer = csv.DictWriter(file, fieldnames=headers)
    if not exists:
        writer.writeheader()

    return file, writer


def write_token(tok):
    with open("./next_token.txt", "w") as file:
        file.write(tok)


def get_next_token():
    with open("./next_token.txt", "r") as file:
        tok = file.read()
        return tok


def write_response(writer, resp):
    userdata = resp.includes["users"]
    users = {}
    for i, user in enumerate(userdata):
        users[user.id] = i

    to_write = []
    for tweet in resp.data:
        user_idx = users[tweet.author_id]
        data = {
            "tweet_id": str(tweet.id),
            "text": str(tweet.text),
            "author_id": tweet.author_id,
            "creation_date": tweet.created_at,
            "likes": tweet.public_metrics["like_count"],
            "replies": tweet.public_metrics["reply_count"],
            "retweets": tweet.public_metrics["retweet_count"],
            "quotes": tweet.public_metrics["quote_count"],
            "user_since": userdata[user_idx].data["created_at"],
            "followers": userdata[user_idx].public_metrics["followers_count"],
            "following": userdata[user_idx].public_metrics["following_count"],
            "tweet_count": userdata[user_idx].public_metrics["tweet_count"],
            "username": userdata[user_idx].data["username"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        to_write.append(data)

    writer.writerows(to_write)


def get_tweepy_client():
    conf = {
        "consumer_key": "4mxtKMBfnKbNF6nTA4chwAE9q",
        "consumer_secret": "LS0m3fd2PItVmuThL6CDYsyTCkBdDzNsQd3VapLQr8E29I4Zf2",
        "access_token": "2879449662-eIW1DR0Wf041ae83bhrE7DQA61h1kHVDDcnwNeS",
        "access_token_secret": "Dzv37H0LsrYHalpSVSedLnaWxmj8e4Xh4NVqrOzmJXP4l",
        "bearer_token": "AAAAAAAAAAAAAAAAAAAAAJszjAEAAAAALo7BuAvooX2r%2F9zQlJf5BBos9Sk%3DOStRXqtdcBcWctTMeYwVMXlIFN1xni6XnrdmyooWSD5NTz5O93",
    }
    client = tweepy.Client(
        conf["bearer_token"],
        conf["consumer_key"],
        conf["consumer_secret"],
        conf["access_token"],
        conf["access_token_secret"],
    )

    return client


def main():

    client = get_tweepy_client()
    file, writer = get_writer()

    tweet_count = 0

    try:
        while tweet_count < 25000:
            next_tok = get_next_token()
            resp = client.search_recent_tweets(
                query="lang:fa -is:reply -is:retweet #مهسا_امینی",
                max_results=100,
                expansions="author_id",
                user_fields=["created_at", "username", "public_metrics"],
                tweet_fields=["created_at", "public_metrics", "entities"],
                # sort_order="relevancy",
                next_token=None if len(next_tok) == 0 else next_tok,
            )

            write_response(writer, resp)
            tweet_count += len(resp.data)
            print(tweet_count)

            key = "next_token"
            if key not in resp.meta:
                break
            write_token(resp.meta[key])

            time.sleep(2)

    except Exception as e:
        print(e)

    finally:
        file.close()


if __name__ == "__main__":
    main()
