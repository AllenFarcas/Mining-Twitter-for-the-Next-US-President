import tweepy
import pandas as pd
import json

#Constants
filename = 'uselection_tweets_12Aug_3_5Mil.csv'
outputname = 'output.jsonl'
chunksize = 100 #Current Twitter API limit per http request
CONSUMER_KEY = "CONSUMER_KEY"
CONSUMER_SECRET= "CONSUMER_SECRET"
ACCESS_TOKEN = "ACCESS_TOKEN"
ACCESS_TOKEN_SECRET = "ACCESS_TOKEN_SECRET"
total_chunk = 0
total_ids = 0
total_out = 0

#Functions
def auth():
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, parser=tweepy.parsers.JSONParser())
    # test authentication
    try:
        api.verify_credentials()
        print("Authentication OK")
        return api
    except:
        print("Error during authentication")
        raise

def processChunk(api, chunk):
    global total_out, total_chunk, total_ids
    getEnglish = chunk[chunk["Language"] == "en"] #Is this meaningful?
    ids = getEnglish["Id"].tolist()
    tweets = api.statuses_lookup(id_=ids, tweet_mode="extended")
    # Check size of input and output
    total_chunk += len(chunk)
    total_ids += len(ids)
    total_out += len(tweets)
    with open(outputname, 'a') as writer:
        for tw in tweets:
            write = json.dumps(tw) + '\n'
            writer.write(write)

if __name__ == "__main__":
    api = auth()
    #Reset output file
    open(outputname, 'w').close()
    i = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        processChunk(api, chunk)
        i += 1
        if i % 1000 == 0:
            print("Chunk #" + str(i) + " done")
    print("total input size: " + str(total_chunk))
    print("total filtered (en) input size: " + str(total_ids))
    print("total returned tweets size: " + str(total_out))
