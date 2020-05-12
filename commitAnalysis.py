import pandas as pd
import nltk
import math
import re
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, _document_frequency
from lempel_ziv_complexity import lempel_ziv_complexity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')


def get_sentiment_scores(msgs):
    sid = SentimentIntensityAnalyzer()
    sentiments = []
    for msg in msgs:
        ss = sid.polarity_scores(msg)
        sentiments.append(ss)
    return sentiments


def get_tf_idf_scores(msgs):
    count_vec = CountVectorizer(binary=False)
    count_df = count_vec.fit_transform(msgs)
    transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
    x1 = transformer.fit_transform(count_df)
    posts_cnt = len(msgs)
    vals = [math.log(x) * math.log(posts_cnt / float(y)) for x, y in
            zip(count_df.sum(axis=0).tolist()[0], _document_frequency(x1))]
    score_map = {k: vals[v] for k, v in count_vec.vocabulary_.items()}

    pattern = re.compile('[\W_]+', re.UNICODE)

    scores = []
    for msg in msgs:
        score = 0
        msg = pattern.sub(' ', msg).lower()
        for word in msg.split():
            if word in score_map:
                score += score_map[word]
        scores.append({"tf_idf_score": score})

    return scores


def get_lempel_ziv_complexities(msgs):
    complexities = []
    for msg in msgs:
        complexities.append({"lempel_ziv_complexity": lempel_ziv_complexity(msg)})
    return complexities


if __name__ == "__main__":
    commits = pd.read_csv("./english/TeamNewPipe-NewPipe-liberty-commits-english.csv")

    sentiments_frame = pd.DataFrame(get_sentiment_scores(commits["msg"]))
    commits = commits.join(sentiments_frame)

    score_frame = pd.DataFrame(get_tf_idf_scores(commits["msg"]))
    commits = commits.join(score_frame)

    lz_frame = pd.DataFrame(get_lempel_ziv_complexities(commits["msg"]))
    commits = commits.join(lz_frame)

    print(commits.columns)
    commits.to_csv("./analysis/TeamNewPipe-NewPipe-commit-analysis.csv")
