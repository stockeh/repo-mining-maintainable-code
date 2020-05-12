import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, OrderedDict


def to_priority(row):
    if len(eval(row["smells"])) > 0:
        return eval(row["smells"])[0]["priority"]
    else:
        return 5


def round_func(x, base=5.0):
    return round(base * round(x/base), 2)


def plot_smell_frequency(smells, out_dir):
    plt.clf()
    frequency = defaultdict(lambda: 0)
    for smell in smells:
        smell = eval(smell)
        if len(smell) == 0:
            frequency[5] += 1
        for s in smell:
            frequency[s["priority"]] += 1

    frequency = OrderedDict(sorted(frequency.items()))
    plt.bar(range(len(frequency)), list(frequency.values()), align='center')
    plt.xticks(range(len(frequency)), list(frequency.keys()))
    plt.xlabel("Code Smell Priority (1 = most severe, 5 = no smell)")
    plt.ylabel("Number of Occurrences")
    plt.title("Code Smell Occurrences by Priority ("+out_dir+")")
    plt.savefig("./"+out_dir+"/code_smell_occurrences.png")


def plot_information_content(info, out_dir):
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(info["Unnamed: 0"], info["tf_idf_score"], label="TF IDF Score")
    ax.scatter(info["Unnamed: 0"], info["lempel_ziv_complexity"], label="Lempel-Ziv Complexity")
    ax.legend()
    plt.xlabel("Commit Index (ordered temporally)")
    plt.ylabel("Information Content")
    plt.title("Information Content of Commit Messages ("+out_dir+")")
    plt.savefig("./"+out_dir+"/information_content.png")


def plot_sentiment_scores(info, out_dir):
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(info["Unnamed: 0"], info["neg"], label="Negativity", alpha=0.5)
    ax.scatter(info["Unnamed: 0"], info["pos"], label="Positivity", alpha=0.5)
    ax.scatter(info["Unnamed: 0"], info["neu"], label="Neutrality", alpha=0.5)
    ax.legend()
    plt.xlabel("Commit Index (ordered temporally)")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score of Commit Messages ("+out_dir+")")
    plt.savefig("./"+out_dir+"/sentiment_scores.png")


def plot_compound_sentiment_scores(info, out_dir):
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(info["Unnamed: 0"], info["compound"])
    plt.xlabel("Commit Index (ordered temporally)")
    plt.ylabel("Compound Sentiment Score of Commit Messages")
    plt.title("Compound Sentiment Score of Commit Messages ("+out_dir+")")
    plt.savefig("./"+out_dir+"/compound_sentiment_scores.png")


def plot_metric_vs_code_smells(info, metric, base, x_label, y_label, title, name, out_dir, severity=False):
    plt.clf()
    data = defaultdict(lambda: [0, 0])
    if severity:
        info = info[info["priority"] != 5]

    for index, row in info.iterrows():
        if severity:
            data[round_func(row[metric], base=base)][0] += row["priority"]
        else:
            if row["priority"] != 5:
                data[round_func(row[metric], base=base)][0] += 1
        data[round_func(row[metric], base=base)][1] += 1

    for k, v in data.items():
        data[k] = v[0] / v[1]

    data = OrderedDict(sorted(data.items()))
    plt.bar(range(len(data)), list(data.values()), align='center')
    plt.xticks(range(len(data)), list(data.keys()))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title+"("+out_dir+")")
    plt.savefig("./"+out_dir+"/"+name)


def run_suite(df, out_dir):
    plot_smell_frequency(df["smells"], out_dir)
    plot_information_content(df, out_dir)
    plot_sentiment_scores(df, out_dir)
    plot_compound_sentiment_scores(df, out_dir)

    plot_metric_vs_code_smells(df, "compound", .2, "Compound Sentiment Score", "Average Code Smell Severity",
                               "Sentiment Score vs. Average Severity", "compound_severity.png", out_dir, severity=True)
    plot_metric_vs_code_smells(df, "compound", .2, "Compound Sentiment Score", "Probability of Code Smell",
                               "Sentiment Score vs. Code Smell Probability", "compound_probability.png", out_dir, severity=False)

    plot_metric_vs_code_smells(df, "tf_idf_score", 80, "TF IDF Score", "Average Code Smell Severity",
                               "TF IDF Score vs. Average Severity", "tf_idf_severity.png", out_dir, severity=True)
    plot_metric_vs_code_smells(df, "tf_idf_score", 80, "TF IDF Score", "Probability of Code Smell",
                               "TF IDF Score vs. Code Smell Probability", "tf_idf_probability.png", out_dir, severity=False)

    plot_metric_vs_code_smells(df, "lempel_ziv_complexity", 30, "Lempel-Ziv Complexity", "Average Code Smell Severity",
                               "Lempel-Ziv Complexity vs. Average Severity", "lempel_ziv_severity.png", out_dir, severity=True)
    plot_metric_vs_code_smells(df, "lempel_ziv_complexity", 30, "Lempel-Ziv Complexity Score",
                               "Probability of Code Smell",
                               "Lempel-Ziv Complexity vs. Code Smell Probability", "lempel_ziv_probability.png",
                               out_dir, severity=False)


if __name__ == "__main__":
    df = pd.read_csv("./analysis/apache-skywalking-commit-analysis.csv")
    df = pd.concat([df, pd.read_csv("./analysis/apache-beam-commit-analysis.csv")], ignore_index=True)
    df = pd.concat([df, pd.read_csv("./analysis/apache-cassandra-commit-analysis.csv")], ignore_index=True)
    df = pd.concat([df, pd.read_csv("./analysis/apache-druid-commit-analysis.csv")], ignore_index=True)
    df = pd.concat([df, pd.read_csv("./analysis/apache-flink-commit-analysis.csv")], ignore_index=True)
    df['priority'] = df.apply(to_priority, axis=1)
    print(df["name"])
    #run_suite(df, "professional")

    df2 = pd.read_csv("./analysis/kiegroup-drools-commit-analysis.csv")
    df2 = pd.concat([df2, pd.read_csv("./analysis/OpenLiberty-open-liberty-commit-analysis.csv")], ignore_index=True)
    df2 = pd.concat([df2, pd.read_csv("./analysis/quarkusio-quarkus-liberty-commit-analysis.csv")], ignore_index=True)
    df2 = pd.concat([df2, pd.read_csv("./analysis/runeline-runelite-liberty-commit-analysis.csv")], ignore_index=True)
    df2 = pd.concat([df2, pd.read_csv("./analysis/TeamNewPipe-NewPipe-commit-analysis.csv")], ignore_index=True)
    df2['priority'] = df2.apply(to_priority, axis=1)
    #run_suite(df2, "amateur")

    df = pd.concat([df2, df], ignore_index=True)
    #run_suite(df, "combined")

    df["normalized_sum"] = df["compound"] + df["lempel_ziv_complexity"] / 250

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        df.nlargest(500, columns=['normalized_sum'], keep='first').to_csv("largest_combined.csv")
        df.nsmallest(500, columns=['normalized_sum'], keep='first').to_csv("smallest_combined.csv")
