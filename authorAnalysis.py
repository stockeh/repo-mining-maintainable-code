import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import numpy as np
from makePlots import to_priority


if __name__ == "__main__":
    df = pd.read_csv("./analysis/apache-skywalking-commit-analysis.csv")
    df = pd.concat([df, pd.read_csv("./analysis/apache-beam-commit-analysis.csv")], ignore_index=True)
    df = pd.concat([df, pd.read_csv("./analysis/apache-cassandra-commit-analysis.csv")], ignore_index=True)
    df = pd.concat([df, pd.read_csv("./analysis/apache-druid-commit-analysis.csv")], ignore_index=True)
    df = pd.concat([df, pd.read_csv("./analysis/apache-flink-commit-analysis.csv")], ignore_index=True)
    df['priority'] = df.apply(to_priority, axis=1)
    print(df.columns)

    grpby = df.groupby(['name'])
    df = grpby.agg({'compound': 'mean', 'lempel_ziv_complexity': 'mean', 'priority': 'mean'})

    df2 = pd.read_csv("./analysis/kiegroup-drools-commit-analysis.csv")
    df2 = pd.concat([df2, pd.read_csv("./analysis/OpenLiberty-open-liberty-commit-analysis.csv")], ignore_index=True)
    df2 = pd.concat([df2, pd.read_csv("./analysis/quarkusio-quarkus-liberty-commit-analysis.csv")], ignore_index=True)
    df2 = pd.concat([df2, pd.read_csv("./analysis/runeline-runelite-liberty-commit-analysis.csv")], ignore_index=True)
    df2 = pd.concat([df2, pd.read_csv("./analysis/TeamNewPipe-NewPipe-commit-analysis.csv")], ignore_index=True)
    df2['priority'] = df2.apply(to_priority, axis=1)

    grpby2 = df2.groupby(['name'])
    df2 = grpby2.agg({'compound': 'mean', 'lempel_ziv_complexity': 'mean', 'priority': 'mean'})

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(df['compound'], df['lempel_ziv_complexity'], df["priority"], color="b", label="Professional")
    ax.scatter(df2['compound'], df2['lempel_ziv_complexity'], df2["priority"], color="orange", label="Amateur")
    ax.set_xlabel("Compound Sentiment Score")
    ax.set_ylabel("Lempel-Ziv Complexity")
    ax.set_zlabel("Average Severity (1=most severe, 5=no smell)")
    ax.set_title("Averaged User Commit Metrics")
    plt.legend()
    plt.savefig("user_s_c.png")
    plt.clf()

    df = pd.concat([df2, df], ignore_index=True)
    clusterer = KMeans(n_clusters=12)
    clusterer.fit(np.array([df['compound'], df['lempel_ziv_complexity']/250, df["priority"]/5]).T)
    out = clusterer.predict(np.array([df['compound'], df['lempel_ziv_complexity']/250, df["priority"]/5]).T)
    df["cluster"] = out

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(12):
        ax.scatter(df[df["cluster"] == i]['compound'], df[df["cluster"] == i]['lempel_ziv_complexity'], df[df["cluster"] == i]['priority'])
    ax.set_xlabel("Compound Sentiment Score")
    ax.set_ylabel("Lempel-Ziv Complexity")
    ax.set_zlabel("Average Severity (1=most severe, 5=no smell)")
    ax.set_title("Averaged User Commit Metrics (clustered)")
    plt.savefig("user_s_c_clustered.png")
