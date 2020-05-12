import pandas as pd
import langid


if __name__ == "__main__":
    commits = pd.read_csv("./raw/TeamNewPipe-NewPipe-commits-smells.csv")
    filtered_commits = commits[commits["msg"].apply(lambda x: langid.classify(x)[0]) == "en"]
    filtered_commits.to_csv("./english/TeamNewPipe-NewPipe-liberty-commits-english.csv")
