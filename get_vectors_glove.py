import numpy as np

# import pandas as pd


def glove_vectors(list_of_words):

    file_name = "./data/glove.6B.300d.txt"

    words_to_vectors = {}
    with open(file_name, "r") as f:
        for line in f.readlines():
            splitted = line.split(" ")
            word = splitted[0]
            vector = splitted[1:]

            if word in list_of_words:
                words_to_vectors[word] = np.array([float(v) for v in vector])

    return words_to_vectors


# if __name__ == "__main__":

#     # Get the words to create vectors for
#     df_compounds = pd.read_csv("./data/compounds.csv", index_col=0)
#     list_of_words = (
#         df_compounds["Compound"].to_list()
#         + df_compounds["left"].to_list()
#         + df_compounds["right"].to_list()
#     )
#     list_of_words = list(
#         set([w.lower() for w in list_of_words])
#     )  # Only unique lowercase words

#     a = glove_vectors(list_of_words)
#     print(set(list_of_words) - set(a.keys()))
