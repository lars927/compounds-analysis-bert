import torch
import numpy as np


def calc_lmd_reversed(compound_vector, left_vector, right_vector):

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    sim1 = cos(torch.Tensor(compound_vector), torch.Tensor(left_vector)).item()
    sim2 = cos(
        torch.Tensor(compound_vector), torch.Tensor(right_vector)
    ).item()

    return 5 + (sim2 - sim1) * 5


def experiment3_reversedLMD(compound_df, word_vectors):
    """
    Performs an experiment on the compounds by calculating
    an LMD score from the vectors in BERT.
    This function returns a dataframe with the original compound_df
    columns, and a column per vector extraction method that contains
    a list for each compound word with calculated LMD scores for
    each layer in the respective BERT model.
    """

    def calc_lmdreversed_row(row, vector_dictionary, is_glove):

        try:
            compound_vectors = vector_dictionary[row["right"] + row["left"]]
            left_vectors = vector_dictionary[row["left"]]
            right_vectors = vector_dictionary[row["right"]]
        except KeyError:
            return np.nan

        if is_glove:
            lmds = []
            lmd = calc_lmd_reversed(
                compound_vectors, left_vectors, right_vectors
            )
            lmds.append(lmd)

        else:
            lmds = []
            for compound_vector, left_vector, right_vector in zip(
                compound_vectors, left_vectors, right_vectors
            ):

                lmd = calc_lmd_reversed(
                    compound_vector, left_vector, right_vector
                )
                lmds.append(lmd)

        return lmds

    for method, vector_dictionary in word_vectors.items():
        # Calculate LMD's
        column = compound_df.apply(
            calc_lmdreversed_row,
            axis=1,
            vector_dictionary=vector_dictionary,
            is_glove=method == "glove",
        )

        # Add new column to dataframe
        compound_df[f"{method}_reversedlmd"] = column

    return compound_df
