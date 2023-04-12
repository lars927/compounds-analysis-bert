import torch
import numpy as np


def calc_tran(compound_vector, left_vector, right_vector, w):

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    sim1 = cos(torch.Tensor(compound_vector), torch.Tensor(left_vector)).item()
    sim2 = cos(
        torch.Tensor(compound_vector), torch.Tensor(right_vector)
    ).item()

    return np.average([sim1, sim2], weights=[w, 1 - w]) * 6 + 1


def experiment2_TRAN(compound_df, word_vectors):
    """
    Performs an experiment on the compounds by calculating
    an TRAN score from the vectors in BERT.
    This function returns a dataframe with the original compound_df
    columns, and a column per vector extraction method and TRAN weight
    that contains a list for each compound word with calculated TRAN
    scores for each layer in the respective BERT model.
    """

    def calc_tran_row(row, vector_dictionary, w, is_glove):

        try:
            compound_vectors = vector_dictionary[row["Compound"]]
            left_vectors = vector_dictionary[row["left"]]
            right_vectors = vector_dictionary[row["right"]]
        except KeyError:
            return np.nan

        if is_glove:

            trans = []
            tran = calc_tran(compound_vectors, left_vectors, right_vectors, w)
            trans.append(tran)

        else:
            trans = []
            for compound_vector, left_vector, right_vector in zip(
                compound_vectors, left_vectors, right_vectors
            ):

                tran = calc_tran(compound_vector, left_vector, right_vector, w)
                trans.append(tran)

        return trans

    for method, vector_dictionary in word_vectors.items():
        # Calculate TRAN's
        for w in np.array(range(11)) * 0.1:
            w = round(w, 2)

            column = compound_df.apply(
                calc_tran_row,
                axis=1,
                vector_dictionary=vector_dictionary,
                w=w,
                is_glove=method == "glove",
            )

            # Add new column to dataframe
            compound_df[f"{method}_w={w}_tran"] = column

    return compound_df
