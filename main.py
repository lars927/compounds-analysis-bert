import os
import pickle
import pandas as pd
import argparse as ap

from make_vectors import make_vectors

from experiment1_LMD import experiment1_LMD
from experiment2_TRAN import experiment2_TRAN
from experiment3_reversedLMD import experiment3_reversedLMD
from experiment4_cases import experiment4_cases


def main_vectors(list_of_words, exp_vectors, exp_tokens, path="./data/"):
    """
    Organizes the loading/creating of the word vectors.

    For each available method, this script will load the vectors in
    a dataframe if available on disk, otherwise it will create those
    vectors and save them on disk.
    This function returns a dictionary in the form:
        method -> word -> vector(s)


    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"The directory {path} does not exist.")

    available_methods = [
        "bertlarge_nocontext",
        "bertlarge_contextual",
        "bertbase_nocontext",
        "bertbase_contextual",
        "glove",
    ]

    available_special_tokens = ["all", "withcls", "nospec"]

    # filter
    available_methods = [m for m in available_methods if m in exp_vectors]
    available_special_tokens = [
        t for t in available_special_tokens if t in exp_tokens
    ]

    method_word_vector = {}
    for method in available_methods:
        # No context vectors
        if method.endswith("nocontext"):
            for special_tokens in available_special_tokens:
                # Check if vectors for this method are saved
                if os.path.exists(
                    f"{path}{method}_{special_tokens}.dictionary"
                ):
                    print(f"Found vectors for {method}_{special_tokens}")

                    with open(
                        f"{path}{method}_{special_tokens}.dictionary", "rb"
                    ) as handle:
                        vectors = pickle.load(handle)

                else:
                    print(
                        f"Did not find vectors for {method}_{special_tokens}"
                    )
                    print(f"Creating vectors for {method}_{special_tokens}")

                    # Create vectors
                    vectors = make_vectors(
                        list_of_words, method, special_tokens
                    )

                    # Save the vectors to disk
                    with open(
                        f"{path}{method}_{special_tokens}.dictionary", "wb"
                    ) as handle:
                        pickle.dump(
                            vectors, handle, protocol=pickle.HIGHEST_PROTOCOL
                        )

                # Save word to vector dictionary in a dictionary
                method_word_vector[f"{method}_{special_tokens}"] = vectors

        # All other methods
        else:
            # Check if vectors for this method are saved
            if os.path.exists(f"{path}{method}.dictionary"):
                print(f"Found vectors for {method}")

                with open(f"{path}{method}.dictionary", "rb") as handle:
                    vectors = pickle.load(handle)

            else:
                print(f"Did not find vectors for {method}")
                print(f"Creating vectors for {method}")

                vectors = make_vectors(list_of_words, method)

                with open(f"{path}{method}.dictionary", "wb") as handle:
                    pickle.dump(
                        vectors, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )

            # Save word to vector dictionary in a dictionary
            method_word_vector[method] = vectors

    return method_word_vector


def main_experiments(
    df_compounds, run_experiments, method_word_vector, path="./data/"
):
    experiments = {
        "Experiment1_LMD": experiment1_LMD,
        "Experiment2_TRAN": experiment2_TRAN,
        "Experiment3_reversedLMD": experiment3_reversedLMD,
        "Experiment4_cases": experiment4_cases,
    }

    experiments = {
        k: v for k, v in experiments.items() if k in run_experiments
    }

    for name, experiment_function in experiments.items():
        print(f"Running experiment '{name}'")
        resulting_df = experiment_function(
            df_compounds.copy(), method_word_vector
        )

        print(f"Saving experiment results to {path}{name}.csv")
        resulting_df.to_csv(f"{path}{name}.csv")

    return


def main_plot():
    # See Plots Notebook for plotting code

    raise NotImplementedError()


def main():
    # Get the words to create vectors for
    df_compounds = pd.read_csv("./data/compounds.csv", index_col=0)
    list_of_words = (
        df_compounds["Compound"].to_list()
        + df_compounds["left"].to_list()
        + df_compounds["right"].to_list()
        + (
            df_compounds["right"] + df_compounds["left"]
        ).to_list()  # For the reversed experiment
    )

    list_of_words = list(
        set([w.lower() for w in list_of_words])
    )  # Only unique lowercase words

    # Find and make vectors (TODO: according to command line arguments)
    method_word_vector = main_vectors(list_of_words, path="./data/")

    # Experiment on vectors (TODO: according to command line arguments)
    main_experiments(df_compounds, method_word_vector, path="./data/")

    # Make plots from experiment results
    # main_plot()

    return


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument(
        "--vectors",
        nargs="+",
        default=[
            "bertlarge_nocontext",
            "bertlarge_contextual",
            "bertbase_nocontext",
            "bertbase_contextual",
            "glove",
        ],
        help=(
            "Which word vectors to create. Available options:"
            " bertlarge_nocontext, bertlarge_contextual, bertbase_nocontext,"
            " bertbase_contextual, glove"
        ),
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=["all", "withcls", "nospec"],
        help=(
            "Which special tokens to use when creating word vectors. Available"
            " options: all, withcls, nospec"
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "Experiment1_LMD",
            "Experiment2_TRAN",
            "Experiment3_reversedLMD",
            "Experiment4_cases",
        ],
        help=(
            "Which experiments to run. Available options: Experiment1_LMD,"
            " Experiment2_TRAN, Experiment3_reversedLMD, Experiment4_cases"
        ),
    )

    args = parser.parse_args()

    # Get the words to create vectors for
    df_compounds = pd.read_csv("./data/compounds.csv", index_col=0)
    list_of_words = (
        df_compounds["Compound"].to_list()
        + df_compounds["left"].to_list()
        + df_compounds["right"].to_list()
        + (
            df_compounds["right"] + df_compounds["left"]
        ).to_list()  # For the reversed experiment
    )

    list_of_words = list(
        set([w.lower() for w in list_of_words])
    )  # Only unique lowercase words

    method_word_vector = main_vectors(list_of_words, args.vectors, args.tokens)
    main_experiments(df_compounds, args.experiments, method_word_vector)

    print("Done!")
