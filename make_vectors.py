from transformers import BertModel, BertTokenizer

from get_vectors_nocontext import nocontext_vector
from get_vectors_glove import glove_vectors


def make_vectors(list_of_words, method, special_tokens=None):
    """
    Function that returns vectors for a list of words in a dictionary
    where the key is the word and the value is a list of embeddings for
    that word, except for the GloVe method which has a value of a single
    embedding. For the other methods each element in the list represents
    the embedding of the word for a single layer in BERT.
    """

    # String parameters to lowercase
    method = method.lower()
    if special_tokens:
        special_tokens = special_tokens.lower()

    # Check parameters
    allowed_methods = [
        "bertlarge_nocontext",
        "bertlarge_contextual",
        "bertbase_nocontext",
        "bertbase_contextual",
        "glove",
    ]

    allowed_special_tokens = ["all", "withcls", "nospec"]

    if method not in allowed_methods:
        raise ValueError(f"Method should be one of {allowed_methods}")

    if (
        method == "bertlarge_nocontext" or method == "bertbase_nocontext"
    ) and special_tokens not in allowed_special_tokens:
        raise ValueError(f"Pooling should be one of {allowed_special_tokens}")

    # Make the respective vectors
    if method in ["bertlarge_nocontext", "bertbase_nocontext"]:

        if method == "bertlarge_nocontext":
            bert_model = "bert-large-uncased"
        elif method == "bertbase_nocontext":
            bert_model = "bert-base-uncased"
        else:
            raise ValueError(" ")

        # Get model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        model = BertModel.from_pretrained(bert_model)

        if special_tokens == "all":
            words_to_vectors = nocontext_vector(
                list_of_words,
                tokenizer,
                model,
                special_tokens="all",
                subword_pooling="mean",
            )

        if special_tokens == "withcls":
            words_to_vectors = nocontext_vector(
                list_of_words,
                tokenizer,
                model,
                special_tokens="withcls",
                subword_pooling="mean",
            )

        if special_tokens == "nospec":
            words_to_vectors = nocontext_vector(
                list_of_words,
                tokenizer,
                model,
                special_tokens="nospec",
                subword_pooling="mean",
            )

    if method == "bertlarge_contextual":
        # Get model and tokenizer
        bert_model = "bert-large-uncased"
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        model = BertModel.from_pretrained(bert_model)

        raise NotImplementedError()

    if method == "bertbase_contextual":
        # Get model and tokenizer
        bert_model = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        model = BertModel.from_pretrained(bert_model)

        raise NotImplementedError()

    if method == "glove":

        words_to_vectors = glove_vectors(
            list_of_words
        )  # This might not be complete

        print("TESTING")
        for word, vector in words_to_vectors.items():

            if len(vector) != 300:
                print(word)
                print(len(vector))

    # Return vectors
    return words_to_vectors
