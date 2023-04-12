import torch
from tqdm import tqdm


def nocontext_vector(
    list_of_words,
    tokenizer,
    model,
    subword_pooling="mean",
    special_tokens="nospec",
):
    """
    Gets the representations of (sub-)word w in sentence s, by
    """

    words_to_vectors = {}
    for w in tqdm(list_of_words):

        # Tokenize sentence
        inputs = tokenizer(w, return_tensors="pt")

        if special_tokens == "nospec":
            word_indices = [1, -1]

        elif special_tokens == "all":
            word_indices = [0, None]

        elif special_tokens == "withcls":
            word_indices = [0, -1]

        else:
            raise NotImplementedError(
                f"Special tokens {special_tokens} is not implemented."
            )

        # Forward
        outputs = model(**inputs, output_hidden_states=True)

        # Get hidden vectors of the (sub-) words, also skip first hidden. This
        # layer is the output of the embedding layer
        hidden_vectors = [
            layer[:, word_indices[0] : word_indices[-1], :]
            for layer in outputs.hidden_states[1:]
        ]

        # Combine vectors of subwords that belong to the same word
        if subword_pooling == "mean":
            pooled_vectors = []
            for layer in hidden_vectors:
                pooled_vectors.append(torch.mean(layer, 1))

        else:
            raise NotImplementedError(
                f"Subword_pooling {subword_pooling} is not implemented."
            )

        # Return tensor with all hidden vectors for word w
        words_to_vectors[w] = (
            torch.stack(pooled_vectors).detach().numpy()
        )  # TODO: Check if we can do this

    return words_to_vectors
