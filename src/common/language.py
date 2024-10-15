from typing import Callable
import pickle
import os
import common.tokenizers as tokenizers

import logging

logger = logging.getLogger(__name__)


class Language:
    """
     A class to represent a language with a specific tokenizer and vocabulary.

    Attributes
    ----------
    name : str
        The name of the language.
    tokenizer_name : str
        The name of the tokenizer function to be used, which should correspond to a function in the tokenizers module.
    add_special_tokens : bool
        A flag indicating whether to add special tokens to the tokenizer.
    tokenizer : Callable
        The tokenizer function assigned to this language object.
    vocabulary : dict
        A dictionary where the key is the token and the value is the count of that token.

    Methods
    -------
    __get_tokenizer() -> Callable
        Assigns a tokenizer function to the language object based on the tokenizer_name attribute.
    scan_corpus(path: str) -> None
        Scans a text corpus and updates the vocabulary with token counts.
    token_to_index(token: str) -> int
        Converts a token to its corresponding index.
    index_to_token(index: int) -> str
        Converts an index back to its corresponding token.
    save(path: str) -> None
        Saves the Language object to a specified file path using pickle.
    """

    def __init__(
        self,
        name: str | None = None,
        tokenizer_name: str = None,
        detokenizer_name: str = None,
        add_special_tokens: bool = True,
    ) -> None:
        logger.info(f"Creating a language object for {name}")
        self.name = name
        self.tokenizer_name = tokenizer_name
        self.detokenizer_name = detokenizer_name
        self.add_special_tokens = add_special_tokens

        # Tokenizer must be a string corresponding to a function
        # in tokenizers.py
        self.tokenizer = self.__get_tokenizer(self.tokenizer_name)
        self.detokenizer = self.__get_tokenizer(self.detokenizer_name)

        # vocabulary will be a dictionary with the token as the key and the
        # count as the value
        self.vocabulary = {}

        return None

    def __get_tokenizer(self, tokenizer_name) -> Callable:
        """
        Assign a tokenizer to the language object
        """
        tokenizer_name = tokenizer_name.lower()

        try:
            # Dynamically get the tokenizer function from the tokenizers module
            tokenizer_function = getattr(tokenizers, tokenizer_name)
            return tokenizer_function()
        except AttributeError:
            raise ValueError(f"No tokenizer found for '{tokenizer_name}'")

    def __repr__(self) -> str:
        return f"Language({self.name})"

    def scan_corpus(
        self,
        path: str,
        max_vocab_size: int | None = None,
    ) -> None:
        """
        Scan a corpus in a given language (text file). This corpus should have
        one 'sentence' per line, already tokenized, in a single language.
        """

        logger.info(f"Scanning corpus in {path}")

        with open(path, "r") as f:
            corpus = f.read()

        # Loop through each line in the corpus
        for line in corpus.split("\n"):
            # Tokenize the line
            tokens = self.tokenizer(line)

            # Update the vocabulary
            self.vocabulary.update(
                {token: self.vocabulary.get(token, 0) + 1 for token in tokens}
            )

        # Order the vocabulary by frequency
        self.vocabulary = dict(
            sorted(self.vocabulary.items(), key=lambda x: x[1], reverse=True)
        )

        # If the max_vocab_size is not None, limit the vocabulary to that size
        if max_vocab_size is not None:
            self.vocabulary = dict(
                list(self.vocabulary.items())[:max_vocab_size]
            )

        # Add special tokens to the front of the vocabulary. <PAD>, <UNK>, <SOS>, <EOS>
        if self.add_special_tokens:
            self.vocabulary = {
                "<PAD>": 0,
                "<UNK>": 0,
                "<SOS>": 0,
                "<EOS>": 0,
                **self.vocabulary,
            }

        logger.info(f"Vocabulary size: {len(self.vocabulary)}")

        return None

    def _single_token_to_index(self, token: str) -> int:
        """
        Convert a single token to an index. If the token is not in the vocabulary,
        return the index of <UNK>
        """
        if token in self.vocabulary:
            return list(self.vocabulary.keys()).index(token)
        else:
            if self.add_special_tokens:
                return list(self.vocabulary.keys()).index("<UNK>")
            else:
                raise ValueError(f"Token {token} not in the vocabulary")

    def token_to_index(self, token: str | list[str]) -> int | list[int]:
        """
        Convert either a single or a list of tokens to indices.
        """
        if isinstance(token, str):
            return self._single_token_to_index(token)
        else:
            return [self._single_token_to_index(t) for t in token]

    def _single_index_to_token(self, index: int) -> str:
        """
        Convert a single index to a token
        """
        return list(self.vocabulary.keys())[index]

    def index_to_token(self, index: int | list[int]) -> str | list[str]:
        """
        Convert either a single or a list of indices to tokens
        """
        if isinstance(index, int):
            return self._single_index_to_token(index)
        else:
            return [self._single_index_to_token(i) for i in index]

    def save(self, path: str) -> None:
        """
        Save the following attributes to a pickle file:
            - self.name
            - self.tokenizer
            - self.detokenizer
            - self.add_special_tokens
            - self.vocabulary
        """

        logger.info(f"Saving to {path}")

        # IF the directory does not exist, create it
        directory = "/".join(path.split("/")[:-1])
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "name": self.name,
                    "tokenizer_name": self.tokenizer_name,
                    "detokenizer_name": self.detokenizer_name,
                    "add_special_tokens": self.add_special_tokens,
                    "vocabulary": self.vocabulary,
                },
                f,
            )

        return None

    def convert_corpus_to_indices(
        self, input_path: str, output_path: str
    ) -> None:
        """
        Convert the corpus to indices.

        Each line in the corpus will be converted to a list of token indices.
        """
        logger.info("Converting corpus to indices")

        # Load the corpus
        with open(input_path, "r") as f:
            corpus = f.read()

        # Tokenize the corpus
        tokens = [self.tokenizer(line) for line in corpus.split("\n")]

        # Convert tokens to indices
        corpus_indices = [self.token_to_index(token) for token in tokens]

        with open(output_path, "w") as f:
            for line in corpus_indices:
                # if line is not a list, convert it to a list
                if not isinstance(line, list):
                    line = [line]

                if self.add_special_tokens:
                    # Add <SOS> and <EOS> tokens
                    line = (
                        [self.token_to_index("<SOS>")]
                        + line
                        + [self.token_to_index("<EOS>")]
                    )

                f.write(" ".join([str(i) for i in line]) + "\n")

        return None


def load_language(path: str) -> Language:
    """
    Create a language object from a pickle file
    """

    logger.info(f"Loading from {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)

    language = Language(
        name=obj["name"],
        tokenizer_name=obj["tokenizer_name"],
        detokenizer_name=obj["detokenizer_name"],
        add_special_tokens=obj["add_special_tokens"],
    )
    language.vocabulary = obj["vocabulary"]

    return language
