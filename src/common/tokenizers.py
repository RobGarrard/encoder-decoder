################################################################################
#
#                                   Tokenizers
#
################################################################################
# If you want to use a custom tokenizer in your language model, add it here.
# For English and French, we use the spaCy tokenizer.

# Libraries
import os
import spacy

################################################################################
# Custom tokenizers


def split_into_chars():
    """
    Returns a tokenizer function that splits a string into its individual
    characters.

    This function returns a lambda function that takes a string as input and
    returns a list of its characters.

    Returns
    -------
    Callable[str]
        A tokenizer function that splits a string into its individual
        characters.

    Examples
    --------
    >>> tokenizer = split_into_chars()
    >>> tokenizer("hello")
    ['h', 'e', 'l', 'l', 'o']

    >>> tokenizer("world")
    ['w', 'o', 'r', 'l', 'd']
    """
    return lambda s: list(s)


def split_into_words():
    """
    Returns a tokenizer function that splits a string into words.

    This function returns a lambda function that takes a string as input and
    returns a list of words by splitting the string at whitespace.

    Returns
    -------
    Callable[[str], List[str]]
        A tokenizer function that splits a string into words.

    Examples
    --------
    >>> tokenizer = split_into_words()
    >>> tokenizer("hello world")
    ['hello', 'world']

    >>> tokenizer("this is a test")
    ['this', 'is', 'a', 'test']
    """
    return lambda s: s.split()


def keep_whole_sentence():
    """
    Returns a tokenizer function that keeps the whole sentence as a single
    token.

    This function returns a lambda function that takes a string as input and
    returns a list containing the entire string as a single element.

    Returns
    -------
    Callable[[str], List[str]]
        A tokenizer function that keeps the whole sentence as a single token.

    Examples
    --------
    >>> tokenizer = keep_whole_sentence()
    >>> tokenizer("hello world")
    ['hello world']

    >>> tokenizer("this is a test")
    ['this is a test']
    """
    return lambda s: [s]


################################################################################
# SpaCy tokenizers


def spacy_english():
    """
    Returns a tokenizer function that uses the spaCy English tokenizer.

    This function checks if the spaCy English model (`en_core_web_sm`) is
    installed, loads the model, and returns a lambda function that tokenizes
    input text using the spaCy tokenizer.

    Returns
    -------
    Callable[[str], List[str]]
        A tokenizer function that tokenizes input text into words using the
        spaCy English tokenizer.

    Examples
    --------
    >>> tokenizer = english_tokenizer()
    >>> tokenizer("Hello, world!")
    ['Hello', ',', 'world', '!']

    >>> tokenizer("This is a test.")
    ['This', 'is', 'a', 'test', '.']
    """
    try:
        en = spacy.load("en_core_web_sm")
    except OSError:
        os.system("uv run spacy download en_core_web_sm")
        en = spacy.load("en_core_web_sm")
    return lambda x: [token.text for token in en.tokenizer(x)]


def spacy_french():
    """
    Returns a tokenizer function that uses the spaCy French tokenizer.

    This function checks if the spaCy French model (`fr_core_news_sm`) is
    installed, loads the model, and returns a lambda function that tokenizes
    input text using the spaCy tokenizer.

    Returns
    -------
    Callable[[str], List[str]]
        A tokenizer function that tokenizes input text into words using the
        spaCy French tokenizer.

    Examples
    --------
    >>> tokenizer = french_tokenizer()
    >>> tokenizer("Bonjour le monde!")
    ['Bonjour', 'le', 'monde', '!']

    >>> tokenizer("Ceci est un test.")
    ['Ceci', 'est', 'un', 'test', '.']
    """
    try:
        fr = spacy.load("fr_core_news_sm")
    except OSError:
        os.system("uv run spacy download fr_core_news_sm")
        fr = spacy.load("fr_core_news_sm")
    return lambda x: [token.text for token in fr.tokenizer(x)]
