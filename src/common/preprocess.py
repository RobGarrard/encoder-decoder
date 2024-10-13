################################################################################
#
#                   Preprocess Name and English-to-French Data
#
################################################################################

# Our Language utility class scans text files, tokenizes the text, and builds a
# vocabulary of unique tokens. When we pull the raw data it is in the structure:

# raw
# ├── data.zip
# ├── eng-fra
# │   └── eng-fra.txt
# └── names
#     ├── Arabic.txt
#     ├── Chinese.txt
#     ├── Czech.txt
#     ├── Dutch.txt
#     ├── English.txt
#     ├── French.txt
#     ...

# eng-fra.txt has a sentence on each line, where the first part is in english
# and the second part is in french (tab delimited). We need to split this into
# two files: english.txt and french.txt

# All of the name text files need to be merged and split into two files:
# names.txt and countries.txt

################################################################################
# Libraries

import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################################################################
# Functions


def process_eng_fra():
    logger.info("Processing the English-to-French data")

    # If data/processed doesn't exist, create it
    if not os.path.exists("data/processed"):
        logger.info("Creating directory data/processed/")
        os.makedirs("data/processed")

    with open("data/raw/eng-fra/eng-fra.txt", "r", encoding="utf-8") as infile:
        english_sentences = []
        french_sentences = []
        for line in infile:
            english, french = line.strip().split("\t")
            english_sentences.append(english)
            french_sentences.append(french)

    with open(
        "data/processed/english.txt", "w", encoding="utf-8"
    ) as english_file:
        for sentence in english_sentences:
            english_file.write(sentence + "\n")

    with open(
        "data/processed/french.txt", "w", encoding="utf-8"
    ) as french_file:
        for sentence in french_sentences:
            french_file.write(sentence + "\n")

    return None


def process_names():
    logger.info("Processing the names data")

    # If data/processed doesn't exist, create it
    if not os.path.exists("data/processed"):
        logger.info("Creating directory data/processed/")
        os.makedirs("data/processed")

    names = []
    countries = []

    for filename in os.listdir("data/raw/names"):
        if filename.endswith(".txt"):
            country = filename[:-4]
            with open(
                f"data/raw/names/{filename}", "r", encoding="utf-8"
            ) as infile:
                for line in infile:
                    names.append(line.strip())
                    countries.append(country)

    with open("data/processed/names.txt", "w", encoding="utf-8") as names_file:
        for name in names:
            names_file.write(name + "\n")

    with open(
        "data/processed/countries.txt", "w", encoding="utf-8"
    ) as countries_file:
        for country in countries:
            countries_file.write(country + "\n")


if __name__ == "__main__":
    process_eng_fra()
    process_names()

    logger.info("Preprocessing complete")
