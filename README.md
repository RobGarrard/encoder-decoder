# Encoder-Decoder Playground

This repo is for playing around with encoder-decoders (see Cho et. al. (2014) in `docs/`).

We have a couple of exercises:
1. Classify names according to their country of origin.
2. Translate English text into French.


## Getting Started 

### Environment

I've run all these locally on a machine with a GPU, so ... I don't know how this is gonna go on Sagemaker. I think we should be moving away from Sagemaker and toward sshing into EC2 instances instead, because the virtual environment/disk management on Sagemaker is... not fun. For anybody who wants to try this on Sagemaker, there's a `requirements.txt` file that you'll need to install into whichever conda environment has pytorch. 

Otherwise, if running on a local machine, create the virtual environment.

```
uv sync
```

(if you don't have uv installed, check [here](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1))



### Data

We pull the data for both exercises from [this Pytorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html). Even though this tutorial is only doing the names exercise, a later one by the same author ([here](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)) does the translation exercise. Note that I haven't followed either of these tutorials too closely; for maximum pain I wanted to go in blind.

We use a Makefile recipe to pull and preprocess the data into a usable format. Then we curate it with our first notebook.

Run:

```
make pull-data
```

This will automatically download, extract, and run preprocessing python scripts over the data. You should now have a `data/` directory with the following file structure:

```
data/
├── processed
│   ├── countries.txt
│   ├── english.txt
│   ├── french.txt
│   └── names.txt
└── raw
    ├── data.zip
    ├── eng-fra
    │   └── eng-fra.txt
    └── names
        ├── Arabic.txt
        ├── Chinese.txt
        ├── Czech.txt
        ├── Dutch.txt
        ├── English.txt
        ├── French.txt
        ├── German.txt
        ├── Greek.txt
        ├── Irish.txt
        ├── Italian.txt
        ├── Japanese.txt
        ├── Korean.txt
        ├── Polish.txt
        ├── Portuguese.txt
        ├── Russian.txt
        ├── Scottish.txt
        ├── Spanish.txt
        └── Vietnamese.txt
```

The raw data is what has been unzipped from `data.zip`. After the unzip step, a preprocessing script is run that does the following:

- Each 'names' file is called `{country}.txt`, and contains one name per line. Transform them into 2 files: `names.txt`, which merges the names in all the files into a single file; and `countries.txt`, which has a country name per line corresponding to the line in the `names.txt` file.
- In the `eng-fra.txt` file, each line is a tab separated pair of an English phrase/sentence and its French translation. Split these out into 2 files, `english.txt` and `french.txt`.


The idea behind this preprocessing step is to neatly split up 'source' text and 'target' text (or input/output). 

### Data Curation

Run the curation notebook `src/0_curate_data.ipynb`.

In this notebook we leverage a utility class called `Language`. Before we can push anything through a neural net, we need to:
1. convert raw text into 'tokens' (tokenization). This could be a simple as splitting a string into its component characters; splitting apart words based on spaces; or something more sophisticated that that pulls out punctuation and special parts of words like the possessive `'s`. A tokenizer is a function that converts a string into a list of strings.
2. A 'vocabulary' of tokens must be constructed. Often we might want to limit this vocabulary to a small number of common tokens. We may also want to add special tokens like <SOS> (start of sentence), <EOS> (end of sentence), <PAD> (padding), and <UNK> (unknown token).
3. Tokens must be converted into a numeric object. The easiest way to do this is to convert each token to its integer index in the vocabulary. Once tokens are numeric, they can become torch tensors.

Once we have run inference and produced an output sequence in the target language, this output will itself be numeric. A good approach is to have the output shape be the size of the target vocabulary, apply some variant of softmax that maps outputs to the 'probability' that it's any given token, then take the argmax to get a predicted token index. From there we need to:

4. Convert the token indices in the target language to tokens (strings).
5. 'detokenize' the tokens into a single string.

The language utility class has this functionality. I've built a custom one here, but the spaCy python library seems to have all of this functionality too, I just don't know how to use it.

