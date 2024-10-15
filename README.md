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
2. A 'vocabulary' of tokens must be constructed. Often we might want to limit this vocabulary to a small number of common tokens. We may also want to add special tokens like `<SOS>` (start of sentence), `<EOS>` (end of sentence), `<PAD>` (padding), and `<UNK>` (unknown token).
3. Tokens must be converted into a numeric object. The easiest way to do this is to convert each token to its integer index in the vocabulary. Once tokens are numeric, they can become torch tensors.

Once we have a trained model run inference and produced an output sequence in the target language, this output will itself be numeric. A good approach is to have the output shape be the size of the target vocabulary, apply some variant of softmax that maps outputs to the 'probability' that it's any given token, then take the argmax to get a predicted token index. From there we need to:

4. Convert the token indices in the target language to tokens (strings).
5. 'detokenize' the tokens into a single string.

The language utility class has this functionality. I've built a custom one here, but the spaCy python library seems to have all of this functionality too, I just don't know how to use it.

#### Use your own data

If you have your own data to use, all you need to do is curate it into the same format: a single file with one line per 'sentence' (string to be encoded) (one file each for source and target languages).

If you want to use a custom tokenizer/detokenizer, add it to the `src/common/tokenizers.py` file.

Then in the curation step, instantiate a Language object:

```
mylanguage = Language(
    name='mylanguage',
    tokenizer='{name_of_function_in_tokenizers.py}',
    detokenizer='{name_of_function_in_tokenizers.py}'
)
```

Then apply the `scan_corpus` method to learn a vocabulary of tokens. Then apply the `convert_corpus_to_indices` method to write out a file where each line is an array of integers that can be read a a torch tensor.

### Train models

Now you're good to start training. 

Run `1_rnn_classification.ipynb` to train a simple RNN classifier. This takes the name as a sequence of letters, and treats the country as a one-hot encoded vector of country names. Standard classification. We do 10 epochs with a one-cycle learning rate policy, this is done on a GPU in no time at all.

Run `2_encoder_decoder_names.ipynb`. This is the same exercise as above, but instead of the output being a one-hot encoded vector of countries; we now use an encoder-decoder architecture. The encoder encodes the names letter-by-letter into a context vector; the decoder gets fed the context and a `<SOS>` token, and needs to reconstruct the name of the target country letter-by-letter.

Run `3_english_to_french.ipynb` to make an english-to-french translator. It runs for 10 epochs, which takes about 10 minutes on my GPU. The quality of the translation isn't great, but it's suprisingly okay for 10 minutes of training.

### Run tensorboard to track your experiments

PyTorch Lighning logs metrics to `logs/`. You can turn on the tensorboard to have a look at how these metrics evolve over the training cycle. Run 

```
make run-tensorboard-name-classification
or
make run-tensorboard-name-encoder-decoder
or 
run-tensorboard-english-french-translation
```

This will spin up the tensorboard on localhost:6007. With the name classification notebook, try running it 3 times: once each with basic RNN, LSTM, and GRU unit types. See on your tensorboard which performs better.

Note that I don't know how tensorboard functions on Sagemaker.