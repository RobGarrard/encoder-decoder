pull-data:
	@mkdir -p data/raw/
	@if [ ! -f data/raw/data.zip ]; then wget -P data/raw/ https://download.pytorch.org/tutorial/data.zip; else echo "Data already downloaded."; fi
	@if [ ! -d data/raw/names ]; then unzip -j -d data/raw/names/ data/raw/data.zip; else echo "Data already extracted."; fi
	@mkdir -p data/raw/eng-fra
	@mv data/raw/names/eng-fra.txt data/raw/eng-fra/eng-fra.txt

	@echo "Data pulled successfully. Preprocessing data..."
	@uv run src/common/preprocess.py
	

run-tensorboard-name-classification:
	@uv run tensorboard --logdir=logs/name-classification/

run-tensorboard-name-encoder-decoder:
	@uv run tensorboard --logdir=logs/name-encoder-decoder/

run-tensorboard-english-french-translation:
	@uv run tensorboard --logdir=logs/english-to-french-translation/