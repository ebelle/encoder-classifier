# encoder-classifier
Train a neural machine translation model, extract the encoder and use the encoder to get word representations for classification tasks such as part-of-speech tagging, named entity recognition, etc.

# Requirements:

If using translation_data_prep:
 - Spacy Chinese, English, French and Spanish: https://spacy.io/usage/models
 - Spacy Russian tokenizer: https://github.com/aatimofeev/spacy_russian_tokenizer

# Example usage of cleaning data:

python translation_data_prep.py  --save-path ../data/en_ru \
--src-file ../data/UNv1.0.6way.en --trg-file ../data/UNv1.0.6way.zh \
--max-len 55 --min-len 5 --src-tok en --trg-tok zh

# Example usage of building vocabulary using English source word embeddings:

python build_vocab.py --data-path ../data/en_zh \
--task translation --source-name src --target-name trg \
--max-vocab-size 60000 --source-vectors ../embeddings/crawl-300d-2M.vec

# Example usage of training NMT:

python train_nmt.py --data-path ../data/en_zh \
--save-path ../checkpoints/en_zh --num-workers 4 \
--num-layers 2 --dropout 0.5 --checkpoint 500 --epochs 10 \
--validate False --bidirectional True
