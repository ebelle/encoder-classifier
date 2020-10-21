# encoder-classifier
Train a neural machine translation model, extract the encoder and use the encoder to get word representations for classification tasks such as part-of-speech tagging, named entity recognition, etc.

# Requirements:

-Pytorch

If using translation_data_prep:
 - Spacy Chinese, English, French and Spanish: https://spacy.io/usage/models
 - Spacy Russian tokenizer: https://github.com/aatimofeev/spacy_russian_tokenizer


# Example usage of cleaning data for NMT task:

python data_prep.py  \
--task translation \
--save-path ../data/en_ru \
--src-file data/UNv1.0.6way.en \
--trg-file data/UNv1.0.6way.zh \
--max-len 55 \
--min-len 5 \
--src-tok en \
--trg-tok zh 

# Example usage of building vocabulary using source word embeddings:

python build_vocab.py \
--data-path /data/en_ru \
--task translation \
--source-name src \
--target-name trg \
--max-vocab-size 60000 \
--source-vectors embeddings/crawl-300d-2M.vec

# Example usage of training NMT:

python train_nmt.py \
--data-path /data/en_ru \
--save-path checkpoints/en_ru \
--num-workers 4 \
--num-layers 2 \
--dropout 0.25 \
--checkpoint 10000 \
--epochs 10 \
--bidirectional 

# Example usage of generating translations:
python generate_nmt.py \
--data-path data/translation/en_ru \
--save-file preds/en_ru_greedy.tsv \
--pretrained-model models/en_ru/model_epoch_5.pt \
--decode-method greedy \
--num-workers 4 \
--batch-size 512 

# Example usage of restarting training:
When restarting, if you froze embeddings the first time, you need to use the freeze flag again.

python train_nmt.py --data-path /data/translation/en_ru \
--save-path checkpoints/en_ru \
--num-workers 4 \
--continue-training-model checkpoints/en_ru/checkpoint_2_40000 \
--freeze-embeddings

# Example usage of training tagger:
python train_tagger.py \
--data-path data/classification_data \
--nmt-data-path data/translation/en_ru/ \
--pretrained-model checkpoints/en_ru/model_epoch_5.pt \
--save-path models/tagger 
--num-workers 4 \
--dropout 0.3  \
--epochs 30 

# Example usage of generating tags:
python generate_tagger.py \
--data-path data/classification_data \
--pretrained-model checkpoints/pos_en_ru/best_model.pt \
--nmt-data-path data/translation/en_ru \
--batch-size 512 \
--save-path preds/pos_en_ru \
