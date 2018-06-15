
#time cat ../snli_1.0/snli_1.0_train.jsonl | ./generate_vocab_from_snli.py  > glove/vocab.tsv


time ./data/embedding/convert_embedding.py \
 --vocab ./data/embedding/cnli_vocab.txt \
 --glove-data ./data/embedding/sgns.merge.word \
 --npy ./data/embedding/cnli_embedding.npy \

#
# --random-projection-dimensionality 100
