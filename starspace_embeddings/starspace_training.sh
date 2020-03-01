#!/usr/bin/env bash

# reference: https://github.com/facebookresearch/StarSpace/blob/master/examples/wikipedia_article_search.sh
./Starspace/starspace train \
-trainFile "./data/ptrain.tsv" \
-model './data/stackoverflow_duplicate' \
-trainMode 3 \
-adagrad true \
-ngrams 1 \
-epoch 5 \
-dim 100 \
-similarity "cosine" \
-minCount 2 \
-verbose true \
-fileFormat labelDoc \
-negSearchLimit 10 \
-lr 0.05 \
-thread 10