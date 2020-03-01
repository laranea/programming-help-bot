import os

os.system(
    'wget https://github.com/hse-aml/natural-language-processing/releases/download/project/dialogues.tsv -O data/dialogues.tsv'
    )
os.system(
    'wget https://github.com/hse-aml/natural-language-processing/releases/download/project/tagged_posts.tsv -O data/tagged_posts.tsv'
    )
os.system(
    'wget https://github.com/hse-aml/natural-language-processing/releases/download/week3/train.tsv -O starspace_embeddings/data/train.tsv'
    )

# unzip trained starspace embedding on my pc
os.system(
    'unzip starspace_embeddings/data/stackoverflow_duplicate.tsv.zip'
)