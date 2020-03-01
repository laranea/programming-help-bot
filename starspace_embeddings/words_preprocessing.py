import re
import nltk
import os
import sys
sys.path.append('..')
from utils import text_prepare


def prepare_file(in_, out_):
    out = open(out_, 'w')
    for line in open(in_, encoding='utf8'):
        line = line.strip().split('\t')
        new_line = [text_prepare(q) for q in line]
        print(*new_line, sep='\t', file=out)
    out.close()


def setup_starspace():
    if not os.path.exists("/usr/local/bin/starspace"):
        # Building StarSpace
        os.system("wget https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.zip")
        os.system("unzip boost_1_63_0.zip && mv boost_1_63_0 /usr/local/bin")
        os.system("git clone https://github.com/facebookresearch/Starspace.git")
        os.system("cd Starspace && make && cp -Rf starspace /usr/local/bin")

# train.tsv format: similar questions in the same row
prepare_file('./data/train.tsv', './data/ptrain.tsv')
setup_starspace()
