import nltk
import numpy as np

from nltk.stem.porter import  PorterStemmer
steamer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem (word):
    return steamer.stem(word.lower())

def bag (token_sentence, all_wo):
    token_sentence = [stem(w) for w in token_sentence]
    bg = np.zeros(len(all_wo), dtype=np.float32)
    for idx, w in enumerate(all_wo):
        if w in token_sentence:
            bg[idx] = 1.0
    return bg
