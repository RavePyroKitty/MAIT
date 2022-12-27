import json

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize


def text_cleanup(corpus):
    stop_words = nltk.corpus.stopwords.words("english")
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    cleaned_corpus_as_sentences = []

    for text in corpus:
        print('Raw article:', text)
        cleaned_text = sent_tokenize(text=text)

        for sentence in cleaned_text:
            # print('Sentence being cleaned:', sentence)
            cleaned_sentence = sentence.lower().strip()
            cleaned_sentence = cleaned_sentence.split()
            # print('Stripped:', cleaned_sentence)
            cleaned_sentence = [word for word in cleaned_sentence if word not in stop_words]
            cleaned_sentence = " ".join(lemmatizer.lemmatize(word) for word in cleaned_sentence)
            " ".join(cleaned_sentence)
            cleaned_corpus_as_sentences.append([cleaned_sentence])

    return cleaned_corpus_as_sentences


def preprocess_for_bert(cleaned_corpus):
    vocab = ['[CLS]\n', '[SEP]\n', '[PAD]\n', '[MASK]\n']
    print('Preprocess input:', cleaned_corpus)

    for sentence in cleaned_corpus:
        unique = np.array(sentence[0].split())
        unique = np.unique(unique)
        unique = [''.join([str(word), '\n']) for word in unique]
        vocab.append(unique)

    vocab_size = len(vocab)

    open("vocab.txt", "w+").write(json.dumps(vocab))

    return vocab_size


def dynamic_batch_size(data, max_batch_size=32):
    batch_size = max_batch_size
    num_interactions = data.shape[1]

    for i in range(max_batch_size):
        if num_interactions % batch_size != 0:
            batch_size = max_batch_size - i
            pass

        if num_interactions % batch_size == 0:
            break

    return int(batch_size)
