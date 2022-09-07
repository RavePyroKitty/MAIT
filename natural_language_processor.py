from tensorflow.keras.preprocessing.text import Tokenizer


def get_data():
    # Should collect up to date articles from all desired news sources and media outlets.

    return None


def preprocess_data():
    # Should tokenize words, and whatever else needs to happen

    return None


sentences = [
    "this is one word",
    "this is another word",
    "I'm curious about punctuation."
]

tokenizer = Tokenizer(num_words=50, oov_token='<OOV>')  # OOV means 'out of value' token, for words that are in the
# training set that are not encoded in the tokenizer,

tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index)

sentences_tokenized = tokenizer.texts_to_sequences(texts=sentences)

print(sentences_tokenized)
