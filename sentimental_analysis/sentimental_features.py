import openai
from transformers import BertTokenizer, TFBertModel

from sentimental_analysis.sautils import text_cleanup, preprocess_for_bert
from sentimental_analysis.sentimental_classes import SentimentalClasses


class SentimentalFeatures(SentimentalClasses):
    def __init__(self, vocab_path=r'C:\Users\nicco_ev5q3ww\OneDrive\Desktop\Market Analysis Tools\data_handling\vocab.txt', **kwargs):
        super().__init__(**kwargs)

        self.vocab_path = vocab_path
        self.raw_articles_list = self.get_news_articles()
        self.cleaned_corpus = text_cleanup(self.raw_articles_list)

        self.vocab_size = preprocess_for_bert(self.cleaned_corpus)  # Update text file with vocab

    def textual_features(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained("bert-base-uncased")

        cleaned_articles = []
        embedded_sentences = []

        print('Num articles:', len(cleaned_articles))
        print('Num sentences:', len(self.cleaned_corpus))

        for sentence in self.cleaned_corpus:
            encoded_input = tokenizer(sentence, return_tensors='tf')
            output = model(encoded_input)
            embedded_sentences.append(output)

        embedded_sentences.append(embedded_sentences)

        return embedded_sentences

    def return_openai_embeddings(self, input_text, model):
        result = openai.Embedding.create(
            model=model,
            input=input_text
        )

        return result["data"][0]["dmbedding"]
