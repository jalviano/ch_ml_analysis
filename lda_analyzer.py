# lda_analyzer.py


import argparse
import gensim
import json
import nltk
import numpy as np
import os
import re
import spacy

from gensim.corpora import Dictionary
from gensim.models import AuthorTopicModel, CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords


# nltk.download('stopwords')  # Only need to do this once


class LdaAnalyzer(object):

    def __init__(self,
                 data_path='data/training_20180910',
                 trigrams=False,
                 allowed_pos=None,
                 mallet_path=None):
        self._data_path = data_path
        self._trigrams = trigrams
        self._allowed_pos = allowed_pos if allowed_pos else ('NOUN', 'ADJ', 'VERB', 'ADV')
        self._mallet_path = mallet_path
        self._nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

    def get_topics_atm(self, label=None, num_topics=50, chunk_size=50, passes=10, alpha='symmetric', beta='symmetric',
                       no_below=3, no_above=0.5, verbose=False):
        """
        Builds author-topic model and determines top 10 representative words for each "author" (or in this case, each
        medical condition).
        """
        # Get pe-processed corpus
        corpus, lemmatized, id2word, auth2doc = self._preprocessing(label, no_below=no_below, no_above=no_above)
        # Build author-topic model with specified hyperparameters
        model = AuthorTopicModel(corpus=corpus, num_topics=num_topics, id2word=id2word, author2doc=auth2doc,
                                 chunksize=chunk_size, passes=passes, alpha=alpha, eta=beta, random_state=1)
        # Calculate coherence of model
        coherence_model = CoherenceModel(model=model, texts=lemmatized, dictionary=id2word, coherence='c_v')
        coherence = coherence_model.get_coherence()
        # Get words associated each medical condition
        topics = {topic: [word[0] for word in words] for topic, words in model.show_topics(num_topics, formatted=False)}
        authors = {auth: model.get_author_topics(auth) for auth in model.id2author.values()}
        authors = {auth: topics.get(max(scores, key=lambda x: x[1])[0]) for auth, scores in authors.items()}
        if verbose:
            print('Results:', authors)
            print('C_V coherence:', coherence)
        # Save model and results
        label = '_{}'.format(label) if label else ''
        model.save('data/model{}.atmodel'.format(label))
        with open('data/results{}.json'.format(label), 'w+') as f:
            json.dump(authors, f)
        return authors, coherence

    def _preprocessing(self, label, no_below=3, no_above=0.5):
        """
        Pre-processes records by:
        1. Tokenizing the text
        2. Cleaning punctuation, PHI and whitspace
        3. Removing stop words
        4. Lemmatizing words
        5. Adding bigrams or trigrams
        6. Removing frequent and rare words
        """
        stop_words = set(stopwords.words('english'))
        stop_words.update(['doctor', 'doctors', 'patient', 'patients', 'hospital', 'program', 'stay', 'examine',
                           'examined', 'refill', 'refills', 'refilled', 'impression', 'plan', 'assessment', 'history',
                           'diagnosis', 'diagnoses', 'procedure', 'procedures', 'result', 'results', 'manage',
                           'experiences', 'experiencing', 'finding', 'findings', 'status', 'screening', 'screenings'])
        # Load documents and dictionary of document "authors" (i.e. conditions)
        docs, auth2doc = self._load_texts()
        # Remove PHI placeholders, punctuation and replace whitespace with single space
        docs = [re.sub(r'\[\*\*.+?\*\*\]|[,.\'!?]', '', sent) for sent in docs]
        docs = [re.sub(r'\s+', r' ', sent) for sent in docs]
        # Remove stopwords
        docs = self._sent_to_words(docs)
        docs = self._remove_stopwords(docs, stop_words)
        # Make bigrams and trigrams
        bigrams = gensim.models.Phrases(docs, min_count=5, threshold=100)
        trigrams = gensim.models.Phrases(bigrams[docs], threshold=100)
        bigrams = gensim.models.phrases.Phraser(bigrams)
        docs = self._get_bigrams(docs, bigrams)
        # Use bigrams by default
        if self._trigrams:
            trigrams = gensim.models.phrases.Phraser(trigrams)
            docs = self._get_trigrams(docs, bigrams, trigrams)
        # Lemmatize text
        lemmatized = self._lemmatization(docs)
        id2word = Dictionary(lemmatized)
        # Filter out very uncommon and very common words
        id2word.filter_extremes(no_below=no_below, no_above=no_above)
        _ = id2word[0]
        # Build corpus and save
        corpus = [id2word.doc2bow(doc) for doc in lemmatized]
        np.save('data/lemmatized{}.npy'.format(label), np.array(lemmatized, dtype=object))
        np.save('data/corpus{}.npy'.format(label), np.array(corpus, dtype=object))
        return corpus, lemmatized, id2word, auth2doc

    def _lemmatization(self, texts):
        """
        Lemmatize tokens in records.
        """
        lemmatized = [[token.lemma_ for token in self._nlp(' '.join(sent)) if token.pos_ in self._allowed_pos]
                      for sent in texts]
        return lemmatized

    def _load_texts(self):
        """
        Load records from provided filepath and create "author" dictionary using "chief complaint" specified in each
        record. If no chief complaint is given, use an empty string. This creates a dictionary of records ids mapped to
        each chief complaint.
        """
        texts = []
        auth2doc = {}
        doc_ids = []
        # Iterate over all files in directory
        for filepath in os.listdir(self._data_path):
            if filepath.endswith('.txt'):
                with open(os.path.join(self._data_path, filepath)) as f:
                    # Get text as string
                    texts.append(f.read())
                    f.seek(0)
                    line = ' '
                    conditions = None
                    # Get list of chief complaints
                    while line:
                        line = f.readline()
                        if 'chief complaint' in line.lower():
                            conditions = f.readline().split(',')
                            break
                    conditions = [''] if not conditions else conditions
                    doc_id = int(filepath.split('.')[0])
                    doc_ids.append(doc_id)
                    for c in conditions:
                        c = c.lower().strip()
                        c = re.sub(r'[,;.\'"!?]', '', c)
                        doc_set = auth2doc.setdefault(c, [])
                        doc_set.append(doc_id)
        # Reformat author dictionary with numeric ids
        doc_id_dict = dict(zip(doc_ids, range(len(doc_ids))))
        for a, ids in auth2doc.items():
            for i, doc_id in enumerate(ids):
                auth2doc[a][i] = doc_id_dict[doc_id]
        return texts, auth2doc

    @staticmethod
    def _load_data():
        """
        Load numpy files if available.
        """
        corpus = np.load('data/corpus.npy', allow_pickle=True).tolist()
        lemmatized = np.load('data/lemmatized.npy', allow_pickle=True).tolist()
        return corpus, lemmatized

    @staticmethod
    def _sent_to_words(sentences):
        """
        Tokenize sentences.
        """
        words = [simple_preprocess(sent) for sent in sentences]
        return words

    @staticmethod
    def _remove_stopwords(texts, stop_words):
        """
        Remove stopwords from records.
        """
        texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
        return texts

    @staticmethod
    def _get_bigrams(texts, bigrams):
        """
        Make bigrams for each record.
        """
        bigrams = [bigrams[doc] for doc in texts]
        return bigrams

    @staticmethod
    def _get_trigrams(texts, bigrams, trigrams):
        """
        Make trigrams for each record.
        """
        bigrams = [trigrams[bigrams[doc]] for doc in texts]
        return bigrams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--label', nargs=1, required=False, type=str, default='', help='Output file label')
    parser.add_argument('-k', '--num_topics', nargs=1, required=False, type=int, default=50, help='Number of topics')
    parser.add_argument('-c', '--chunk_size', nargs=1, required=False, type=int, default=50, help='Chunk size')
    parser.add_argument('-p', '--passes', nargs=1, required=False, type=int, default=10, help='Number of passes')
    parser.add_argument('-A', '--alpha', nargs=1, required=False, type=float, default=0.1, help='Alpha parameter')
    parser.add_argument('-B', '--beta', nargs=1, required=False, type=float, default=0.1, help='Beta parameter')
    parser.add_argument('-b', '--no_below', nargs=1, required=False, type=int, default=3,
                        help='Min number of documents containing token')
    parser.add_argument('-a', '--no_above', nargs=1, required=False, type=float, default=0.3,
                        help='Max ratio of documents containing token')
    args = parser.parse_args()
    lda = LdaAnalyzer()
    lda.get_topics_atm(label=args.label[0] if isinstance(args.label, list) else args.label,
                       num_topics=args.num_topics[0] if isinstance(args.num_topics, list) else args.num_topics,
                       chunk_size=args.chunk_size[0] if isinstance(args.chunk_size, list) else args.chunk_size,
                       passes=args.passes[0] if isinstance(args.passes, list) else args.passes,
                       alpha=args.alpha[0] if isinstance(args.alpha, list) else args.alpha,
                       beta=args.beta[0] if isinstance(args.beta, list) else args.beta,
                       no_below=args.no_below[0] if isinstance(args.no_below, list) else args.no_below,
                       no_above=args.no_above[0] if isinstance(args.no_above, list) else args.no_above,
                       verbose=True)


if __name__ == '__main__':
    main()
