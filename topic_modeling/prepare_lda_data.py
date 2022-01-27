# packages to store and manipulate data
import pandas as pd
import re
import string
import unicodedata
from pathlib import Path

from tqdm import tqdm

import nltk
import spacy
from gensim import corpora, models
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')

import multiprocessing

my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@#-'

def get_words_corpus(data_path, n_workers=4):
    path = Path(data_path)
    if path.is_dir():
        pool = multiprocessing.Pool(processes = n_workers)
        words_corpus_list = pool.map(create_words_corpus, path.iterdir())
        words_corpus = [word for w_c in words_corpus_list for word in w_c]
        pool.close()
    else:
        words_corpus = create_words_corpus(path)
    return words_corpus

def _reader_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)

def raw_newline_count(fname):
    """
    Counts number of lines in file
    """
    f = open(fname, 'rb')
    f_gen = _reader_generator(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen)

def create_words_corpus(data_path):
    words_corpus = []
    print('Processing {}'.format(data_path.name))
    # num_lines = raw_newline_count(data_path)
    df = pd.read_csv(data_path, header=None)
    df.columns = ['text']
    num_lines = df.shape[0]

    counter = 0
    # with open(data_path,'r') as f:
    with tqdm(total=num_lines) as pbar:
        for line in df['text'].tolist():
            counter += 1
            stripped_line = strip_text(line)
            # remove stop words
            elem = remove_stopwords(stripped_line)
            # lemmatize text
            elem = lemmatize_text(elem)
            words_corpus.append(elem.lower().split())
            # # print update after every 1m lines
            # if counter % 1000000 == 0:
            #     print('Processed {} lines'.format(counter))
            pbar.update(1)
    return words_corpus

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ' ')    
    return text
    
def strip_mentions(text):
    entity_prefixes = ['@']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def strip_hashtags(text):
    entity_prefixes = ['#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)
        
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    text = re.sub('['+my_punctuation + ']+', ' ', text) # strip punctuation
    text = re.sub('\s+', ' ', text) #remove double spacing
    text = re.sub(r'\b(?:amp)\b', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def strip_text(elem):
    elem = strip_links(elem)
    elem = strip_mentions(elem)
    elem = strip_hashtags(elem)
    elem = elem.replace('RT', '')
    elem = remove_special_characters(elem)
    elem = [y for x in elem.split() for y in x.split() if len(y) > 2] # remove short words < 2 characters
    elem = ' '.join(elem)
    return elem

def strip_data(raw_texts):
    stripped_tweet_text = []
    for elem in raw_texts:
        elem = strip_links(elem)
        elem = strip_mentions(elem)
        elem = strip_hashtags(elem)
        elem = elem.replace('RT', '')
        elem = remove_special_characters(elem)
        elem = [y for x in elem.split() for y in x.split() if len(y) > 2] # remove short words < 2 characters
        elem = ' '.join(elem)
        stripped_tweet_text.append(elem)
    return stripped_tweet_text

# Stemming / Lemming
nlp = spacy.load('en_core_web_sm')

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_dict(words_corpus):

    dictionary = corpora.Dictionary(words_corpus)
    
    # filter out words with frequency< 15 in a document and filter out words that appear in more than %30 of the documents
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
     # remove gaps in ids after the filter
    dictionary.compactify()

    corpus_bow = [dictionary.doc2bow(text) for text in words_corpus]
    tfidf = models.TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    return dictionary, corpus_tfidf
