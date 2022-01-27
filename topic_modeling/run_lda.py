# packages to store and manipulate data
from gensim import models

import pickle
import argparse
import os
import json
from datetime import datetime
from pathlib import Path

from topic_modeling.prepare_lda_data import *

parser = argparse.ArgumentParser(description='LDA Topic Modeling')
parser.add_argument('--data_path', help='path to input data')
# parser.add_argument('output_dir_path', help='path to output dir')
parser.add_argument('--num-topics', type=int, default=50,
                        help='num topics (default=50)')
parser.add_argument('--num-terms', type=int, default=20,
                        help='num terms to print (default=20)')
parser.add_argument('--num-workers', type=int, default=8,
                        help='num topics (default=8)')

args = parser.parse_args()


#parameters
n_topics = args.num_topics # you can experiment with this parameter
n_terms = args.num_terms  # you can experiment with this parameter
n_workers = args.num_workers

data_path = args.data_path
# output_dir = args.output_dir_path

# prep data
# raw_texts = LDA_methods.load_data(data_file)

words_corpus = get_words_corpus(data_path, n_workers=n_workers)

dictionary, corpus_tfidf = get_dict(words_corpus)

## create LDA model 
lda_model_tfidf = models.LdaMulticore(corpus_tfidf, num_topics=n_topics, id2word=dictionary, passes=10, workers=n_workers)

# save LDA model, dictionary and corpus objects
now = datetime.now()
dt_string = now.strftime("%b%d_%H-%M-%S")
save_directory = './saved_models'
output_dir = os.path.join(save_directory, dt_string)
os.makedirs(output_dir, exist_ok=True)
output_model_file = os.path.join(output_dir, 'ldamodel.pkl')
with open(output_model_file, 'wb') as f: 
    pickle.dump((lda_model_tfidf, corpus_tfidf, dictionary), f)

# save config data
output_config_file = os.path.join(output_dir, 'config.json')
config_data = {'num_topics': n_topics}
with open(output_config_file, 'w') as fp:
    json.dump(config_data, fp)

# examine learned topics
topics_list=[]
for topic_ind in range(n_topics):
    topic = lda_model_tfidf.get_topic_terms(topic_ind, n_terms)
    topics_list.append([dictionary[pair[0]] for pair in topic])
    print("Topic", topic_ind,":", topics_list[topic_ind])