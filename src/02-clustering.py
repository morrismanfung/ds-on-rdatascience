# Author: Morris M. F. Chan
# 2023-03-01

'''
This script clusters the saved posts into n clusters.

Usage: 02-clustering.py --file=<file> --n=<n>

Options:
--file=<file>   The file name of the saved posts
--n=<n>         The desired number of cluster [default: 10]
'''

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt 
from docopt import docopt
opt = docopt(__doc__)

import json
import pandas as pd
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
nltk.download("punkt")
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def main(file, n):
    n = int(n)
    with open(file) as f:
        data=json.load(f)
    
    df = pd.DataFrame(data)
    df['created'] = df['created'].apply(datetime.fromtimestamp)

    sia = SentimentIntensityAnalyzer()

    df['title_nltk_neg'] = [d['neg'] for d in df['title'].apply(sia.polarity_scores)]
    df['title_nltk_neu'] = [d['neu'] for d in df['title'].apply(sia.polarity_scores)]
    df['title_nltk_pos'] = [d['pos'] for d in df['title'].apply(sia.polarity_scores)]
    df['title_nltk_compound'] = [d['compound'] for d in df['title'].apply(sia.polarity_scores)]

    df['body_nltk_neg'] = [d['neg'] for d in df['selftext'].apply(sia.polarity_scores)]
    df['body_nltk_neu'] = [d['neu'] for d in df['selftext'].apply(sia.polarity_scores)]
    df['body_nltk_pos'] = [d['pos'] for d in df['selftext'].apply(sia.polarity_scores)]
    df['body_nltk_compound'] = [d['compound'] for d in df['selftext'].apply(sia.polarity_scores)]

    df['title_textblob_polarity'] = df['title'].apply(get_polarity)
    df['title_textblob_subjectivity'] = df['title'].apply(get_subjectivity)

    df['body_textblob_polarity'] = df['selftext'].apply(get_polarity)
    df['body_textblob_subjectivity'] = df['selftext'].apply(get_subjectivity)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb_title = model.encode(df['title'])
    emb_body = model.encode(df['selftext'])

    emb = pd.concat([pd.DataFrame(emb_title, columns=[f'title_{i}' for i in range(emb_title.shape[1])]),
                 pd.DataFrame(emb_body, columns=[f'body_{i}' for i in range(emb_body.shape[1])])],
                 axis = 1)
    
    linkage_array = linkage(emb, method='complete', metric='cosine')
    dendrogram(linkage_array, p=4, truncate_mode='level')
    plt.savefig('test.png')

    labels = fcluster(linkage_array, n, criterion='maxclust')
    df['label'] = labels

    for i in range(1, n+1):
        filtered = df[df['label']==i].drop('comments', axis=1)
        filtered.to_csv(f'df_label_{i}.csv')

    df.to_csv('data/labelled.csv')

def get_polarity(text):
    tb_text = TextBlob(text)
    return tb_text.sentiment.polarity

def get_subjectivity(text):
    tb_text = TextBlob(text)
    return tb_text.sentiment.subjectivity

if __name__ == '__main__':
    main(opt['--file'], opt['--n'])