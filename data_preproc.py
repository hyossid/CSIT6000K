import pandas as pd
import numpy as np
import glob
import re
from nltk.stem.snowball import SnowballStemmer

from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
  
warnings.filterwarnings(action = 'ignore')
  
import gensim
from gensim.models import Word2Vec


def proc_df(df):
    df = df[["date", "user", "quoted_text"]]
    
    txt = list()
    for line in df["quoted_text"]:
        txt.append(re.sub(r"[^a-zA-Z0-9#]+", ' ', line))

    df["unstemmed"] = txt
    df = df.drop(columns=['quoted_text'])
    
    df['stemmed'] = df['unstemmed'].apply(lambda x: [stemmer.stem(y) for y in x]) 
    df = df.drop(columns=['unstemmed'])
    
    data = word_2_vec(df)
    return data
    
    
def word_2_vec(df):
    model = Word2Vec(df['stemmed'], min_count=1)
    word_embd = model.wv
    
    df = df.drop(columns=['stemmed'])
    df["wv"] = word_embd
    return df
    

def main():
    f_names = glob.glob("data/tweets/*.csv")
    
    for f_name in f_names:
        df = pd.read_csv(f_name)
        proc_data = proc_df(df)
        proc_data.to_csv('data/preproc_data.csv', mode='a', index=False, header=False)
        

if __name__ == "__main__":
    main()