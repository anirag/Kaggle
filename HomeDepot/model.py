import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import Stemmer
from nltk.corpus import stopwords
StopWords = stopwords.words("english")
import re, math
from collections import Counter
WORD = re.compile(r'\w+')
from math import *
stemmer = Stemmer.Stemmer('english')
import aa
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
import scipy
import re
import sys
import ngram
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
import utils

#from nlp_utils import stopwords, english_stemmer, stem_tokens
sys.path.append("../")

import Stemmer
stemmer = Stemmer.Stemmer('english')
from nltk.corpus import stopwords
StopWords = stopwords.words("english")

print("reading data...")
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('product_descriptions.csv')
attributes = pd.read_csv('attributes.csv')
df_brand = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
df_color = attributes[(attributes.name == "color family") & (attributes.name == "color/finish") & (attributes.name == "color") & (attributes.name == "top color family")][["product_uid", "value"]].rename(columns={"value": "color"})
df_appl = attributes[attributes.name == "Application Method"][["product_uid", "value"]].rename(columns={"value": "appl"})

print("concat attributes...")
attributes.dropna(how="all", inplace=True)
df_brand.dropna(how="all", inplace=True)
df_color.dropna(how="all", inplace=True)

attributes["product_uid"] = attributes["product_uid"].astype(int)

attributes["value"] = attributes["value"].astype(str)

def concate_attrs(attrs):
    """
    attrs is all attributes of the same product_uid
    """
    names = attrs["name"]
    values = attrs["value"]
    pairs  = []
    for n, v in zip(names, values):
        pairs.append(' '.join((n, v)))
    return ' '.join(pairs)

product_attrs = attributes.groupby("product_uid").apply(concate_attrs)

product_attrs = product_attrs.reset_index(name="product_attributes")

num_train = df_train.shape[0]

def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


def rm_StopWords(s):
    return " ".join([word for word in s.lower().split() if word not in set(StopWords)])
    
stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
StopWords = StopWords + stop_w

df_appl = df_appl.drop_duplicates(cols='product_uid',take_last=True)
print("merge data frame...")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

df_all = pd.merge(df_all, product_attrs, how="left", on="product_uid")
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all = pd.merge(df_all, df_color, how='left', on='product_uid')
df_all = pd.merge(df_all, df_appl, how='left', on='product_uid')
df_all['product_attributes'] = df_all['product_attributes'].fillna('')
df_all['brand'] = df_all['brand'].fillna('')
df_all['color'] = df_all['color'].fillna('')
df_all['appl'] = df_all['appl'].fillna('')
df_all['product_attributes'] = unicode(df_all['product_attributes'])
df_all['brand'] = unicode(df_all['brand'])
df_all['color'] = unicode(df_all['color'])
df_all['search_term'] = df_all['search_term'].replace(aa.replace_dict)
df_all['product_title'] = df_all['product_title'].replace(aa.replace_dict)
df_all['product_description'] = df_all['product_description'].replace(aa.replace_dict)
df_all['product_attributes'] = df_all['product_attributes'].replace(aa.replace_dict)
df_all['brand'] = df_all['brand'].replace(aa.replace_dict)
df_all['color'] = df_all['color'].replace(aa.replace_dict)
df_all['appl'] = df_all['appl'].replace(aa.replace_dict)

print("Removing Stop Words")
df_all['search_term'] = df_all['search_term'].map(lambda x:rm_StopWords(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:rm_StopWords(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:rm_StopWords(x))
df_all['product_attributes'] = df_all['product_attributes'].map(lambda x:rm_StopWords(x))
df_all['brand'] = df_all['brand'].map(lambda x:rm_StopWords(x))
df_all['color'] = df_all['color'].map(lambda x:rm_StopWords(x))
df_all['appl'] = df_all['appl'].map(lambda x:rm_StopWords(x))


print("stem columns words...")
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
df_all['product_attributes'] = df_all['product_attributes'].map(lambda x:str_stemmer(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stemmer(x))
df_all['color'] = df_all['color'].map(lambda x:str_stemmer(x))
df_all['appl'] = df_all['appl'].map(lambda x:str_stemmer(x))
df_all['all_text'] = df_all['search_term']+" "+df_all['product_title']+" "+df_all['product_description']
df_all['pt_pd'] = df_all['product_title']+" "+df_all['product_description'] 

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in df_all["all_text"]:
    sentences += review_to_sentences(review, tokenizer)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 100    # Word vector dimensionality                      
min_word_count = 5   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, \
           size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
            

#import gensim
#model = gensim.models.word2vec.Word2Vec.load_word2vec_format('/Users/araghavan/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
model.syn0.shape


print "Creating average feature vecs for search_terms"
clean_search_reviews = []
for review in df_all["search_term"]:
    clean_search_reviews.append( query_to_words( review,remove_stopwords=True))

DataVecs_st = getAvgFeatureVecs( clean_search_reviews, model, num_features )

print "Creating average feature vecs for product_title"
clean_pt_reviews = []
for review in df_all["product_title"]:
    clean_pt_reviews.append( query_to_words( review, \
        remove_stopwords=True ))

DataVecs_pt = getAvgFeatureVecs( clean_pt_reviews, model, num_features )

print "Creating average feature vecs for product_descriptions"
clean_pd_reviews = []
for review in df_all["product_description"]:
    clean_pd_reviews.append( query_to_words( review, \
        remove_stopwords=True ))

DataVecs_pd = getAvgFeatureVecs( clean_pd_reviews, model, num_features )

def cosine_similarity(s,v):
    return nltk.cluster.util.cosine_distance(s,v)
  
### Similarity Distances
cos_st_pt = []
ed_st_pt = []
mhd_st_pt = []
md_st_pt = []
for i in range(len(DataVecs_st)):
    cos_st_pt.append(cosine_similarity(DataVecs_st[i], DataVecs_pt[i]))
    ed_st_pt.append(euclidean_distance(DataVecs_st[i], DataVecs_pt[i]))
    mhd_st_pt.append(manhattan_distance(DataVecs_st[i], DataVecs_pt[i]))
    md_st_pt.append(minkowski_distance(DataVecs_st[i], DataVecs_pt[i],3))
print "Done"

cos_st_pd = []
ed_st_pd = []
mhd_st_pd = []
md_st_pd = []
for i in range(len(DataVecs_st)):
    cos_st_pd.append(cosine_similarity(DataVecs_st[i], DataVecs_pd[i]))
    ed_st_pd.append(euclidean_distance(DataVecs_st[i], DataVecs_pd[i]))
    mhd_st_pd.append(manhattan_distance(DataVecs_st[i], DataVecs_pd[i]))
    md_st_pd.append(minkowski_distance(DataVecs_st[i], DataVecs_pd[i],3))
print "Done"


for i in range(len(cos_st_pt)):
    if isnan(cos_st_pt[i]):
        cos_st_pt[i] = 0
        ed_st_pt[i] = 0
        md_st_pt[i] = 0
        mhd_st_pt[i] = 0
        cos_st_pd[i] = 0
        ed_st_pd[i] = 0
        md_st_pd[i] = 0
        mhd_st_pd[i] = 0
for i in range(len(cos_st_pd)):
    if isnan(cos_st_pd[i]):
        cos_st_pd[i] = 0
        ed_st_pd[i] = 0
        md_st_pd[i] = 0
        mhd_st_pd[i] = 0

a = pd.DataFrame(cos_st_pt)
b = pd.DataFrame(ed_st_pt)
c = pd.DataFrame(mhd_st_pt)
d = pd.DataFrame(md_st_pt)
e = pd.DataFrame(cos_st_pd)
f = pd.DataFrame(ed_st_pd)
g = pd.DataFrame(mhd_st_pd)
h = pd.DataFrame(md_st_pd)
dis = pd.concat([a,b,c,d,e,f,g,h],axis=1)
dis.columns = ['cos_st_pt','ed_st_pt','mhd_st_pt','md_st_pt','cos_st_pd','ed_st_pd','mhd_st_pd','md_st_pd']
dis.to_csv("/Users/araghavan/w2v_feats.csv",index=False)


### tfidf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=1, max_df = 0.2)
all_vecs = vectorizer.fit(df_all["all_text"])
vocabulary = all_vecs.vocabulary_
tfidf = TfidfVectorizer(vocabulary=vocabulary)
tfidf.fit(df_all['all_text'])f
search_vecs = tfidf.transform(df_all['search_term'])
common_vecs = tfidf.transform(df_all['pt_pd'])
feature_vecs = scipy.sparse.vstack([search_vecs,common_vecs])
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=100, random_state = 2016)
tsvd = tsvd.fit(feature_vecs)
reduced_feature_vecs = tsvd.transform(feature_vecs)                
common_vecs_ = search_vecs.multiply(common_vecs)
srch_sum = search_vecs.sum(axis=1)
df_all['srch_sum'] = srch_sum
df_all['sums'] = common_vecs_.sum(axis=1)
df_all['counts'] = common_vecs_.getnnz(axis=1)
df_all['mean'] = df_all['sums'] / df_all['counts']
print "done"
a = df_all[['sums','counts','mean','srch_sum']]
a['mean'] = a['mean'].fillna(0)
a.to_csv("data_tfidf.csv",index=False)


def extract_feat(df_all):
    df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)

    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']+"\t"+df_all['product_attributes']+"\t"+df_all['brand']+"\t"+df_all['color']+"\t"+df_all['appl']

    df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    df_all['word_in_attributes'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))
    df_all['word_in_brand'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[4]))
    df_all['word_in_color'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[5]))
    df_all['word_in_appl'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[6]))

    
    df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
    df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
    df_all['ratio_attributes'] = df_all['word_in_attributes']/df_all['len_of_query']
    df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_query']
    df_all['ratio_color'] = df_all['word_in_color']/df_all['len_of_query']
    df_all['ratio_appl'] = df_all['word_in_appl']/df_all['len_of_query']


    df_all['cs_1'] = df_all['product_info'].map(lambda x:cs(x.split('\t')[0],x.split('\t')[1]))
    df_all['cs_2'] = df_all['product_info'].map(lambda x:cs(x.split('\t')[0],x.split('\t')[2]))
    df_all['cs_3'] = df_all['product_info'].map(lambda x:cs(x.split('\t')[1],x.split('\t')[2]))
    print "generate unigram"
    df_all["query_unigram"] = list(df_all.apply(lambda x: x["search_term"].lower().split(), axis=1))
    df_all["title_unigram"] = list(df_all.apply(lambda x: x["product_title"].lower().split(), axis=1))
    df_all["description_unigram"] = list(df_all.apply(lambda x: x["product_description"].lower().split(), axis=1))
 
    print "generate bigram"
    join_str = "_"
    df_all["query_bigram"] = list(df_all.apply(lambda x: ngram.getBigram(x["search_term"].split(), join_str), axis=1))
    df_all["title_bigram"] = list(df_all.apply(lambda x: ngram.getBigram(x["product_title"].split(), join_str), axis=1))
    df_all["description_bigram"] = list(df_all.apply(lambda x: ngram.getBigram(x["product_description"].split(), join_str), axis=1))
    ## trigram
    print "generate trigram"
    join_str = "_"
    df_all["query_trigram"] = list(df_all.apply(lambda x: ngram.getTrigram(x["search_term"].split(), join_str), axis=1))
    df_all["title_trigram"] = list(df_all.apply(lambda x: ngram.getTrigram(x["product_title"].split(), join_str), axis=1))
    df_all["description_trigram"] = list(df_all.apply(lambda x: ngram.getTrigram(x["product_description"].split(), join_str), axis=1))
    
    join_str = "X"
    # query unigram
    df_all["query_unigram_title_unigram"] = list(df_all.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["title_unigram"], join_str), axis=1))
    df_all["query_unigram_title_bigram"] = list(df_all.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["title_bigram"], join_str), axis=1))
    df_all["query_unigram_description_unigram"] = list(df_all.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["description_unigram"], join_str), axis=1))
    df_all["query_unigram_description_bigram"] = list(df_all.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["description_bigram"], join_str), axis=1))
    # query bigram
    df_all["query_bigram_title_unigram"] = list(df_all.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["title_unigram"], join_str), axis=1))
    df_all["query_bigram_title_bigram"] = list(df_all.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["title_bigram"], join_str), axis=1))
    df_all["query_bigram_description_unigram"] = list(df_all.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["description_unigram"], join_str), axis=1))
    df_all["query_bigram_description_bigram"] = list(df_all.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["description_bigram"], join_str), axis=1))
    
    
    
    print "generate word counting features"
    feat_names = ["query", "title","description"]
    grams = ["unigram","bigram", "trigram"]
    count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    for feat_name in feat_names:
        for gram in grams:
            ## word count
                df_all["count_of_%s_%s"%(feat_name,gram)] = list(df_all.apply(lambda x: len(x[feat_name+"_"+gram]), axis=1))
                df_all["count_of_unique_%s_%s"%(feat_name,gram)] = list(df_all.apply(lambda x: len(set(x[feat_name+"_"+gram])), axis=1))
                df_all["ratio_of_unique_%s_%s"%(feat_name,gram)] = map(try_divide, df_all["count_of_unique_%s_%s"%(feat_name,gram)], df_all["count_of_%s_%s"%(feat_name,gram)])
    print "generate intersect word counting features"
    #### unigram
    for gram in grams:
        for obs_name in feat_names:
            for target_name in feat_names:
                 if target_name != obs_name:
                     ## query
                        df_all["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = list(df_all.apply(lambda x: sum([1. for w in x[obs_name+"_"+gram] if w in set(x[target_name+"_"+gram])]), axis=1))
                        df_all["ratio_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = map(try_divide, df_all["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)], df_all["count_of_%s_%s"%(obs_name,gram)])

                        
            ## some other feat
        df_all["title_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df_all["count_of_title_%s_in_query"%gram], df_all["count_of_query_%s"%gram])
        df_all["title_%s_in_query_div_query_%s_in_title"%(gram,gram)] = map(try_divide, df_all["count_of_title_%s_in_query"%gram], df_all["count_of_query_%s_in_title"%gram])
        #df_all["description_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df_all["count_of_description_%s_in_query"%gram], df_all["count_of_query_%s"%gram])
        #df_all["description_%s_in_query_div_query_%s_in_description"%(gram,gram)] = map(try_divide, df_all["count_of_description_%s_in_query"%gram], df_all["count_of_query_%s_in_description"%gram])

    print "generate intersect word position features"
    for gram in grams:
        for target_name in feat_names:
            for obs_name in feat_names:
                 if target_name != obs_name:
                     pos = list(df_all.apply(lambda x: get_position_list(x[target_name+"_"+gram], obs=x[obs_name+"_"+gram]), axis=1))
                     ## stats feat on pos
                     df_all["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(np.min, pos)
                     df_all["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(np.mean, pos)
                     df_all["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(np.median, pos)
                     df_all["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(np.max, pos)
                     df_all["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(np.std, pos)
                     ## stats feat on normalized_pos
                     df_all["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(try_divide, df_all["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)], df_all["count_of_%s_%s" % (obs_name, gram)])
                     df_all["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(try_divide, df_all["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)], df_all["count_of_%s_%s" % (obs_name, gram)])
                     df_all["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(try_divide, df_all["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)], df_all["count_of_%s_%s" % (obs_name, gram)])
                     df_all["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(try_divide, df_all["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)], df_all["count_of_%s_%s" % (obs_name, gram)])
                     df_all["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(try_divide, df_all["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] , df_all["count_of_%s_%s" % (obs_name, gram)])
    
    print "generate jaccard coef and dice dist for n-gram"
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["bigram", "trigram"]
    feat_names = ["query", "title","description"]
    for dist in dists:
        for gram in grams:
            for i in range(len(feat_names)-1):
                for j in range(i+1,len(feat_names)):
                     target_name = feat_names[i]
                     obs_name = feat_names[j]
                     df_all["%s_of_%s_between_%s_%s"%(dist,gram,target_name,obs_name)] = \
                     list(df_all.apply(lambda x: compute_dist(x[target_name+"_"+gram], x[obs_name+"_"+gram], dist), axis=1))

    return df_all

df_all['coocurrence_text'] = df_all['query_unigram_title_unigram']+df_all['query_unigram_title_bigram']+df_all['query_unigram_description_unigram']+df_all['query_unigram_description_bigram']+df_all['query_bigram_title_unigram']+df_all['query_bigram_title_bigram']+df_all['query_bigram_description_unigram']+df_all['query_bigram_description_bigram']


### tfidf of bigrams and trigrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1, max_df = 0.2)
all_vecs = vectorizer.fit(df_all["coocurrence_text"])
vocabulary = all_vecs.vocabulary_
tfidf = TfidfVectorizer(vocabulary=vocabulary)
tfidf.fit(df_all['coocurrence_text'])
vecs_1 = tfidf.transform(df_all['query_unigram_title_unigram'])
vecs_2 = tfidf.transform(df_all['query_unigram_title_bigram'])
vecs_3 = tfidf.transform(df_all['query_unigram_description_unigram'])
vecs_4 = tfidf.transform(df_all['query_unigram_description_bigram'])
vecs_5 = tfidf.transform(df_all['query_bigram_title_unigram'])
vecs_6 = tfidf.transform(df_all['query_bigram_title_bigram'])
vecs_7 = tfidf.transform(df_all['query_bigram_description_unigram'])
vecs_8 = tfidf.transform(df_all['query_bigram_description_bigram'])


df_all['sum_1'] = vecs_1.sum(axis=1)
df_all['counts_1'] = vecs_1.getnnz(axis=1)
df_all['mean_1'] = df_all['sums_1'] / df_all['counts_1']
df_all['sum_2'] = vecs_2.sum(axis=1)
df_all['counts_2'] = vecs_2.getnnz(axis=1)
df_all['mean_2'] = df_all['sums_2'] / df_all['counts_2']
df_all['sum_3'] = vecs_3.sum(axis=1)
df_all['counts_3'] = vecs_3.getnnz(axis=1)
df_all['mean_3'] = df_all['sums_3'] / df_all['counts_3']
df_all['sum_3'] = vecs_4.sum(axis=1)
df_all['counts_3'] = vecs_4.getnnz(axis=1)
df_all['mean_3'] = df_all['sums_4'] / df_all['counts_4']
df_all['sum_4'] = vecs_5.sum(axis=1)
df_all['counts_4'] = vecs_5.getnnz(axis=1)
df_all['mean_4'] = df_all['sums_5'] / df_all['counts_5']
df_all['sum_5'] = vecs_6.sum(axis=1)
df_all['counts_5'] = vecs_6.getnnz(axis=1)
df_all['mean_5'] = df_all['sums_6'] / df_all['counts_6']
df_all['sum_6'] = vecs_7.sum(axis=1)
df_all['counts_6'] = vecs_7.getnnz(axis=1)
df_all['mean_6'] = df_all['sums_7'] / df_all['counts_7']
df_all['sum_7'] = vecs_8.sum(axis=1)
df_all['counts_7'] = vecs_8.getnnz(axis=1)
df_all['mean_7'] = df_all['sums_8'] / df_all['counts_8']
print "done"
coc = df_all[['sum_1','counts_1','mean_1','sum_2','counts_2','mean_2','sum_3','counts_3','mean_3','sum_4','counts_4','mean_4','sum_5','counts_5','mean_5','sum_6','counts_6','mean_6','sum_7','counts_7','mean_7','sum_8','counts_8','mean_8']]
coc['mean_1'] = coc['mean_1'].fillna(0)
coc['mean_2'] = coc['mean_2'].fillna(0)
coc['mean_3'] = coc['mean_3'].fillna(0)
coc['mean_4'] = coc['mean_4'].fillna(0)
coc['mean_5'] = coc['mean_5'].fillna(0)
coc['mean_6'] = coc['mean_6'].fillna(0)
coc['mean_7'] = coc['mean_7'].fillna(0)
coc['mean_8'] = coc['mean_8'].fillna(0)
coc.to_csv("coc_data_tfidf.csv",index=False)


df_all = extract_feat(df_all)
df_all = pd.concat([df_all,dis],axis=1)
df_all = pd.concat([df_all,a],axis=1)
df_all = df_all.drop(['all_text','pt_pd','search_term','product_title','product_description','product_info', 'product_attributes','brand','color','query_unigram','title_unigram','description_unigram','query_bigram','title_bigram','description_bigram','query_trigram','title_trigram','description_trigram'],axis=1)

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id'].astype(int)
y = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

###################################### random forest model ###############################################
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
rf = RandomForestRegressor(n_estimators=15, max_depth=10, random_state=0,max_features = 100)
clf = BaggingRegressor(rf, n_estimators=45,max_samples=0.1, random_state=25)
print("fit...")
from sklearn import datasets,cross_validation, grid_search
kf_total = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=4)

from sklearn.metrics import mean_squared_error, make_scorer
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

err = []
for train_index, test_index in kf_total:
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    y_train, y_test = y[train_index], y[test_index] 
    x_train, x_test = X_train[train_index], X_train[test_index]
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    err.append(fmean_squared_error(y_test,y_pred))
print sum(err)/len(err) + 0.003

print("predict...")
y_pred = clf.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0
print("output result...")
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('rf_submission_cv.csv',index=False)

###################################### xgboost model ###############################################
import pickle
svd_1 = pickle.load(open("a.p","rb"))
svd_2 = pickle.load(open("b.p","rb"))
svd_pd1 = pd.DataFrame(svd_1)
svd_pd2 = pd.DataFrame(svd_2)
qid = pickle.load(open("qid.p","rb"))
qid_dense = qid.todense()
qid_pd = pd.DataFrame(qid_dense)
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
a = pca.fit(qid_pd)
#svd_ = pickle.load(open("c.p","rb"))
extra = 100
svd_pd2.columns = [col+extra for col in svd_pd2.columns]

df_all = pd.concat([df_all,svd_pd1],axis=1)
df_all = pd.concat([df_all,svd_pd2],axis=1)
import numpy as np
import xgboost as xgb

        
id_test = df_test['id'].astype(int)
y = df_train['relevance'].values

X_train = df_train.drop(['id','relevance'],axis=1)
X_test = df_test.drop(['id','relevance'],axis=1)

cols_1 = X_train.columns[0:277]
cols_2 = X_train.columns[297:302]
cols_3 = X_train.columns[277:297]
cols_4 = X_train.columns[302:528]
cols_a = cols_1 + cols_2
cols_b = cols_3 + cols_4
X_train_a = X_train[cols_a]
X_train_b = X_train[cols_b]
X_test_a = X_test[cols_a]
X_test_b = X_test[cols_b]
dtrain = xgb.DMatrix(X_train,y)
dtest= xgb.DMatrix(X_test)
dtrain_b = xgb.DMatrix(X_train_b,y)
dtest_b = xgb.DMatrix(X_test_b)

param_a = {'subsample':0.55, 'eta':0.05, 'seed': 10, 'max_depth': 4, 'gamma': 0.75,'objective':'reg:linear','colsample_bytree':0.7,'eval_metric': 'rmse','nthread': 8, 'min_child_weight': 4.0 ,'early_stopping_rounds':10,'verbose_eval':10,'booster':'gbtree'}
num_round_a = 3000
n = int(0.3*74067)
df_train_train = df_train.iloc[:n]
df_train_test = df_train.iloc[n:]
id_test = df_train_test['id'].astype(int)

clf_a = xgb.cv(param_a, dtrain, num_round_a,nfold = 5,metrics={'rmse'}, seed = 0)
clf_a[clf_a['test-rmse-mean']==min(clf_a['test-rmse-mean'])].index.tolist()

param_b = {'subsample':0.55, 'eta':0.05, 'seed': 10, 'max_depth': 4, 'gamma': 0.75,'objective':'reg:linear','colsample_bytree':0.7,'eval_metric': 'rmse','nthread': 8, 'min_child_weight': 4.0 ,'early_stopping_rounds':10,'verbose_eval':10,'booster':'gblinear'}
num_round_b = 8000
clf_b = xgb.cv(param_b, dtrain_b, num_round_b,nfold = 5,metrics={'rmse'}, seed = 0)
clf_b[clf_b['test-rmse-mean']==min(clf_b['test-rmse-mean'])].index.tolist()

########################## xgboost model 1 ##################################
num_round = 8000
bst = xgb.train(param_a,dtrain_a,num_round_a)
print("predict...")
y_pred = bst.predict(dtest_a)
for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0
print("output result...")
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('xgb_submission_cv_a.csv',index=False)

########################## xgboost model 2 ##################################
num_round = 8000
bst = xgb.train(param_b,dtrain_b,num_round_b)
print("predict...")
y_pred = bst.predict(dtest_b)
for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0
print("output result...")
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('xgb_submission_cv_b.csv',index=False)





