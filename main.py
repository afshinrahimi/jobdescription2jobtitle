'''
Created on 1Oct.,2016

@author: avaf
'''
# import modules & set up logging
import codecs, gensim, logging, string, re, operator, pdb
from scipy import spatial
from collections import OrderedDict
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

word2vec_model = None
job_description = None
word2vec_file = './resources/GoogleNews-vectors-negative300.bin.gz'
occupation_file = './resources/Occupation Data.txt'
regex = re.compile('[%s]' % re.escape(string.punctuation))

def remove_punctuation(str):
    return regex.sub(' ', str)

def load_word2vec(fname):
    ''' load a pre-trained binary format word2vec into a dictionary
    the model is downloaded from https://docs.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download'''
    word2vec = gensim.models.word2vec.Word2Vec.load_word2vec_format(fname, binary=True)
    return word2vec

def load_jobs(fname):
    ''' read ONET occupational dataset from tab delimited text file downloaded from
    https://www.onetcenter.org/dl_files/database/db_21_0_text/Occupation%20Data.txt'''
    
    jobtitle_jobdescription = {}
    with codecs.open(fname, 'r', encoding='utf-8') as fin:
        for line in fin:
            fields = line.strip().split('\t')
            if len(fields) != 3:
                continue
            job_code = fields[0]
            job_title = remove_punctuation(fields[1].lower())
            _job_description = remove_punctuation(fields[2].lower())
            jobtitle_jobdescription[job_title] = _job_description
    return jobtitle_jobdescription

def idtext2vec(id_text, word2vec_model):
    '''convert a dictionary of id:text to text_id:vector by averaging the word vectors'''
    id_vec = {}
    for id, text in id_text.iteritems():
        vec = text2vec(text, word2vec_model)
        id_vec[id] = vec
    return id_vec

def text2vec(text, word2vec_model):
    '''convert a text to a vector by averaging the word vectors'''
    text = text.lower()
    words = text.split()
    vec = 0
    num_words = 0
    for word in words:
        if word in word2vec_model:
            num_words += 1
            vec += word2vec_model[word]
    if num_words == 0:
        vec = np.asarray([0] * 300)
    else:
        vec = vec / num_words
    return vec

def sort_dic_by_value(dic):
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1))
    return OrderedDict(sorted_x)

def get_job_dict_ordered(id_text1, id_text2, word2vec_model):
    id_vec1 = idtext2vec(id_text1, word2vec_model)
    id_vec2 = idtext2vec(id_text2, word2vec_model)
    id1_id2distances = {}
    for id1, vec1 in id_vec1.iteritems():
        id2_distances = {}
        for id2, vec2 in id_vec2.iteritems():
            distance = spatial.distance.cosine(vec1, vec2)
            id2_distances[id2] = distance
        id1_id2distances[id1] = sort_dic_by_value(id2_distances)
    return id1_id2distances


def get_features(text_pairs, jobtitle_jobdesc, word2vec_model):
    '''given a list of text pairs as [('t11', 't12'), ('t21', 't22')....]
    returns features, a vector where the first element is the job similarity of 't11', 't12'.
    The length of the features vector equals the length of the pairs.'''
    jobtitle_vec = idtext2vec(jobtitle_jobdesc, word2vec_model)
    jobtitles = sorted(set(jobtitle_vec.keys()))
    features = []
    for text_pair in text_pairs:
        text1, text2 = text_pair
        vec1 = text2vec(text1, word2vec_model)
        vec2 = text2vec(text2, word2vec_model)
        vec1distances = []
        vec2distances = []
        for jobtitle in jobtitles:
            vec = jobtitle_vec[jobtitle]
            distance1 = spatial.distance.cosine(vec1, vec)
            distance2 = spatial.distance.cosine(vec2, vec)
            vec1distances.append(distance1)
            vec2distances.append(distance2)
        jobsim = 1 - spatial.distance.cosine(vec1distances, vec2distances)
        features.append(jobsim)
    features = np.asarray(features)
    return features
    
if __name__ == '__main__':
    word2vec_model = load_word2vec(fname=word2vec_file)
    job_description = load_jobs(fname=occupation_file)
    #add job title to job description
    job_description = {job:job + ' ' + desc for job, desc in job_description.iteritems()}
    #just for sanity check
    #text_job_distances = get_job_dict_ordered({1:'i am a computer programmer'}, job_description}, word2vec_model)
    #print(text_job_distances[1].keys()[0:30])
    text_pairs = [('programmer', 'developer'), ('manager', 'ceo'), ('chef', 'pilot'), ('aeroplane', 'pilot')]
    features = get_features(text_pairs=text_pairs, jobtitle_jobdesc=job_description, word2vec_model=word2vec_model) 
    pdb.set_trace()
