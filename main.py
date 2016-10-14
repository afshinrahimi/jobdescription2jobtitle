'''
Created on 1Oct.,2016

@author: avaf
'''
# import modules & set up logging
import codecs, gensim, logging, string, re, operator, pdb
from scipy import spatial
from collections import OrderedDict
import numpy as np
from sklearn import preprocessing
import csv

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

def textsimilarity(text_pairs, word2vec_model):
    text_similarity_features = []
    for text_pair in text_pairs:
        text1, text2 = text_pair
        vec1 = text2vec(text1, word2vec_model)
        vec2 = text2vec(text2, word2vec_model)
        similarity = 1 - spatial.distance.cosine(vec1, vec2)
        text_similarity_features.append(similarity)
    features = np.asarray(text_similarity_features).reshape(len(text_similarity_features), 1)
    return features
        

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
    features = np.asarray(features).reshape(len(features), 1)
    return features

def normalize_features(train_features, test_features):
    ''' scale the feature values '''
    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train_features)
    normal_train = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    return train_features, test_features

def import_train_test_data(train_file, test_file):
    html_dir='./resources/html_clean/'
    #['Id', 'AUrl', 'ATitle', 'ASnippet', 'BUrl', 'BTitle', 'BSnippet']
    train_pairs = {}
    test_pairs = {}
    #load train file
    with open(train_file, 'rb') as fin:
        reader = csv.reader(fin, delimiter=',', quotechar='"')
        header = True
        for row in reader:
            if header:
                header = False
                continue
            pair_id = int(row[0])
            html_1 = html_dir + str(pair_id) + '_A.clean'
            html_2 = html_dir + str(pair_id) + '_B.clean'
            htmltext1 = ''
            htmltext2 = ''
            with codecs.open(html_1, 'r', encoding='utf-8') as fin:
                htmltext1 = fin.read()
                htmltext1 = htmltext1.decode(encoding='utf-8')
            with codecs.open(html_2, 'r', encoding='utf-8') as fin:
                htmltext2 = fin.read()
                htmltext2 = htmltext2.decode(encoding='utf-8')
            title1 = row[2].decode(encoding='utf-8')
            title2 = row[5].decode(encoding='utf-8')
            text1 = row[3].decode(encoding='utf-8')
            text2 = row[6].decode(encoding='utf-8')
            train_pairs[pair_id] = (title1 + ' ' + text1 + ' ' + htmltext1 , title2 + ' ' + text2 + ' ' + htmltext2)

    #load test file
    with open(test_file, 'rb') as fin:
        reader = csv.reader(fin, delimiter=',', quotechar='"')
        header = True
        for row in reader:
            if header:
                header = False
                continue
            pair_id = int(row[0])
            title1 = row[2]
            title2 = row[5]
            text1 = row[3]
            text2 = row[6]
            test_pairs[pair_id] = (title1 + ' ' + text1 , title2 + ' ' + text2)
    
            
    return train_pairs, test_pairs

def sanity_check(word2vec_model, job_description):
    #just for sanity check
    text_job_distances = get_job_dict_ordered({1:'i am a computer programmer'}, job_description, word2vec_model)
    print(text_job_distances[1].keys()[0:30])    

def write_features(feature_file, features):
    np.savetxt(feature_file, features)
    
if __name__ == '__main__':
    logging.info('loading train and test search results...')
    train_pairs, test_pairs = import_train_test_data(train_file='./resources/alta16_kbcoref_train_search_results.csv', test_file='./resources/alta16_kbcoref_test_search_results.csv')
    logging.info('loading job descriptions...')
    job_description = load_jobs(fname=occupation_file)
    #add job title to job description
    job_description = {job:job + ' ' + desc for job, desc in job_description.iteritems()}
    logging.info('loading word2vec model (takes a while)...')
    word2vec_model = load_word2vec(fname=word2vec_file)
    train_features_job = get_features(text_pairs=[train_pairs[id] for id in sorted(train_pairs.keys())], jobtitle_jobdesc=job_description, word2vec_model=word2vec_model) 
    test_features_job = get_features(text_pairs=[test_pairs[id] for id in sorted(test_pairs.keys())], jobtitle_jobdesc=job_description, word2vec_model=word2vec_model)
    train_features_txtsim = textsimilarity(text_pairs=[train_pairs[id] for id in sorted(train_pairs.keys())], word2vec_model=word2vec_model)
    test_features_txtsim = textsimilarity(text_pairs=[test_pairs[id] for id in sorted(test_pairs.keys())], word2vec_model=word2vec_model)
    train_features = np.hstack((train_features_job, train_features_txtsim))
    test_features = np.hstack((test_features_job, test_features_txtsim))
    #train_features = preprocessing.scale(train_features)
    #test_features = preprocessing.scale(test_features)
    #train_features, test_features = normalize_features(train_features=train_features, test_features=test_features)
    features = np.vstack((train_features, test_features))
    np.savetxt('./resources/features.txt', features)
    pdb.set_trace()
