
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


email = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_persisting_employees.csv")


# In[5]:


del email['Unnamed: 0']
email.head()


# In[6]:


content = email['content']


# In[7]:


content


# In[8]:


import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string


# In[9]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(cont):
    stop_free = " ".join([i for i in cont.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

content_clean = [clean(cont).split() for cont in content]


# In[10]:


content_clean


# In[11]:


import gensim
from gensim import corpora


# In[12]:


dictionary = corpora.Dictionary(content_clean)


# In[14]:


content_term_matrix = [dictionary.doc2bow(cont) for cont in content_clean]


# In[15]:


Lda = gensim.models.ldamodel.LdaModel


# In[16]:


ldamodel = Lda(content_term_matrix, num_topics=30, id2word = dictionary, passes=1, chunksize = 2000, iterations = 100)


# In[18]:


print(ldamodel.print_topics(num_topics=30, num_words=5))


# In[19]:


ldamodel.save('/Users/deepikamulchandani/Downloads/DataSets2_10012017/lda_full')


# In[21]:


model = ldamodel.load('/Users/deepikamulchandani/Downloads/DataSets2_10012017/lda_full')


# In[19]:


print(model.print_topics(num_topics=20, num_words=10))


# In[20]:


email_re = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_removed_employees.csv")


# In[28]:


email_re.shape[0]


# In[21]:


content = email_re['content']


# In[22]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(cont):
    stop_free = " ".join([i for i in cont.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

content_clean_re = [clean(cont).split() for cont in content]


# In[23]:


dictionary_re = corpora.Dictionary(content_clean_re)


# In[24]:


content_term_matrix_re = [dictionary_re.doc2bow(cont) for cont in content_clean_re]


# In[25]:


Lda = gensim.models.ldamodel.LdaModel


# In[26]:


ldamodel_re = Lda(content_term_matrix_re, num_topics=30, id2word = dictionary_re, passes=1, chunksize = 2000, iterations = 100)


# In[27]:


print(ldamodel_re.print_topics(num_topics=30, num_words=5))


# In[29]:


email_Sample = email.sample(n = 241863, axis = 0)


# In[30]:


email_Sample.shape[0]


# In[31]:


content_Sample = email['content']


# In[32]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(cont):
    stop_free = " ".join([i for i in cont.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

content_Sample_clean = [clean(cont).split() for cont in content_Sample]


# In[33]:


dictionary_Sample = corpora.Dictionary(content_Sample_clean)


# In[ ]:


content_term_matrix_Sample = [dictionary_Sample.doc2bow(cont) for cont in content_Sample_clean]


# In[ ]:


ldamodel_Sample = Lda(content_term_matrix_Sample, num_topics=30, id2word = dictionary_Sample, passes=1, chunksize = 2000, iterations = 100)


# In[2]:


email_domain = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_onlypersonaldomains.csv")


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[4]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=1000,
                                   stop_words='english')


# In[5]:


tfidf = tfidf_vectorizer.fit_transform(email_domain['content'])


# In[6]:


lda = LatentDirichletAllocation(n_components=30, max_iter=100,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)


# In[7]:


lda.fit(tfidf)


# In[8]:


tf_feature_names = tfidf_vectorizer.get_feature_names()


# In[9]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# In[14]:


print_top_words(lda,tf_feature_names,5)

