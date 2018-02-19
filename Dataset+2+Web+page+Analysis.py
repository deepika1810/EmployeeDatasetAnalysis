
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


websites = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/http_info.csv")


# In[ ]:


websites['date']=pd.to_datetime(websites['date'])


# In[3]:


websites.head()


# In[4]:


websites.shape[0]


# In[3]:


Web_Sample = websites.sample(n=2500000,axis=0)


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[5]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=1000,
                                   stop_words='english')


# In[6]:


tfidf = tfidf_vectorizer.fit_transform(Web_Sample['content'])


# In[7]:


lda = LatentDirichletAllocation(n_components=30, max_iter=100,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)


# In[ ]:


lda.fit(tfidf)


# In[11]:


tf_feature_names = tfidf_vectorizer.get_feature_names()


# In[12]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()



# In[13]:


print_top_words(lda,tf_feature_names,10)


# In[3]:


job = websites[websites['content'].str.contains("skills")&websites['content'].str.contains("resume")&websites['content'].str.contains("experience")&websites['content'].str.contains("responsibilities")&websites['content'].str.contains("develop")&websites['content'].str.contains("contribute")&websites['content'].str.contains("required")&websites['content'].str.contains("team")&websites['content'].str.contains("degree")&websites['content'].str.contains("multitask")]


# In[4]:


job['date']=pd.to_datetime(job['date'])


# In[21]:


job['count of job URLs']=1


# In[16]:


job['Date'] = job['date'].dt.date
job['Time'] = job['date'].dt.time


# In[17]:


job


# In[22]:


job3 = job.groupby(['user','Date'])['count of job URLs'].sum()


# In[23]:


job3


# In[26]:


dftry = pd.DataFrame(job3)


# In[6]:


job


# In[27]:


dftry


# In[30]:


dftry.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/AllEmpLookingforJobs_DailyCount.csv")


# In[7]:


job2


# In[8]:


job2[job2['id']>100]


# In[10]:


removed_emp = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/Employees_removed.csv")


# In[11]:


job['Does the employee still exist?'] = 1


# In[12]:


job


# In[22]:


def webpages_of_removed_employees():
    for row in removed_emp.itertuples():
        for row2 in job.itertuples():
            if row[2]==row2[3]:
                job.set_value(row2.Index, 'Does the employee still exist?', 0)
webpages_of_removed_employees()


# In[37]:


removed_emp['Count of Job Search URLs']=0


# In[38]:


removed_emp


# In[40]:


def count_emp_web():
    for row in job.itertuples():
        if row[7]==0:
            for row2 in removed_emp.itertuples():
                if row[3]==row2[2]:
                    removed_emp.set_value(row2.Index,'Count of Job Search URLs',row2[10]+1)                    
count_emp_web()


# In[41]:


removed_emp


# In[71]:


removed_emp.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/Employees_removed_withURLcount.csv",index=False)


# In[53]:


removed_emp[removed_emp['Count of Job Search URLs']>100]


# In[54]:


job2[job2['id']>100]


# In[14]:


all_emp=pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/2009-12.csv")


# In[63]:


all_emp['Count of Job Search URLs']=0


# In[15]:


all_emp.groupby(['role']).count()


# In[65]:


def count_empall_web():
    for row in job.itertuples():
            for row2 in all_emp.itertuples():
                if row[3]==row2[2]:
                    all_emp.set_value(row2.Index,'Count of Job Search URLs',row2[10]+1)                    
count_empall_web()


# In[66]:


all_emp


# In[68]:


emp_looking_for_jobs=all_emp[all_emp['Count of Job Search URLs']>0]


# In[69]:


emp_looking_for_jobs


# In[72]:


emp_looking_for_jobs.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/AllEmpLookingforJobs.csv",index=False)


# In[2]:


emp_looking_for_jobs = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/AllEmpLookingforJobs.csv")


# In[37]:


get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set_style("whitegrid")
ax = sns.pairplot(x="index",y="count of job URLs",data=dftry.reset_index())


# In[16]:


emp_looking_for_jobs.groupby(['role']).count()


# In[17]:


emp_looking_for_jobs['date']

