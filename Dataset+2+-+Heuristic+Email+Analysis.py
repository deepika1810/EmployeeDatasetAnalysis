
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
np.set_printoptions(linewidth=120)
np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_row', 1000)
pd.options.display.max_colwidth = 500


# In[72]:


email = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_info.csv")


# In[3]:


email.head()


# In[4]:


suspicious = email[(email['content'].str.contains("growth | initiative | salary | relocation | work"))]


# In[5]:


email[~email['from'].str.contains("@dtaa.com")]


# In[6]:


suspicious.shape[0]


# In[7]:


suspicious_email_senders = suspicious['user'].unique()


# In[8]:


suspicious_email_senders


# In[9]:


suspicious_email_senders.size


# In[18]:


removed_employees = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/Employees_removed.csv")


# In[20]:


removed_employees


# In[19]:


del removed_employees['Unnamed: 0']


# In[13]:


removed_employees.groupby(removed_employees['role']).count()


# In[8]:


email['Does the employee still exist?'] = 1


# In[9]:


def emails_of_removed_employees():
    for row in removed_employees.itertuples():
        for row2 in email.itertuples():
            if row[2]==row2[3]:
                email.set_value(row2.Index, 'Does the employee still exist?', 0)
emails_of_removed_employees()


# In[10]:


email_re = email[email['Does the employee still exist?']==0]


# In[11]:


email_e = email[email['Does the employee still exist?']==1]
email_e.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_persisting_employees.csv")


# In[17]:


email_re


# In[1]:


suspicious_re = email_re[(email_re['content'].str.contains("salary"))]


# In[19]:


suspicious_re


# In[20]:


suspicious_re.shape[0]


# In[1]:


email_re.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_removed_employees.csv")


# In[22]:


emails_of_a_user = email_re[email_re['user']=='AKR0057']


# In[23]:


emails_of_a_user


# In[28]:


email_re[~email_re['from'].str.contains("@dtaa.com")]


# In[30]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# In[32]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[33]:


transformer = TfidfTransformer(smooth_idf=False)


# In[34]:


tfidf = transformer.fit_transform(email['content'])


# In[31]:


model=KMeans(n_clusters=2)
model.fit_transform(email)
email['Label']=model.labels_
colormap1=np.array(['Red','Yellow'])
plt.ylabel('x2')
plt.xlabel('x1')
plt.title('K-Means Predicted Clustering')
plt.scatter(email['user'],email['content'],c=colormap1[model.labels_])
plt.show()


# In[37]:


email_re = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_removed_employees.csv")


# In[55]:


suspicious_re = email_re[(email_re['content'].str.contains("salary"))]


# In[56]:


suspicious_re = suspicious_re[~(suspicious_re['from'].str.contains("@dtaa.com"))]


# In[57]:


suspicious_re3 = suspicious_re[(suspicious_re['from'].str.contains("@hotmail.com"))]


# In[58]:


suspicious_re2 = suspicious_re[(suspicious_re['from'].str.contains("@gmail.com"))]


# In[59]:


suspicious_re2['user'].unique()


# In[60]:


suspicious_re3['user'].unique()


# In[61]:


suspicious_re4 = suspicious_re[(suspicious_re['from'].str.contains("@yahoo.com"))]


# In[62]:


suspicious_re4['user'].unique()


# In[44]:


suspicious_re2


# In[45]:


suspicious_re3


# In[46]:


suspicious_re4


# In[3]:


email_e=pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_persisting_employees.csv")


# In[63]:


suspicious_e = email_e[(email_e['content'].str.contains("salary"))]


# In[64]:


suspicious_e = suspicious_e[~(suspicious_e['from'].str.contains("@dtaa.com"))]


# In[65]:


suspicious_e1 = suspicious_e[(suspicious_e['from'].str.contains("@hotmail.com"))]


# In[66]:


suspicious_e2 = suspicious_e[(suspicious_e['from'].str.contains("@gmail.com"))]


# In[67]:


suspicious_e3 = suspicious_e[(suspicious_e['from'].str.contains("@yahoo.com"))]


# In[68]:


np.count_nonzero(users)


# In[69]:


suspicious_e1['user'].unique()


# In[70]:


suspicious_e2['user'].unique()


# In[71]:


suspicious_e3['user'].unique()


# In[80]:


email_domain = email[email['from'].str.contains("@hotmail.com") | email['from'].str.contains("@gmail.com") | email['from'].str.contains("@yahoo.com")]


# In[82]:


email_domain.shape[0]


# In[83]:


email_domain.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_onlypersonaldomains.csv")


# In[6]:


email_domain=pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_onlypersonaldomains.csv")
del email_domain['Unnamed: 0']


# In[7]:


filtered = email_domain[email_domain['content'].str.contains("skills")&email_domain['content'].str.contains("salary")&email_domain['content'].str.contains("resume")&email_domain['content'].str.contains("passion")&email_domain['content'].str.contains("experience")]


# In[8]:


filtered


# In[9]:


all_emp=pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/2009-12.csv")


# In[12]:


all_emp['Count of Job Emails']=0
def count_empall_email():
    for row in filtered.itertuples():
            for row2 in all_emp.itertuples():
                if row[3]==row2[2]:
                    all_emp.set_value(row2.Index,'Count of Job Emails',row2[10]+1)                    
count_empall_email()


# In[13]:


all_emp


# In[14]:


employees_job_emails = all_emp[all_emp['Count of Job Emails']>0]


# In[16]:


employees_job_emails.shape[0]


# In[17]:


employees_job_emails.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/AllEmpSendingJobEmailsfrompersonalDomains.csv",index=False)


# In[89]:


filtered.groupby(['user']).count()


# In[90]:


filtered2 = email[email['content'].str.contains("skills")&email['content'].str.contains("salary")&email['content'].str.contains("resume")&email['content'].str.contains("experience")&email['content'].str.contains("compensation")]


# In[95]:


users_job = filtered2.groupby(['user']).count()


# In[97]:


users_job[users_job['id']>10]


# In[21]:


removed_employees['Count of Job Emails']=0
def count_empremoved_email():
    for row in filtered.itertuples():
            for row2 in removed_employees.itertuples():
                if row[3]==row2[2]:
                    removed_employees.set_value(row2.Index,'Count of Job Emails',row2[10]+1)                    
count_empremoved_email()


# In[22]:


employeesremoved_job_emails = removed_employees[removed_employees['Count of Job Emails']>0]


# In[24]:


employeesremoved_job_emails.shape[0]


# In[25]:


employeesremoved_job_emails.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/RemovedEmpSendingJobEmailsfrompersonalDomains.csv",index=False)

