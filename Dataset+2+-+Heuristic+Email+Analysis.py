
# coding: utf-8
import pandas as pd
import numpy as np
np.set_printoptions(linewidth=120)
np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_row', 1000)
pd.options.display.max_colwidth = 500

email = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_info.csv")
email.head()
suspicious = email[(email['content'].str.contains("growth | initiative | salary | relocation | work"))]
email[~email['from'].str.contains("@dtaa.com")]
suspicious.shape[0]
suspicious_email_senders = suspicious['user'].unique()
suspicious_email_senders
suspicious_email_senders.size

removed_employees = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/Employees_removed.csv")
removed_employees
del removed_employees['Unnamed: 0']
removed_employees.groupby(removed_employees['role']).count()
email['Does the employee still exist?'] = 1

def emails_of_removed_employees():
    for row in removed_employees.itertuples():
        for row2 in email.itertuples():
            if row[2]==row2[3]:
                email.set_value(row2.Index, 'Does the employee still exist?', 0)
emails_of_removed_employees()

email_re = email[email['Does the employee still exist?']==0]
email_e = email[email['Does the employee still exist?']==1]
email_e.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_persisting_employees.csv")
email_re
suspicious_re = email_re[(email_re['content'].str.contains("salary"))]
suspicious_re
suspicious_re.shape[0]

email_re.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_removed_employees.csv")
emails_of_a_user = email_re[email_re['user']=='AKR0057']
emails_of_a_user
email_re[~email_re['from'].str.contains("@dtaa.com")]


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(email['content'])
model=KMeans(n_clusters=2)
model.fit_transform(email)
email['Label']=model.labels_
colormap1=np.array(['Red','Yellow'])
plt.ylabel('x2')
plt.xlabel('x1')
plt.title('K-Means Predicted Clustering')
plt.scatter(email['user'],email['content'],c=colormap1[model.labels_])
plt.show()

email_re = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_removed_employees.csv"
suspicious_re = email_re[(email_re['content'].str.contains("salary"))]
suspicious_re = suspicious_re[~(suspicious_re['from'].str.contains("@dtaa.com"))]
suspicious_re3 = suspicious_re[(suspicious_re['from'].str.contains("@hotmail.com"))]
suspicious_re2 = suspicious_re[(suspicious_re['from'].str.contains("@gmail.com"))]
suspicious_re2['user'].unique()
suspicious_re3['user'].unique()
suspicious_re4 = suspicious_re[(suspicious_re['from'].str.contains("@yahoo.com"))]
suspicious_re4['user'].unique()
suspicious_re2
suspicious_re3
suspicious_re4


email_e=pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_persisting_employees.csv")
suspicious_e = email_e[(email_e['content'].str.contains("salary"))]
suspicious_e = suspicious_e[~(suspicious_e['from'].str.contains("@dtaa.com"))]
suspicious_e1 = suspicious_e[(suspicious_e['from'].str.contains("@hotmail.com"))]
suspicious_e2 = suspicious_e[(suspicious_e['from'].str.contains("@gmail.com"))]
suspicious_e3 = suspicious_e[(suspicious_e['from'].str.contains("@yahoo.com"))]
np.count_nonzero(users)
suspicious_e1['user'].unique()
suspicious_e2['user'].unique()
suspicious_e3['user'].unique()

email_domain = email[email['from'].str.contains("@hotmail.com") | email['from'].str.contains("@gmail.com") | email['from'].str.contains("@yahoo.com")]
email_domain.shape[0]
email_domain.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_onlypersonaldomains.csv")
email_domain=pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_onlypersonaldomains.csv")
del email_domain['Unnamed: 0']
filtered = email_domain[email_domain['content'].str.contains("skills")&email_domain['content'].str.contains("salary")&email_domain['content'].str.contains("resume")&email_domain['content'].str.contains("passion")&email_domain['content'].str.contains("experience")]
filtered


all_emp=pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/2009-12.csv"
all_emp['Count of Job Emails']=0
def count_empall_email():
    for row in filtered.itertuples():
            for row2 in all_emp.itertuples():
                if row[3]==row2[2]:
                    all_emp.set_value(row2.Index,'Count of Job Emails',row2[10]+1)                    
count_empall_email()



all_emp
employees_job_emails = all_emp[all_emp['Count of Job Emails']>0]
employees_job_emails.shape[0]
employees_job_emails.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/AllEmpSendingJobEmailsfrompersonalDomains.csv",index=False)
filtered.groupby(['user']).count()
filtered2 = email[email['content'].str.contains("skills")&email['content'].str.contains("salary")&email['content'].str.contains("resume")&email['content'].str.contains("experience")&email['content'].str.contains("compensation")]
users_job = filtered2.groupby(['user']).count()
users_job[users_job['id']>10]

removed_employees['Count of Job Emails']=0
def count_empremoved_email():
    for row in filtered.itertuples():
            for row2 in removed_employees.itertuples():
                if row[3]==row2[2]:
                    removed_employees.set_value(row2.Index,'Count of Job Emails',row2[10]+1)                    
count_empremoved_email()

employeesremoved_job_emails = removed_employees[removed_employees['Count of Job Emails']>0]
employeesremoved_job_emails.shape[0]
employeesremoved_job_emails.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/RemovedEmpSendingJobEmailsfrompersonalDomains.csv",index=False)

