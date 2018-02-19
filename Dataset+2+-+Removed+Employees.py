
# coding: utf-8
import pandas as pd
import numpy as np
#np.set_printoptions(linewidth=120)
#np.set_printoptions(threshold=np.nan)
#pd.set_option('display.max_row', 1000)
#pd.options.display.max_colwidth = 500
employees_start = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/2009-12.csv")
employees_end = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/2011-05.csv")
employees_left = employees_start.merge(employees_end, indicator=True, how='outer')
employees_removed=employees_left[employees_left['_merge'] == 'left_only']
del employees_removed['_merge']
employees_removed.shape[0]
employees_removed.to_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/LDAP/Employees_removed.csv")

