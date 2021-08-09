#!/usr/bin/env python
# coding: utf-8

# # Predict the customer churn of a telecom company and find out the key drivers that lead to churn
# 

# * Customer churn, is the percentage of customers who stop doing business with an entity

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#setting the working directory
os.chdir('C:/Users/huiga/Desktop/Data')


# In[3]:


#reading the required data files
trainer = pd.read_csv('Telecom_Train.csv')
tester = pd.read_csv('Telecom_Test.csv')


# # Exploratory Data Analysis (EDA)

# In[4]:


#checking the train to test data ratio
tester.shape[0]/trainer.shape[0]


# In[5]:


trainer.shape


# In[6]:


trainer.head()


# In[7]:


trainer.dtypes.sort_values()


# In[8]:


#checking for missing values
trainer.isna().sum()


# In[9]:


trainer.nunique().sort_values()


# In[10]:


trainer.head()


# In[11]:


#Creating a function that does all of the above tasks in one go
def get_basic_stats(dfname):
    print("Shape of dataframe is " + str(dfname.shape))
    print("Below are datatypes of columns in DF")
    print(dfname.dtypes.sort_values())
    print("Below are missing values in each column")
    print(dfname.isna().sum().sort_values())
    print("Below are the number of unique values taken by a column")
    print(dfname.nunique().sort_values())
    print("Below are some records in DF")
    print("Below is distribution of numeric variables")
    print(dfname.describe())
    print(dfname.head())


# In[12]:


get_basic_stats(trainer)


# In[13]:


get_basic_stats(tester)


# In[14]:


#Removing the junk column
trainer = trainer.drop(['Unnamed: 0'], axis =1 )
tester = tester.drop(['Unnamed: 0'], axis = 1)

#creating a copy to keep the original dataframes intact
trainer2 = trainer.copy()
tester2 = tester.copy()


# In[15]:


#creating binary variables from categorical variables that take just 2 unique values
yes_no_vars = ['churn', 'international_plan', 'voice_mail_plan']

def cat_to_binary(df, varname):
    df[varname + '_num'] = df[varname].apply(lambda x : 1 if x == 'yes' else 0)
    print("checking")
    print(df.groupby([varname + '_num', varname]).size())
    return df


for indexer, varname in enumerate(yes_no_vars):
    trainer2 = cat_to_binary(trainer2, varname)
    tester2 = cat_to_binary(tester2, varname)


# In[16]:


#dropping object vars that have been converted to numeric
trainer2 = trainer2.drop(yes_no_vars, axis =1)
tester2 = tester2.drop(yes_no_vars, axis =1)


# In[17]:


#now we are left with just 2 categorical variables
trainer2.dtypes.sort_values()


# * Univariate Analysis

# In[18]:


#univariate analysis of categorical variables
plt.hist(list(trainer2['area_code']))
#plt.show
plt.figure(figsize = (20,10))
plt.hist(list(trainer2['state']), bins = 100)
plt.show()


# In[19]:


trainer2.mean()#this is sufficient for univariate analysis of binary variables


# In[20]:


# Visualizing the churn variable
topie = trainer['churn'].value_counts(sort = True)
print(topie.dtype)
print(topie)
colorss = ["darkgreen","red"] 
plt.pie(topie,labels = topie.index.values, explode= [0, 0.2],  colors=colorss, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('%Churn in Training Data')
plt.show()


# In[21]:


#creating a list of continuous variables, which would be visualized using boxplot
continuous_vars = trainer.select_dtypes([np.number]).columns.tolist()
print(continuous_vars)


# In[22]:


#univariate analysis of continuous variables
type_of_vars = ['intl', 'customer', 'minutes', 'calls', 'charge']
remaining_list = trainer2.columns
for vartype in type_of_vars:
    temp_list = [x for x in remaining_list if vartype in x]
    
    #set(A).difference(set(B)) = A - B -> mathematical term: set contains distinct values
    remaining_list = list(set(remaining_list).difference(set(temp_list)))
    trainer2.boxplot(column=temp_list, figsize = (20,10) )
    plt.title('Boxplot for '+ vartype + ' variables')
    plt.show()


# # Feature Creation

# In[23]:


#Creating a charge per minute variable..in both dataframes
#Intuitively, we expect customer with high value of this variable to have higher churn rate
charge_vars = [x for x in trainer.columns if 'charge' in x]
minutes_vars = [x for x in trainer.columns if 'minutes' in x]
print(charge_vars)
print(minutes_vars)

#df is shorthand for dataframe
def create_cpm(df):
    df['total_charges'] = 0
    df['total_minutes'] = 0
    for indexer in range(0, len(charge_vars)):
        df['total_charges'] +=  df[charge_vars[indexer]]
        df['total_minutes'] +=  df[minutes_vars[indexer]]
        
    #np.where() from numpy, np.where(condition, true value, false value)
    df['charge_per_minute'] = np.where(df['total_minutes'] >0, df['total_charges']/df['total_minutes'], 0)
    
    #when inplace = TRUE, return none, when inplace = false, return a copy
    df.drop(['total_minutes', 'total_charges' ], axis = 1, inplace = True)
    print(df['charge_per_minute'].describe())
    return df


trainer2 = create_cpm(trainer2)
tester2 = create_cpm(tester2)

trainer2.boxplot(column='charge_per_minute', figsize = (20,10) )


# - Bi-variate Analysis

# In[24]:


# drop(), inplace = false thus return a copy of the dropping value
X = trainer2.drop('churn_num', axis=1)


# In[25]:


all_corr = X.corr().unstack().reset_index()

# delete all the duplicates and correlation of the variable itself: acc_length vs acc_length
corr_table = all_corr[all_corr['level_0'] > all_corr['level_1']]

# set column names
corr_table.columns = ['var1', 'var2', 'corr_value']

# create another column 
corr_table['corr_abs'] = corr_table.loc[:, 'corr_value'].abs()
corr_table = corr_table.sort_values(by= ['corr_abs'], ascending = False )
corr_table.head(10)#total_day_charge, total_eve_charge, total_night_charge, total_intl_charge, voice_mail_plan_num
#these 5 variables can be dropped


# In[26]:


#creating a heat map to see the degree of correlation visually
plt.figure(figsize=(13, 13))

# continuous_vars = trainer.select_dtypes([np.number]).columns.tolist()
vg_corr = trainer2[continuous_vars].corr()

sns.heatmap(vg_corr, xticklabels = vg_corr.columns.values,yticklabels = vg_corr.columns.values, annot = True)


# In[27]:


# PLotting each variable against the other..the diagonals show the histogram
sns.pairplot(trainer2[continuous_vars]) 


# * Plotting predictor trends with dependent variable 

# In[28]:


# Plotting PDF of all variables based on Churn..
def create_pdf(df, varname):
    plt.figure(figsize=(20,5))
    plt.hist(list(df[df['churn_num'] == 0 ][varname]), bins = 50, label = 'non-churned', density = True, color = 'g', alpha = 0.8)
    plt.hist(list(df[df['churn_num'] == 1 ][varname]), bins = 50, label = 'churned', density = True, color = 'r', alpha = 0.8)
    plt.legend(loc='upper right')
    plt.xlabel(varname)
    plt.ylabel('Probability Distribution Function')
    plt.show

for varname in trainer2.columns:
    create_pdf(trainer2, varname)


# In[29]:


#we have identified 5 variables that can be dropped because of the high correleations
drop_after_corr = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge', 'voice_mail_plan_num']
trainer3 = trainer2.drop(drop_after_corr, axis=1)
tester3 = tester2.drop(drop_after_corr, axis=1)
trainer3.shape


# In[30]:


#doing one of the 2 categorical varables 
cat_columns = ['state', 'area_code']
trainer3.shape

#only successful once? get_dummies do not work twice because the columns have been removed
temp_trainer = pd.get_dummies(trainer3.loc[:, cat_columns], drop_first = True)
temp_tester = pd.get_dummies(tester3.loc[:, cat_columns], drop_first = True)
trainer3 = pd.concat([temp_trainer, trainer3], axis=1)
tester3 = pd.concat([temp_tester, tester3], axis=1)
trainer3 = trainer3.drop(cat_columns, axis = 1)
tester3 = tester3.drop(cat_columns, axis = 1)


# In[31]:


print(trainer3.shape)
print(tester3.shape)


# # Modeling and Performance

# In[32]:


X_train = trainer3.drop('churn_num',axis=1)
Y_train = trainer3['churn_num']
X_test = tester3.drop('churn_num', axis = 1)
Y_test = tester3['churn_num'] 


# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, average_precision_score, f1_score, confusion_matrix, roc_auc_score,auc, accuracy_score, log_loss, roc_curve, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier


# In[34]:


# Logistic Regression with hyper-parameter tuning
lr=LogisticRegression(random_state=42, solver='liblinear')
param_gridd = { 'penalty': ['l1', 'l2'], 'C': [0.1, 1, 2, 3, 5]}
CV_lr = GridSearchCV(estimator=lr, param_grid=param_gridd, cv= 5)#do this with 5 folds
CV_lr.fit(X_train, Y_train)
lr_best = CV_lr.best_estimator_


# In[35]:


lr_best


# In[36]:


test_score_lr = lr_best.predict_proba(X_test)[:, 1]
pd.Series(test_score_lr).describe()


# In[46]:


# Gradient Boosting with hyper-parameter tuning
gbr=GradientBoostingClassifier(random_state=42)
param_grid = { 
'n_estimators': [50, 100, 500], 'max_features': ['auto'], 'learning_rate': [0.01, 0.05, 0.1, 0.2]
}
CV_gbr = GridSearchCV(estimator=gbr, param_grid=param_grid, cv= 5)
CV_gbr.fit(X_train, Y_train)
gbr_best = CV_gbr.best_estimator_
print(gbr_best)


# In[47]:


test_score_gbm = gbr_best.predict_proba(X_test)[:, 1]
pd.Series(test_score_gbm).describe()


# * Performance Comparison of Models

# In[48]:


#Area Under ROC and PR curves for LR model
roc_auc = (roc_auc_score(Y_test, test_score_lr, average='macro'))
avg_pre = average_precision_score(Y_test, test_score_lr)
print (roc_auc)
print (avg_pre)


# In[49]:


#Area Under ROC and PR curves for GBM model
roc_auc_gbm = (roc_auc_score(Y_test, test_score_gbm, average='macro'))
avg_pre_gbm = average_precision_score(Y_test, test_score_gbm)
print (roc_auc_gbm)
print (avg_pre_gbm)


# In[50]:


fpr_gbm, tpr_gbm, _ =roc_curve(Y_test, test_score_gbm)
plt.plot(fpr_gbm, tpr_gbm, label ='GBM')
fpr_lr, tpr_lr, _ =roc_curve(Y_test, test_score_lr)
plt.plot(fpr_lr, tpr_lr, label ='LR')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend()


# In[51]:


precision_gbm, recall_gbm, _ =precision_recall_curve(Y_test, test_score_gbm)
plt.plot(recall_gbm, precision_gbm, label ='GBM')
precision_lr, recall_lr, _ =precision_recall_curve(Y_test, test_score_lr)
plt.plot(recall_lr, precision_lr, label ='LR')
plt.xlabel('Recall'); 
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.legend()


# In[52]:


import seaborn as sns
import matplotlib.pyplot as plt     
cm = confusion_matrix(Y_test, (test_score_gbm >= 0.5)) 
ax= plt.subplot()
sns.heatmap(cm, annot=True,  ax = ax, fmt='g')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['retained', 'churned']); ax.yaxis.set_ticklabels(['retained', 'churned']);


# In[53]:


print(accuracy_score(Y_test, (test_score_lr >= 0.5), normalize=True))
print(accuracy_score(Y_test, (test_score_gbm >= 0.5), normalize=True))


# In[86]:


def get_FI(modelname, dfname):
    feature_importance = pd.DataFrame([X_train.columns.tolist(), gbr_best.feature_importances_ ]).T
    feature_importance.columns = ['varname', 'importance']
    feature_importance = feature_importance.sort_values(by=['importance'], ascending=False)
    feature_importance['cum_importance'] = feature_importance['importance'].cumsum()
    return feature_importance


# In[87]:


def get_FI(modelname, dfname):
    importance_list = pd.DataFrame(modelname.feature_importances_, columns=['importance'])
    varnames_list = pd.DataFrame(dfname.columns.tolist(), columns=['feature'])
    feature_importance = pd.concat([varnames_list, importance_list], axis=1)
    feature_importance.reindex()
    feature_importance = feature_importance.sort_values(by=['importance'], ascending=False)
    feature_importance['cum_importance'] = feature_importance['importance'].cumsum()
    return feature_importance

get_FI(gbr_best, X_train)


# # Recursive Feature Elimination

# In[88]:


state_vars = [x for x in X_train.columns if 'state' in x]
area_vars = [x for x in X_train.columns if 'area' in x]
rfe_vars = state_vars + area_vars
print(len(rfe_vars))
X_train_rfe = X_train.drop(rfe_vars, axis=1 ) 
X_test_rfe = X_test.drop(rfe_vars, axis=1)
X_test_rfe.shape


# In[89]:


# Gradient Boosting on reduced feature set
gbr=GradientBoostingClassifier(random_state=42)
param_grid = { 
'n_estimators': [50, 100, 500], 'max_features': ['auto'], 'learning_rate': [0.01, 0.05, 0.1, 0.2]
}
CV_gbr = GridSearchCV(estimator=gbr, param_grid=param_grid, cv= 5)
CV_gbr.fit(X_train_rfe, Y_train)
gbr_best_rfe = CV_gbr.best_estimator_
print(gbr_best_rfe)


# In[90]:


test_score_rfe = gbr_best_rfe.predict_proba(X_test_rfe)[:, 1]
print(pd.Series(test_score_rfe).describe())
roc_auc_gbm = (roc_auc_score(Y_test, test_score_rfe, average='macro'))
avg_pre_gbm = average_precision_score(Y_test, test_score_rfe)
print (roc_auc_gbm)
print (avg_pre_gbm)


# In[91]:


FI = get_FI(gbr_best_rfe, X_train_rfe)
print(FI)


# # Key Drivers of Churn

# In[92]:


vals = list(FI.loc[:, 'importance'])
plt.barh(FI.loc[:, 'feature'], FI.loc[:, 'importance'])
plt.title('Importance of different variables')
plt.gca().xaxis.grid(linestyle=':')


# # Model Implementation

# In[93]:


#saving the required files
import pickle
model_columns = list(X_train_rfe.columns)
pickle.dump(gbr_best_rfe, open('model.pkl', 'wb'))
pickle.dump(model_columns, open('model_columns.pkl', 'wb'))


# * Run the python code churn_API.py using command prompt and then do the steps below 

# In[94]:


import json
import requests
churn_url = 'http://127.0.0.1:8888' # change to required url


# In[95]:


churn_dict = dict(trainer.iloc[1])


# In[96]:


churn_dict ={'state': 'OH',
 'account_length': 107,
 'area_code': 'area_code_415',
 'international_plan': 'no',
 'voice_mail_plan': 'yes',
 'number_vmail_messages': 26,
 'total_day_minutes': 161.6,
 'total_day_calls': 123,
 'total_day_charge': 27.47,
 'total_eve_minutes': 195.5,
 'total_eve_calls': 103,
 'total_eve_charge': 16.62,
 'total_night_minutes': 254.4,
 'total_night_calls': 103,
 'total_night_charge': 11.45,
 'total_intl_minutes': 13.7,
 'total_intl_calls': 3,
 'total_intl_charge': 3.7,
 'number_customer_service_calls': 1,
 'churn': 'no'}


# In[97]:


#Requesting the API for a result
churn_json = json.dumps(churn_dict)
send_request= requests.post(churn_url, churn_json)
print(send_request)#200 means we got the result, 500 means there was an error in processing
send_request.json()


# In[ ]:


#Checking the above results
train_score_rfe = gbr_best_rfe.predict_proba(X_train_rfe)[:, 1]
results_check = pd.concat([X_train_rfe, pd.Series(train_score_rfe, name = 'model_score')], axis=1)
results_check.iloc[1,:]

