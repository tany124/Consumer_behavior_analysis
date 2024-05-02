#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd,numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[3]:


df=pd.read_csv("social_ads.csv")


# In[9]:


df.head()


# In[10]:


df.shape


# In[11]:


df.info()


# In[12]:


# Checking for null values
df.isnull().sum()


# In[19]:


df[['Age','EstimatedSalary','Purchased']].describe()


# In[34]:


#As the data set is small and contains no missing or incorrect values to clean , we will proceed with the analysis of data
#Univariate analysis
#Age
# Box plot
plt.boxplot(df.Age)
plt.show()


# In[38]:


def cal_range(val):
    if val>=10 and val< 20:
        return ("10-20")
    elif val>=20 and val< 30:
        return ("20-30")
    elif val>=30 and val< 40:
        return ("30-40")
    elif val>=40 and val<50:
        return ("40-50")
    else:
        return ("50+")
    
df['Age_Range']=df.Age.apply(cal_range)
color=['green','yellow','blue','red','purple']
df.Age_Range.value_counts().plot.bar(color=color)
plt.show()


# ### Univariate analysis

# In[25]:


#Estimated Salary
#Box plot
plt.boxplot(df.EstimatedSalary)
plt.show()


# In[42]:


def cal_salary_range(val):
    if val>=10000 and val< 30000:
        return ("10000-30000")
    elif val>=30000 and val< 50000:
        return ("30000-50000")
    elif val>=50000 and val< 70000:
        return ("50000-70000")
    elif val>=70000 and val<90000:
        return ("70000-90000")
    elif val>=90000 and val<100000:
        return ("90000-100000")
    else:
        return ("100000+")
df['Salary_Range']=df.EstimatedSalary.apply(cal_salary_range)
color=['red','yellow','blue','green','purple','seagreen']
df.Salary_Range.value_counts().plot.bar(color=color)
plt.show()


# In[33]:


#Target variable
# Purchased
(df.Purchased.value_counts(normalize=True)*100).plot.pie(y='Purchased',autopct="%1.0f%%")
plt.show()


# ### Bivariate Analysis

# ### Numerical vs Numerical

# In[43]:


#Bivariate Analysis
#Age vs Salary
plt.scatter(df.Age,df.EstimatedSalary)
plt.show()


# In[79]:


# It is observed from the scatter plot that there is no almost insignificant dependency between salary and age


# ### Numerical vs Categorical

# In[83]:


# Checking relation between age_group and salary
df.groupby("Age_Range")['EstimatedSalary'].mean().plot.bar()
plt.ylabel("Estimated Salary")
plt.show()


# ### Categorical vs Categorical

# In[70]:


#Age vs response
df.groupby('Age_Range')['Purchased'].mean().plot.bar()
plt.title("Age range vs Response")
plt.show()


# In[53]:


#Box plot between age and response
sns.boxplot(x=df.Purchased,y=df.Age)
plt.show()


# In[69]:


#Salary vs response
df.groupby('Salary_Range')['Purchased'].mean().plot.bar()
plt.show()


# In[52]:


#Box plot between salary and response
sns.boxplot(x=df.Purchased,y=df.EstimatedSalary)
plt.show()


# In[71]:


#Subplots of age vs response and salary vs response
plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
plt.title("Age range vs Response")
df.groupby('Age_Range')['Purchased'].mean().plot.bar()
plt.subplot(1,2,2)
plt.title("Salary range vs Response")
df.groupby('Salary_Range')['Purchased'].mean().plot.bar()
plt.show()



# ### Prediction of response variable using Logistic regression

# In[86]:


#As there are currently no categorical variables in data set apart from target variable, we do not have to create dummy features
from sklearn.model_selection import train_test_split
#Putting feature variable to X
X=df.drop(['Age_Range','Salary_Range','Purchased'],axis=1)
X.head()


# In[88]:


#Putting response variable to y
y=df['Purchased']
y.head()


# In[89]:


#Splitting the data into train test data
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=100)


# In[92]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sclaer=StandardScaler()
X_train[['Age','EstimatedSalary']]=sclaer.fit_transform(X_train[['Age','EstimatedSalary']])
X_train.head()


# ### Model Building

# In[93]:


import statsmodels.api as sm


# In[95]:


#Logistics Regression model
logm1=sm.GLM(y_train,(sm.add_constant(X_train)),family=sm.families.Binomial())
logm1.fit().summary()


# In[101]:


#feature selection using RFE
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()


# In[105]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg,n_features_to_select=2)
rfe=rfe.fit(X_train,y_train)


# In[106]:


rfe.support_


# In[107]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[108]:


col=X_train.columns[rfe.support_]


# #### Assessing the model with stats model

# In[109]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[110]:


#Getting predicted values on trained set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[111]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[113]:


# Creating dataframe with actual and predicted probability


# In[115]:


y_train_pred_final=pd.DataFrame({'Response':y_train.values,'Response_Prob':y_train_pred})
y_train_pred_final['Response_ID']=y_train.index
y_train_pred_final.head()


# In[118]:


#Creating new column 'predicted' with 1 if Response_prob > 0.6 else 0


# In[119]:


y_train_pred_final['predicted']=y_train_pred_final.Response_Prob.map(lambda x: 1 if x>0.6 else 0)
y_train_pred_final.head(20)


# In[121]:


from sklearn import metrics


# In[123]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Response, y_train_pred_final.predicted )
print(confusion)


# In[124]:


# Predicted     not_churn    churn
# Actual
# not_churn        3270      365
# churn            579       708  


# In[125]:


#Accuracy
print(metrics.accuracy_score(y_train_pred_final.Response, y_train_pred_final.predicted))


# #### Checking VIFs

# In[126]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[127]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[128]:


#As VIF for both the fields is less than 5, we do not need to drop any and proceed with making predictions


# In[129]:


#Plotting the ROC Curve


# In[130]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[131]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Response, y_train_pred_final.Response_Prob, drop_intermediate = False )


# In[132]:


draw_roc(y_train_pred_final.Response, y_train_pred_final.Response_Prob)


# In[133]:


# Finding the optimal cut off point


# ### Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[134]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Response_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[135]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Response, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[136]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# #### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

# In[137]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Response_Prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[138]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Response, y_train_pred_final.final_predicted)


# In[139]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Response, y_train_pred_final.final_predicted )
confusion2


# In[140]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[141]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[142]:


# Let us calculate specificity
TN / float(TN+FP)


# In[144]:


# Calculate false postive rate - predicting consumer will buy when customer will not buy in actual
print(FP/ float(TN+FP))


# In[145]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[146]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### Precision and Recall

# In[147]:


#Looking at the confusion matrix again


# In[148]:


confusion = metrics.confusion_matrix(y_train_pred_final.Response, y_train_pred_final.predicted )
confusion


# In[ ]:


#Precision
# TP / TP + FP


# In[149]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[ ]:


#Recall
#TP/TP+FN


# In[150]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[151]:


#Precision Recall Trade off


# In[152]:


from sklearn.metrics import precision_recall_curve


# In[153]:


y_train_pred_final.Response, y_train_pred_final.predicted


# In[154]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Response, y_train_pred_final.Response_Prob)


# In[155]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ### Making predictions on test data set

# In[179]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_test[['Age','EstimatedSalary']]=scaler.fit_transform(X_test[['Age','EstimatedSalary']])


# In[180]:


X_test = X_test[col]
X_test.head()


# In[181]:


X_test_sm = sm.add_constant(X_test)
X_test_sm


# In[182]:


#Making predictions on dataset
y_test_pred = res.predict(X_test_sm)


# In[183]:


y_test_pred[:20]


# In[184]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[185]:


# Let's see the head
y_pred_1.head()


# In[186]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[187]:


# Putting CustID to index
y_test_df['ResponseID'] = y_test_df.index


# In[188]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[189]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[190]:


y_pred_final.head()


# In[191]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Response_Prob'})


# In[192]:


# Rearranging the columns
y_pred_final = y_pred_final.reindex(['ResponseID','Purchased','Response_Prob'], axis=1)


# In[193]:


y_pred_final.head(30)


# In[195]:


y_pred_final['final_predicted'] = y_pred_final.Response_Prob.map(lambda x: 1 if x > 0.25 else 0)


# In[196]:


y_pred_final.head()


# In[197]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Purchased, y_pred_final.final_predicted)


# In[198]:


confusion2 = metrics.confusion_matrix(y_pred_final.Purchased, y_pred_final.final_predicted )
confusion2


# In[199]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[200]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[201]:


# Let us calculate specificity
TN / float(TN+FP)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




