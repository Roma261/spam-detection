#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


col_names = ['UserDescriptionLength','UserFollowersCount','UserFriendsCount','AvgHashtag','AvgURLCount','AvgMention','AvgRetweet','TweetCount','SpammerOrNot' ]


# In[3]:


df = pd.read_csv('s.csv', sep=";")


# In[4]:


df = df.iloc[:,2:]


# In[5]:


df


# In[6]:


len(df[df['SpammerOrNot'] == 0]) / len(df[df['SpammerOrNot'] == 1])


# In[7]:


plt.bar(['Spammer', 'Legitimate'], df['SpammerOrNot'].value_counts(ascending=True))
plt.show()


# In[8]:


# Checking imbalance ratio
ratio = len(df[df['SpammerOrNot'] == 0]) / len(df[df['SpammerOrNot'] == 1])
print(f'Ratio between Spammer : Legitimate user = 1:{ratio:.4f}')


# In[9]:


df.info()


# In[10]:


df.drop(['UserLocation', 'AvgFavCount'],axis = 1,  inplace = True)


# In[11]:


df


# The user is node. It has in degree due to number of people following the user. It has out degree, the number of people he is following - friends. 
# 
# Avg Hashtag, AvgURLCount in post - the number of links used in posting two content based features
# 
# AvgMention - the number of nodes(users) it mentions in post, meaning the number of nodes to be involed in activity of the user.
# 
# AvgRetweet - the number of retweets done in a day, the number of how many nodes were reached in a day
# 
# AvgRetweet and AvgMention - How many nodes were involved and connected by the user during his activity.
# 
# TweetCount - number of tweets - posts that were done by the user

# In[12]:


df[df['SpammerOrNot'] == 1].hist(bins=100, figsize=(20, 15))
plt.show()
df[df['SpammerOrNot'] == 0].hist(bins=100, figsize=(20, 15))
plt.show()


# ## EDA

# In[13]:


## User desctipion length


# In[14]:


df[df['SpammerOrNot'] == 1]['UserDescriptionLength'].describe()


# In[15]:


df[df['SpammerOrNot'] == 0]['UserDescriptionLength'].describe()


# In[16]:


plt.plot(df[df['SpammerOrNot'] == 1]['AvgRetweet'].values) # User behavior feature
plt.plot(df[df['SpammerOrNot'] == 0]['AvgRetweet'].values)
plt.legend(['Spammer', 'Legitimate'])
plt.title('Avg Retweet')


# In[17]:


plt.bar(['Spammer', 'Legitimate'], [df[df['SpammerOrNot'] == 1]['AvgRetweet'].describe()['mean'],df[df['SpammerOrNot'] == 0]['AvgRetweet'].describe()['mean']])
plt.title('AvgRetweet')

plt.show()


# In[18]:


## Hashtag and URL count - considered as insignificant features - Content features 
## Why?  may be due to the fact that usually, spammers share only one link, which is also common for legitimate users
df[df['SpammerOrNot'] == 1]['AvgHashtag'].describe()


# In[19]:


df[df['SpammerOrNot'] == 0]['AvgHashtag'].describe()


# In[20]:


plt.plot(df[df['SpammerOrNot'] == 1]['AvgURLCount'])
plt.plot(df[df['SpammerOrNot'] == 0]['AvgURLCount'])
plt.legend(['Spammer', 'Legitimate'])
plt.title('Avg URL Count')


# In[21]:


## User description length


# In[22]:


df['UserDescriptionLength'].describe()


#  0 - no description 
#  1- 10 Name, profession
#  10 - 50 - Name, hobby, profession
#  50 - 160 - Interests, thoughts, goals and etc. 

# In[23]:


df['typeofdescription'] = df['UserDescriptionLength'].map(lambda x: 'no description' if x == 0 else('basic info' if 1<x<=10  else('interests included' if 10<x<50 else 'full description')))


# In[24]:


colors = ["windows blue", "amber", "grey", "green"]


# In[25]:


sns.countplot(data=df[df['SpammerOrNot']==1],x="typeofdescription",palette=sns.xkcd_palette(colors))
plt.title('Barplot of Desciption Type(Spammer)', fontsize=18) #we can specify the title
plt.xlabel('Type of Description', fontsize=16)  #we can name x-axis
plt.ylabel('frequency', fontsize=16) #we can name y-axis
plt.show()

sns.countplot(data=df[df['SpammerOrNot']==0],x="typeofdescription",palette=sns.xkcd_palette(colors))
plt.title('Barplot of Description Type(Legitimate)', fontsize=18) #we can specify the title
plt.xlabel('Type of Description', fontsize=16)  #we can name x-axis
plt.ylabel('frequency', fontsize=16) #we can name y-axis
plt.show()


# ## New features to be created

#  Included in model: 
#  UserDescriptionLength, AvgRetweet, AvgMention and 5 below
#  
#  Try all above plus Hashtag and URL(if use this one try to exlpain why)
# 
#  Features to be done:
#  
# 1.Copeland score
#  
# 2.Degree ratio score
# 
# 3.Account age: Current time - UserCreatedAt
# 
# 4.Tweet per day: Tweet Count / Longetivity (!) 
# 
# 5.UserFriendsCount / Age of account ( Longetivity)  - the rate of creating new edges (connections) 
# 

# ## Think how to connect with network, crowdsourcing 

# Copeland score, degree ratio score
# 
# Degree ratio score works well in this situation, as it is out degree of node / in degree of node which is supposed to be very high for spammer, as not many people follow them while they follow thousands of people. 

# ## Feature Engineering

# In[26]:


## SET HYPOTHESIS FOR EACH NEW DERIVED FEATURES


# In[27]:


df


# ## Copeland score

# In[28]:


# Copeland score # High for spammers and really low for legitimate users
df['copeland score'] = df['UserFriendsCount'] - df['UserFollowersCount']


# In[29]:


df['copeland score']


# In[30]:


plt.plot(df[df['SpammerOrNot'] == 1]['copeland score'].values)
plt.title('Copeland Score Spammer')


# In[31]:


plt.plot(df[df['SpammerOrNot'] == 0]['copeland score'].values)
plt.title('Copeland Score Legitimate')


# ### Degree ratio score

# In[32]:


df['degreeratio'] =  df['UserFriendsCount'] / df['UserFollowersCount']


# In[33]:


plt.plot(df[df['SpammerOrNot'] == 1]['degreeratio'].values)
plt.plot(df[df['SpammerOrNot'] == 0]['degreeratio'].values)
plt.legend(['Spammer', 'Legitimate'])
plt.title('Degree Ratio Score')


# ### Account age

# In[34]:


df['UserCreatedAt']


# In[35]:


df['Current Time']


# In[36]:


from datetime import datetime
date_format = "%Y-%m-%d %H:%M:%S"
n = df['Current Time'].map(lambda x: datetime.strptime(x,date_format))
m = df['UserCreatedAt'].map(lambda x: datetime.strptime(x,date_format))
df['Account age'] = (n - m)/np.timedelta64(1, 'D')


# In[37]:


df[df['SpammerOrNot'] == 1]['Account age'].hist()
plt.title('Spammer')


# In[38]:


df[df['SpammerOrNot'] == 0]['Account age'].hist()
plt.title('Legitimate User')


# ### Tweet per day: Tweet Count / Longetivity 

# In[39]:


df['Tweetperday'] = df['TweetCount'] / df['Account age'] # should be high for spammers and low for legitimate users


# In[40]:


plt.plot(df[df['SpammerOrNot'] == 1]['Tweetperday'].values)
plt.plot(df[df['SpammerOrNot'] == 0]['Tweetperday'].values)
plt.legend(['Spammer', 'Legitimate'])
plt.title('Tweet per day')


# ### UserFriendsCount / Age of account ( Longetivity)  - the rate of creating new edges (connections) 
# 
# high for spammers, and low for legitimate users
# 

# In[41]:


df['rate'] = df['UserFriendsCount'] / df['Account age']


# In[42]:


plt.plot(df[df['SpammerOrNot'] == 1]['rate'].values)
plt.plot(df[df['SpammerOrNot'] == 0]['rate'].values)
plt.legend(['Spammer', 'Legitimate'])
plt.title('The rate of creating new edges (connections)')


# In[43]:


df[df['SpammerOrNot'] == 1]['rate'].describe()


# In[44]:


df[df['SpammerOrNot'] == 0]['rate'].describe()


# In[45]:


import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from(
    [('A', 'B'),('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
     ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])

val_map = {'A': 1.0,
           'D': 0.5714285714285714,
           'H': 0.0}
         

values = [val_map.get(node, 0.25) for node in G.nodes()]

# Specify the edges you want here
red_edges = [('A', 'C'), ('E', 'C')]
edge_colours = ['black' if not edge in red_edges else 'red'
                for edge in G.edges()]
black_edges = [edge for edge in G.edges() if edge not in red_edges]

# Need to create a layout when doing
# separate calls to draw nodes and edges
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                       node_color = values, node_size = 500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=True)
plt.show()


# ## Creating Models

# In[46]:


df


# In[47]:


df = df[['UserDescriptionLength', 'AvgMention', 'AvgRetweet', 'copeland score', 'degreeratio', 'Account age','Tweetperday', 'rate', "SpammerOrNot"]]


# In[48]:


from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(df, test_size=0.2, random_state=1, stratify=df[['SpammerOrNot']])


# In[49]:


# Checking imbalance ratio
ratio = len(training_set[training_set['SpammerOrNot'] == 0]) / len(training_set[training_set['SpammerOrNot'] == 1])
print(f'Ratio between Spammer : Legitimate user = 1:{ratio:.4f}')


# In[50]:


# Checking imbalance ratio
ratio = len(test_set[test_set['SpammerOrNot'] == 0]) / len(test_set[test_set['SpammerOrNot'] == 1])
print(f'Ratio between Spammer : Legitimate user = 1:{ratio:.4f}')


# In[51]:


num_feat = ['UserDescriptionLength', 'AvgMention', 'AvgRetweet', 'copeland score', 'degreeratio', 'Account age','Tweetperday', 'rate']


# In[52]:


# copy data for preventing damage in raw training data
data = training_set.copy()
data1=test_set.copy()


# In[53]:


X_train = data.iloc[:,:-1]
y_train = data.iloc[:,-1]
X_test = data1.iloc[:,:-1]
y_test = data1.iloc[:,-1]


# In[54]:


# import scaler
from sklearn.preprocessing import StandardScaler


# In[55]:


scalar = StandardScaler()
scalar.fit(X_train[num_feat].values)
X_train = scalar.transform(X_train[num_feat].values)
X_test=scalar.transform(X_test[num_feat].values)


# In[56]:


X_test.shape


# In[57]:


X_train.shape


# ## Training models

# In[58]:


from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, precision_recall_fscore_support,roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


# In[59]:


from sklearn.linear_model import LogisticRegression
log_reg_cw = LogisticRegression(class_weight='balanced',max_iter=800)
log_reg_cw.fit(X_train, y_train)
f1_score_logreg = f1_score(y_train, log_reg_cw.predict(X_train))


# In[60]:


from sklearn.ensemble import RandomForestClassifier
randfor = RandomForestClassifier(n_estimators=100, random_state=0,class_weight='balanced',max_depth=5)
randfor.fit(X_train,y_train)
f1_score_randfor= f1_score(y_train, randfor.predict(X_train))


# In[61]:


from sklearn import svm
suppvm = svm.SVC(kernel='poly',class_weight='balanced')
suppvm.fit(X_train, y_train)
f1_score_suppvm= f1_score(y_train, suppvm.predict(X_train))


# In[62]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 15,weights='distance') 
knn.fit(X_train, y_train) 
f1_score_knn= f1_score(y_train, knn.predict(X_train))


# ## Stratified K fold cross validation

# In[63]:


from sklearn.model_selection import KFold,StratifiedKFold
strfold = StratifiedKFold(n_splits = 5,random_state= None,shuffle=False)


# In[64]:


from sklearn.model_selection import cross_val_score
def validation(model, x, y):
    scores = cross_val_score(model, x, y, scoring="f1", cv= strfold)
    return scores.mean()


# In[65]:


val_logreg=validation(log_reg_cw,X_train,y_train)
val_randfor=validation(randfor, X_train, y_train)
val_suppvm=validation(suppvm, X_train, y_train)
val_knn=validation(knn, X_train, y_train)


# In[66]:


evaluation = pd.DataFrame({'Model': ['Logistic regression', 'Random Forest',
              'SVM(poly)', 'k-NN'],
                           'F1 Score on training set': [f1_score_logreg, 
             f1_score_randfor, 
             f1_score_suppvm, f1_score_knn],
                           'F1 using stratified 10-fold cross validation': [val_logreg, 
              val_randfor, 
              val_suppvm, val_knn]})
evaluation = evaluation.set_index('Model')


# In[67]:


evaluation


# In[83]:


f1_score(y_test, randfor.predict(X_test))


# In[68]:


plot_confusion_matrix(randfor, X_test, y_test)
plt.title('Random Forest')


# In[69]:


#random forest
y_proba= randfor.predict_proba(X_test)[:, 1]
roc_auc_logreg=roc_auc_score(y_test, y_proba)


# In[70]:


from sklearn.metrics import auc, roc_curve
#Random Forest
fpr, tpr, threshold = roc_curve(y_test, y_proba)
rocauc = auc(fpr, tpr)


# In[71]:


plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest (area={rocauc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()


# ## Ensemble Model

# In[72]:


from xgboost import XGBClassifier


# In[73]:


xgb = XGBClassifier(random_state = 0,scale_pos_weight = 2, max_depth = 1)


# In[74]:


xgb.fit(X_train, y_train,)
f1_score_xgb = f1_score(y_train, xgb.predict(X_train))


# In[75]:


val_xgb=validation(xgb, X_train, y_train)
val_xgb


# In[76]:


f1_score(y_test, xgb.predict(X_test))


# In[77]:


plot_confusion_matrix(xgb, X_test, y_test)
plt.title('XGBoost Classifier')


# In[78]:


y_proba1= xgb.predict_proba(X_test)[:, 1]
roc_auc_xgb =roc_auc_score(y_test, y_proba1)


# In[79]:


fpr1, tpr1, threshold1 = roc_curve(y_test, y_proba1)
rocauc1 = auc(fpr1, tpr1)


# In[80]:


plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest (area={rocauc:.4f})')
plt.plot(fpr1, tpr1, label=f'XGBoost  (area={rocauc1:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()


# In[81]:


evaluation = pd.DataFrame({'Model': [ 'Random Forest',
             'XGBoost Classifier'],
                           'F1 Score on training set': [f1_score_randfor,  f1_score_xgb],
                           'F1 using stratified 10-fold cross validation': [val_randfor, val_xgb]})
evaluation = evaluation.set_index('Model')


# In[82]:


evaluation

