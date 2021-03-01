# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 19:05:49 2021

@author: Ahamad Husssain, D

Carrier Inc. Data Science & Innovation Take Home Challenge
THE CHALLENGE: Zeta Disease Predictio
Note: the Code is Confidential, needs author prior approval  
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# Function to Get the current
# working directory 
def current_path():
    print("Current working directory before")
    print(os.getcwd())
    print()
current_path()
# Changing the CWD 
os.chdir('F:/Analytics_course/Python_practice/TakeHome_Carrier')
# Printing CWD after 
current_path()

data=pd.read_csv('2021-01-21_zeta-disease_training-data_dsi-take-home-challenge.csv',index_col=False)
print(len(data))
print(len(data.columns))
data.info()
print(data.describe)
df=data
df.shape

######Pre processing Steps/ Feature selection/DAta Visuvalization steps********

dupli_df=df[df.duplicated()]
print("no of duplicate rows:",dupli_df.shape)
dupli_df

df.sum(axis = 0, skipna = True) 
df['zeta_disease'].sum(axis = 0, skipna = True)
df.iloc[:,-1:].sum()

df=df.drop_duplicates()
df.info()

plt.hist(df['zeta_disease'])
plt.xlabel('Zeta_Desease')
plt.ylabel('Frequency')
plt.xticks([0,1])
plt.title("Zeta Disease Distribution")
plt.show()

df=df.dropna()
print(len(df))
df.isnull().sum() 

#histogram of all variables
df.hist(figsize=(10,10))
plt.show()

df1=df.iloc[:,0:8]
df1.info()
df1.describe()

dfo=[]
dfo=df.iloc[:,8:9]
dfo.info()
dfo.describe()

corrmat=df1.corr()
top_corr_features=corrmat.index
print(corrmat)
plt.figure(figsize=(10,10))
plt.title('Variables Correlation map')
g=sns.heatmap(df1[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#sns.pairplot(df1)
sns.pairplot(df, hue = 'zeta_disease')

corrmat['age'].sort_values(ascending=False)

from scipy import stats
import pylab
df2=df1
summary1=df2.describe()
print(summary1)

for i in range(len(df1.columns)):
    mean1=summary1.iloc[1,i]
    std1=summary1.iloc[2,i]
    df2.iloc[:,i:(i+1)]=(df1.iloc[:,i:(i+1)]-mean1)/std1

print(df2.describe())
plt.figure(figsize=(10,10))
plotray= df2.values
#boxplot(plotray)
#plot.xticks(range(1,9),abalone.columns[1:9])
sns.boxplot(data=plotray)
plt.xlabel('Varialbes in the order of Data Frame')
plt.ylabel('Standard Deviations')
plt.title("Standardised Varialbes Box Plots")
plt.show()

z=np.abs(stats.zscore(plotray))
print(z)
q=np.amax(z)
print(q)

df2['zeta_disease']=dfo
df3 = df2[np.abs(z < 6).all(axis=1)]


df2.info()
df3.describe()
df2.describe()
X=df3.iloc[:,0:8]
Y=df3.iloc[:,8:9]
from sklearn import*
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf=ExtraTreesClassifier()
clf.fit(X, Y)
clf.feature_importances_ 
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
X=df3.iloc[:,0:8]
y=df3.iloc[:,8:9]
# Build a forest and compute the impurity-based feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

################Model Buildings*****************
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.linear_model import *

# Creating an empty Dataframe with column names only
AlSumm = pd.DataFrame(columns=['Model','ModelParameter','TN','FP','FN','TP','Accuracy','F1 Score','Precesion','Recall','FNR'])


#Logistic Regression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


cv = KFold(n_splits=5, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
y_train_pred=cross_val_predict(model,X_train,y_train.values.ravel(),cv=cv)
#lrcm=confusion_matrix(y_train,y_train_pred)
scores = cross_val_score(model, X_train,y_train.values.ravel(), scoring='accuracy', cv=cv)
# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
y_pred=cross_val_predict(model,X_test,y_test.values.ravel(),cv=cv)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'LogisticRegression','ModelParameter':0,'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)


#apply the below code to store all confusion parameters


#print(metrics.classification_report(y_test, y_pred, *))
#LDA
# grid search solver for lda
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define dataset
model = LinearDiscriminantAnalysis()
# define model evaluation method

# define grid
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv)
# perform the search
results = search.fit(X_train, y_train)
y_pred=search.predict(X_test)
lrcm=confusion_matrix(y_test,y_pred)
# summarize #print('Mean Accuracy: %.3f' % results.best_score_)
#print('Config: %s' % results.best_params_)
AlSumm= AlSumm.append({'Model':'LDA','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)




#QDA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train,y_train.values.ravel())
y_pred=(qda.predict(X_test))

qdacm=confusion_matrix(y_pred,y_test)


# create model
model = QuadraticDiscriminantAnalysis()
# evaluate model
y_train_pred=cross_val_predict(model,X_train,y_train.values.ravel(),cv=cv)
#lrcm=confusion_matrix(y_train,y_train_pred)
scores = cross_val_score(model, X_train,y_train.values.ravel(), scoring='accuracy', cv=cv)
# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
y_pred=cross_val_predict(model,X_test,y_test.values.ravel(),cv=cv)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'QDA','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)


#Support Vector Classifier
from sklearn.svm import *
clf1=SVC(kernel='linear',coef0=1,C=5)


model = clf1
# evaluate model
y_train_pred=cross_val_predict(model,X_train,y_train.values.ravel(),cv=cv)
#lrcm=confusion_matrix(y_train,y_train_pred)
scores = cross_val_score(model, X_train,y_train.values.ravel(), scoring='accuracy', cv=cv)
# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
y_pred=cross_val_predict(model,X_test,y_test.values.ravel(),cv=cv)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'SVC_linear','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)


clf1=SVC(kernel='rbf',gamma=0.01)
model = clf1
# evaluate model
y_train_pred=cross_val_predict(model,X_train,y_train.values.ravel(),cv=cv)
#lrcm=confusion_matrix(y_train,y_train_pred)
scores = cross_val_score(model, X_train,y_train.values.ravel(), scoring='accuracy', cv=cv)
# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
y_pred=cross_val_predict(model,X_test,y_test.values.ravel(),cv=cv)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'SVC_rbf','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=15, random_state=0)
model = rf
# evaluate model
y_train_pred=cross_val_predict(model,X_train,y_train.values.ravel(),cv=cv)
#lrcm=confusion_matrix(y_train,y_train_pred)
scores = cross_val_score(model, X_train,y_train.values.ravel(), scoring='accuracy', cv=cv)
# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
y_pred=cross_val_predict(model,X_test,y_test.values.ravel(),cv=cv)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'Random Forest Classifier','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)

#ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=500, random_state=0)

# evaluate model
y_train_pred=cross_val_predict(model,X_train,y_train.values.ravel(),cv=cv)
#lrcm=confusion_matrix(y_train,y_train_pred)
scores = cross_val_score(model, X_train,y_train.values.ravel(), scoring='accuracy', cv=cv)
# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
y_pred=cross_val_predict(model,X_test,y_test.values.ravel(),cv=cv)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'ExtraTreesClassifier','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)


#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,100,5):
    model = KNeighborsClassifier(n_neighbors=i)
    # evaluate model
    y_train_pred=cross_val_predict(model,X_train,y_train.values.ravel(),cv=cv)
    scores = cross_val_score(model, X_train,y_train.values.ravel(), scoring='accuracy', cv=cv)
    y_pred=cross_val_predict(model,X_test,y_test.values.ravel(),cv=cv)
    lrcm=confusion_matrix(y_test,y_pred)
    AlSumm= AlSumm.append({'Model':'KNN','ModelParameter':i, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)



#*******SDG Classifier
import sklearn

from sklearn.linear_model import SGDClassifier
model=SGDClassifier(random_state=42)
y_train_pred=cross_val_predict(model,X_train,y_train.values.ravel(),cv=cv)
#lrcm=confusion_matrix(y_train,y_train_pred)
#scores = cross_val_score(model, X_train,y_train.values.ravel(), scoring='accuracy', cv=cv)
# report performance
#print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
y_pred=cross_val_predict(model,X_test,y_test.values.ravel(),cv=cv)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'SGD Classifier','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)

##################
#DNN
import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
#Swish
model.add(Dense(8, activation='swish', input_shape=(8,)))

model.add(Dense(8, activation='swish'))

model.add(Dense(8, activation='swish'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, y_train,epochs=5, batch_size=1, verbose=1)

y_pred = model.predict_classes(X_test)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'DNN-Swish','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)




mode2 = Sequential()
#Swish
mode2.add(Dense(8, activation='relu', input_shape=(8,)))

mode2.add(Dense(8, activation='relu'))

mode2.add(Dense(8, activation='relu'))

mode2.add(Dense(1, activation='sigmoid'))

mode2.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
                   
mode2.fit(X_train, y_train,epochs=5, batch_size=1, verbose=1)


y_pred = mode2.predict_classes(X_test)

lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'DNN-ReLU','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)


###********* Bagging, Out of Bag, Ada Boosting 
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier( DecisionTreeClassifier(),max_samples=100, bootstrap=True)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'Bagging','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)


bag_clf = BaggingClassifier(DecisionTreeClassifier(),bootstrap=True, oob_score=True)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'OOB','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)


from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500,
                             algorithm="SAMME.R", learning_rate=0.4 )
ada_clf.fit(X_train, y_train)
lrcm=confusion_matrix(y_test,y_pred)
AlSumm= AlSumm.append({'Model':'AdaBoost','ModelParameter':0, 'TN':lrcm[0][0],'FP':lrcm[0][1],'FN':lrcm[1][0],'TP':lrcm[1][1],
                      'Accuracy':accuracy_score(y_test, y_pred),'F1 Score':f1_score(y_test, y_pred),
                      'Precesion':precision_score(y_test, y_pred),'Recall':recall_score(y_test, y_pred),
                      'FNR':(1-recall_score(y_test, y_pred))}, ignore_index=True)

#### ****** Selecting Best Model *********


AlSumm['score']=AlSumm['Accuracy']+AlSumm['F1 Score']-AlSumm['FNR']
maxs=max(AlSumm['score'])
for i in range(len(AlSumm)):
    if (AlSumm['score'][i])==maxs:
        poli=i
print(poli)
AlSumm.iloc[poli,:]

######FINAL Testing With Data********
tdf=pd.read_csv('2021-01-21_zeta-disease_prediction-data_dsi-take-home-challenge.csv',index_col=False)

tdf2=tdf.iloc[:,0:8]
tdf2.info()
tdf.head()
for i in range(len(tdf.columns)-1):
    mean1=summary1.iloc[1,i]
    std1=summary1.iloc[2,i]
    tdf2.iloc[:,i:(i+1)]=(tdf.iloc[:,i:(i+1)]-mean1)/std1
tdf2.head()

#Select the Final Model based on confusion matrix and predict using that final Model
tdf.iloc[:,8:9]=(mode2.predict_classes(tdf2))
tdf.head()

tdf.to_csv('zeta-disease_predictions_AhamadHussain.csv') 
AlSumm.to_csv('All_Models_Summary_AhamadHussain.csv')