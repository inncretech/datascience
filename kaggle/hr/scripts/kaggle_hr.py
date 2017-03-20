
# coding: utf-8

# In[62]:

# We'll start by loading in all the required packages.

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from __future__ import division

from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
#sns.set(style="white", context="talk")
sns.set(style="whitegrid")

from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

print "Done!"


# In[156]:

# Some utility functions that will help us perform machine learning

def classify(grid, X_train, y_train, X_test, y_test):
    results = dict()
    
    #Training the model using grid search & cross validation
    start_time = time.time()
    grid.fit(X_train, y_train)
    end_time = time.time() - start_time
    results['training_time'] = end_time
    
    
    #Testing the model on the held out test data set
    start_time = time.time()
    grid_test = grid.predict(X_test)
    end_time = time.time() - start_time
    results['testing_time'] = end_time
    
    results['accuracy'] = metrics.accuracy_score(y_test, grid_test)
    results['report'] = metrics.classification_report(y_test, grid_test)
    results['matrix'] = metrics.confusion_matrix(y_test, grid_test)
    
    results['grid'] = grid
    results['grid_test'] = grid_test
    
    return(results)

def get_feature_importances(grid, X_test):
    #Returns a dataframe with feature importance info
    ls = list()
    for a,b in enumerate(grid.best_estimator_.feature_importances_):
        ls.append({'feature':X_test.columns[a], 'importance':b})
    feature_importances = pd.DataFrame(ls).sort_values(by = ['importance'], ascending=False)
    return(feature_importances)

def plot_feature_importances(feature_importances):
    ax = sns.stripplot(x = "importance", y = "feature", data = feature_importances)
    ax.set(xlabel = 'Importance', ylabel='Feature')
    return(ax)

def plot_validation_curve(model, X, y, items):
    train_scores, test_scores = validation_curve(model,
                                                 X,
                                                 y,
                                                 param_name=items['param_name'],
                                                 param_range=items['param_range'],
                                                 cv=10, 
                                                 scoring=items['scoring'],
                                                 n_jobs=-1)
    
    train_score_means = np.mean(train_scores, axis=1)
    test_score_means = np.mean(test_scores, axis=1)

    plt.title(items['title'])
    plt.xlabel(items['param_name'])
    plt.ylabel(items['scoring'])
    plt.ylim(0.0, 1.0)
    plt.plot(items['param_range'], train_score_means, color="darkorange", label="Training Score")
    plt.plot(items['param_range'], test_score_means, color="navy", label="Testing Score")
    plt.legend(loc="best")
    
    return(plt)


def plot_learning_curve(model, X, y, items):
    train_sizes_abs, train_scores, test_scores = learning_curve(model,
                                                                X,
                                                                y,
                                                                train_sizes=items['train_sizes'],
                                                                cv=items['cv'],
                                                                n_jobs=1)
    
    train_score_means = np.mean(train_scores, axis=1)
    test_score_means = np.mean(test_scores, axis=1)

    plt.title(items['title'])
    plt.xlabel('train_sizes')
    plt.ylabel(items['scoring'])
    plt.ylim(0.0, 1.0)
    plt.plot(train_sizes_abs, train_score_means, color="darkorange", label="Training Score")
    plt.plot(train_sizes_abs, test_score_means, color="navy", label="Testing Score")
    plt.legend(loc="best")
    
    return(plt)

def plot_roc_curve(items):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(items['y_test'], items['y_pred'])
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.title(items['title'])
    plt.plot(false_positive_rate, true_positive_rate, "b", label='AUC = %0.2f'% roc_auc)
    plt.plot([0,1],[0,1],'r--')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return(plt)

def print_results(results):
    print "Best Estimator: \n\n%s"%(results['grid'].best_estimator_)
    print "\nTraining Time: \n\n%s seconds"%(results['training_time'])
    print "\nTesting Time: \n\n%s seconds"%(results['testing_time'])
    print "\nAccuracy: \n\n%s"%(results['accuracy'])
    print "\nConfusion Matrix: \n\n%s"%(results['matrix'])
    print "\nClassification Report: \n\n%s"%(results['report'])


# In[33]:

#Reading in the data
df = pd.read_csv("/Users/nandu/desktop/WORK/kaggle/hr/data/HR_comma_sep.csv")

print df.shape

print df.info()


# In[17]:

# We see that the column "sales" represents the department to which each employee belongs. 
print pd.unique(df.sales.ravel())


# In[34]:

# We'll rename the "sales" column to "department"

df.rename(columns={"sales":"department"}, inplace=True)


# In[100]:

#Lets look at the distribution of the attribute "department".

ax = None
ax = sns.countplot(y="department", data=df)
sns.plt.show()

# Employees belonging to the Sales department constitute the majority.


# In[119]:

# Lets see what the average monthly hours look like for each salary group.

ax = None
ax = sns.factorplot(data=df,
                    x="salary",
                    y="average_montly_hours",
                    col="left",
                    kind="bar")
(ax
 .set_axis_labels("Salary", "Mean Average Monthly Hour")
 .set_xticklabels(["Low", "Medium", "High"])
 .despine(left=True, right=True))
sns.plt.show()


# In[99]:

#Lets see how the satisfaction level of the employees varies across different salary groups.

ax = None
ax = sns.factorplot(data=df,
                    x="salary",
                    y="satisfaction_level",
                    col="left")
(ax
 .set_axis_labels("Salary", "Satisfaction Level")
 .set_xticklabels(["Low", "Medium", "High"])
 .despine(left=True, right=True))
sns.plt.show()

#It appears that when employees left, they reported lower satisfaction levels, ranging from a little over 0.37 to 0.46. 
#Employees who've stayed possess relatively higher satisfaction levels.


# In[116]:

#Lets see how the satisfaction level of the employees varies with respect to the number of projects assigned.

ax = None
ax = sns.factorplot(data=df,
                    x="number_project",
                    y="satisfaction_level",
                    col="salary")
(ax
 .set_axis_labels("Number of Projects", "Satisfaction Level")
 .set_xticklabels([1, 2, 3, 4, 5, 6, 7])
 .despine(left=True, right=True))
sns.plt.show()

#It appears that the people who are most satisfied with their jobs are assigned to around 2 to 4 projects.
#Employees who are assigned to just 1 or even more than 4 projects have relatively lower satisfaction levels.


# In[26]:

# We see that the columns "salary" and "department" are both categorical 
# and will need to be encoded before we go on to do any machine learning.

print pd.unique(df.salary.ravel())
print pd.unique(df.department.ravel())


# In[27]:

#One hot encoding - Transforming all the categorical variables to their binary representations
salary = pd.get_dummies(df['salary'], drop_first=False)
department = pd.get_dummies(df['department'], drop_first=False)
df.drop(['salary', 'department'], axis=1, inplace=True)
df = pd.concat([df, salary, department], axis=1)

print df.info()


# In[102]:

#Lets look at how our class label "left" is distributed.

ax = None
ax = sns.countplot(x="left", data=df)
sns.plt.show()

# We see that a vast majority of the data points belong to class 0, meaning 
# that those employees have not left the company. The remaining belong to class 1
# which indicates that they have left.


# In[32]:

#Splitting the dataset into two parts, train set and test set, using stratified sampling.
#We use stratified sampling as the dataset is imbalanced and skewed towards class 0.

train, test = train_test_split(df, test_size = 0.3, random_state = 5, stratify = df['left'])
X_train = train.ix[:, df.columns.difference(['left'])]
y_train = train.ix[:, 'left']
X_test = test.ix[:, df.columns.difference(['left'])]
y_test = test.ix[:, ['left']]

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape


# In[120]:

#Random Forests classification
n = range(1, 101)
param_grid = dict(n_estimators=n)

random_forests = RandomForestClassifier(random_state=5)
grid = GridSearchCV(random_forests, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

results1 = classify(grid, X_train, y_train, X_test, y_test)

#Visualizing the importance of the features
#The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. 
#It is also known as the Gini importance.

feature_importances = get_feature_importances(results1['grid'], X_test)

ax = None
ax = plot_feature_importances(feature_importances)
sns.plt.show()


# In[121]:

print_results(results1)


# In[122]:

#Our Random Forests model gives us the importance of each feature.
#Lets drop the ones with relatively lower scores, generate new train/test sets, 
#and retrain the RF model to see if there are any changes in the accuracy.

top_features = feature_importances[:5]['feature'].tolist() + ['left']
new_df = df[top_features]

train, test = train_test_split(new_df, test_size = 0.3, random_state = 5, stratify = new_df['left'])
X_train = train.ix[:, new_df.columns.difference(['left'])]
y_train = train.ix[:, 'left']
X_test = test.ix[:, new_df.columns.difference(['left'])]
y_test = test.ix[:, ['left']]


# In[138]:

#Random Forests classification with reduced features
n = range(1, 101)
param_grid = dict(n_estimators=n)

random_forests = RandomForestClassifier(random_state=5)
grid = GridSearchCV(random_forests, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

results1 = classify(grid, X_train, y_train, X_test, y_test)

feature_importances = get_feature_importances(results1['grid'], X_test)

ax = None
ax = plot_feature_importances(feature_importances)
sns.plt.show()


# In[139]:

print_results(results1)

#Although the model takes quite a while to train, it produces pretty impressive class precision and recall scores.


# In[149]:




# In[157]:

#Lets take a look at what the ROC curve looks like for our Random Forests model.
#The ROC curve is a plot of the False Positive Rate vs the True Positive Rate of a binary classification model.
#AUC, which stands for Area Under the Curve, tells us what the value of the area under the curve is. 
#The closer AUC is to 1.0, the better is our model at performing classifications.

items = {"title":"Receiver Operating Characteristic (Random Forests)", 
         "y_test":y_test, 
         "y_pred":results1['grid_test']}

plt = plot_roc_curve(items)
plt.show()


# In[ ]:

#Validation curves help us visualize how the performance of our model during training and testing 
#varies with changes in it hyperparameter values. 


# In[125]:

#Random Forests Validation Curve 

random_forests_classifier = RandomForestClassifier(random_state=5)
n_estimators = range(1, 101)

items = {'title':'Random Forests Validation Curve',
         'param_range':n_estimators,
         'param_name':'n_estimators',
         'scoring':'accuracy'}

plt = plot_validation_curve(random_forests_classifier, X_train, y_train, items)
plt.show()


# In[ ]:

#Learning curves help us visualize how the performance of our model during training and testing 
#varies with changes in training set size. 


# In[137]:

#Random Forests Learning Curve

random_forests_classifier = RandomForestClassifier(random_state=5, n_estimators=10)
train_sizes = np.linspace(.05, 1.0, 10)

items = {'title':'Random Forests Learning Curve',
         'train_sizes':train_sizes,
         'cv':10,
         'scoring':'accuracy'}

plt = plot_learning_curve(random_forests_classifier, X_train, y_train, items)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[155]:

#SVM Classification
kernel = ['sigmoid', 'rbf']
param_grid = dict(kernel=kernel)

support_vector_classifier = svm.SVC(random_state=5)
grid = GridSearchCV(support_vector_classifier, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

results2 = classify(grid, X_train, y_train, X_test, y_test)

print_results(results2)


# In[159]:

items = {"title":"Receiver Operating Characteristic (SVM)", 
         "y_test":y_test, 
         "y_pred":results2['grid_test']}

plt = plot_roc_curve(items)
plt.show()


# In[132]:

#SVM Validation Curve
support_vector_classifier = svm.SVC(random_state=5, kernel="rbf")
gamma = np.logspace(-6, -1, 5)

items = {'title':'SVM Validation Curve',
         'param_range':gamma,
         'param_name':'gamma',
         'scoring':'accuracy'}

plt = plot_validation_curve(support_vector_classifier, X_train, y_train, items)
plt.show()


# In[133]:

#SVM Learning Curve

support_vector_classifier = svm.SVC(random_state=5, kernel="rbf")
train_sizes = np.linspace(.05, 1.0, 10)

items = {'title':'SVM Learning Curve',
         'train_sizes':train_sizes,
         'cv':10,
         'scoring':'accuracy'}

plt = plot_learning_curve(support_vector_classifier, X_train, y_train, items)
plt.show()


# In[ ]:




# In[ ]:




# In[161]:

#KNN Classification
p = range(1, 3)
n_neighbors = range(1, 101)
param_grid = dict(p=p, n_neighbors=n_neighbors)

knn_classifier = KNeighborsClassifier()
grid = GridSearchCV(knn_classifier, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

results3 = classify(grid, X_train, y_train, X_test, y_test)

print_results(results3)


# In[162]:

items = {"title":"Receiver Operating Characteristic (K Nearest Neighbors)", 
         "y_test":y_test, 
         "y_pred":results3['grid_test']}

plt = plot_roc_curve(items)
plt.show()


# In[161]:




# In[134]:

#KNN Validation curve

knn_classifier = KNeighborsClassifier()
n_neighbors = range(1, 50)

items = {'title':'KNN Validation Curve',
         'param_range':n_neighbors,
         'param_name':'n_neighbors',
         'scoring':'accuracy'}

plt = plot_validation_curve(knn_classifier, X_train, y_train, items)
plt.show()


# In[135]:

#KNN Learning curve

#train_sizes = [np.round(percent*X_train.shape[0]).astype(int) for percent in np.arange(0.1, 1.1, 0.1)]
#train_sizes = [percent for percent in np.arange(0.1, 1.1, 0.1)]

knn_classifier = KNeighborsClassifier(n_neighbors=2)
train_sizes = np.linspace(.05, 1.0, 10)

items = {'title':'KNN Learning Curve',
         'train_sizes':train_sizes,
         'cv':10,
         'scoring':'accuracy'}

plt = plot_learning_curve(knn_classifier, X_train, y_train, items)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[163]:

#Decision Tree Classifier

max_depth = range(10, 31)
param_grid = dict(max_depth=max_depth)
decision_tree_classifier = DecisionTreeClassifier(random_state=5)
grid = GridSearchCV(decision_tree_classifier, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
results4 = classify(grid, X_train, y_train, X_test, y_test)
print_results(results4)


# In[165]:

items = {"title":"Receiver Operating Characteristic (Decision Tree)", 
         "y_test":y_test, 
         "y_pred":results3['grid_test']}

plt = plot_roc_curve(items)
plt.show()


# In[164]:

#Printing out the decision tree
my_tree = results4['grid'].best_estimator_
dot_data = export_graphviz(my_tree, out_file=None, feature_names=X_test.columns, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("/users/nandu/Desktop/tree.pdf")
Image(graph.create_png())


# In[187]:

#Lets visualise the ROC curves of all our models together to find out which one performed the best.

colors = ['cyan', 'indigo','blue', 'darkorange']
titles = ["Random Forests", "Support Vector Machines", "K Nearest Neighbors", "Decision Trees"]
results = [results1, results2, results3, results4]
models = zip(colors, results, titles)

for model in models:
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, model[1]['grid_test'])
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, model[0], label=model[2]+'/AUC = %0.2f'% roc_auc)

plt.plot([0,1],[0,1],'r--')
plt.title("Receiver Operating Characteristic")
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

