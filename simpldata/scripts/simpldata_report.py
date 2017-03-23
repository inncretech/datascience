
# coding: utf-8

# In[14]:

get_ipython().run_cell_magic(u'HTML', u'', u"<h1>1. Abstract</h1>\n<h3>\n<p>In this report, we'll be exploring a dataset consisting of app interaction data from a popular online food delivery platform and using it to answer some business questions.</p>\n</h3>")


# In[15]:

import sys
import time
import warnings
import pylab as P
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from __future__ import division


# In[32]:

pd.set_option('display.float_format', lambda x: '%.0f' % x)
cols = ["id", "platform", "timestamp", "event_type"]
sv_cols = ["id", "platform", "timestamp", "event_type", "screenname"]
sns.set(style="darkgrid")
warnings.filterwarnings("ignore")


# In[5]:

#reading in all of the event data
data = pd.read_csv("/data.csv")[cols].sort_values(by = ['id', 'timestamp'])

#reading in all of 'Screen Visited' events' data
svdata = pd.read_csv("/svdata.csv")[sv_cols].sort_values(by = ['id', 'timestamp'])
svdata = svdata.loc[svdata['screenname'].notnull()]


# In[8]:

#Lets first see what our data looks like.
print data.head()
print "\n"
print svdata.head()


# In[5]:

get_ipython().run_cell_magic(u'HTML', u'', u'<h3>\n<p>Where do all these app interaction events come from? Lets find out. From the plot below we can see that most of the app interaction events are triggered on the android application of our e-commerce portal.</p> \n</h3>')


# In[7]:

plot = ''
plot = sns.countplot(x="platform", data=data)
sns.plt.show()


# In[53]:

#Number of unique users who logged in on each platform.
print dict(data[["id", "platform"]].drop_duplicates().groupby(by=['platform']).size())

#Number of unique users who made a transaction on each platform.
print dict(data.loc[data.event_type=="Charged"][["id", "platform"]].drop_duplicates().groupby(by=['platform']).size())


# In[16]:

get_ipython().run_cell_magic(u'HTML', u'', u'<h3>\n<p>Where are all the unique users coming from and how many of them actually make a purchase on our website? We noticed from the data that for a time period of two days (from 10-01-2016 to 10-03-2016), the vast majority of unique users (53,196) accessed the app on Web. Of these users, 4940 users had placed orders and were charged. The second most popular platform is Android, with a unique active user count of 30,884, of which 4001 users were charged for their orders.</p>\n</h3>')


# In[52]:

from collections import OrderedDict
active = data[["id", "platform"]].drop_duplicates().groupby(by=['platform']).size()
charged = data.loc[data.event_type=="Charged"][["id", "platform"]].drop_duplicates().groupby(by=['platform']).size()
sales = OrderedDict([ ('Platform', ['Android', 'iOS', 'Web']), ('Charged', [4001, 2487, 4940]), ('Active',  [30884, 11433, 53196])])
df = pd.DataFrame.from_dict(sales).sort_values(by = ['Active', 'Charged'], ascending = [False, False])
sns.set_color_codes("pastel")
ax = ''
ax = sns.barplot(x="Platform", y="Active", data=df, label="Active", color="b")
sns.set_color_codes("muted")
ax = sns.barplot(x="Platform", y="Charged", data=df, label="Charged", color="b")
ax.set(xlabel='Platform', ylabel='Number of Users')
ax.legend(ncol=2, loc="upper right", frameon=True)
sns.despine(left=True, bottom=True)
sns.plt.show()


# In[10]:

get_ipython().run_cell_magic(u'HTML', u'', u'<h3>\n<p>There are over 60 different types of events that can be tracked through the app. Which ones occur most frequently? We see that \u201cScreen Visited\u201d seems to be the most frequently occurring event on our app, followed by the event \u201cMenu Category Changed\u201d. We can also observe that over the course of two days less than 15000 transactions have occurred (the event \u201cCharged\u201d has a frequency of 14,417).</p>\n</h3>')


# In[15]:

plot = ''
plot = sns.countplot(x="event_type", data=data)
plot.set_xticklabels(plot.get_xticklabels(), rotation=-90)
sns.plt.show()


# In[16]:

categories = ['american', 'asian', 'chinese', 'continental', 'european', 'fusion', 
              'indian', 'italian', 'latin-american', 'mediterranean', 'mexican', 
              'middle-east', 'oriental', 'pan-asian', 'thai', 'universal']

for category in categories:    
    svdata.loc[svdata.screenname.str.startswith('/'+category), 'category'] = category    


# In[19]:

svdata.head()


# In[33]:

category_df = svdata.loc[svdata.category.notnull()]
category_df.loc[:, 'isveg'] = np.where(category_df['screenname'].str.contains('vegetarian'), 'Veg', 'Nonveg')


# In[22]:

category_df.head()


# In[17]:

get_ipython().run_cell_magic(u'HTML', u'', u'<h3>\n<p>Users often browse through food from different cuisines before making a final choice on what to purchase. Lets find out which cuisines are getting the most screen views. Over here, we observe that Continental with a count of 5620, is the food category with the highest number of user views. Latin-American dishes were the least popular (with a view count of only 1).<p>\n</h3>')


# In[30]:

plot = ''
plot = sns.countplot(x="category", data=category_df)
plot.set_xticklabels(plot.get_xticklabels(), rotation=-90)
sns.plt.show()


# In[12]:

get_ipython().run_cell_magic(u'HTML', u'', u'<h3>\n<p>What is the ratio of vegetarian food views to non vegetarian food views withing each cuisine? The plot below shows us that Pan-Asian seems to be the most popular choice among people who\u2019re interested in vegetarian food. And for non vegetarians, the most popular is Continental.</p>\n</h3>')


# In[36]:

plot = ''
category_count = category_df[["id", "category", "isveg"]].groupby(by = ['category', 'isveg']).count().reset_index().sort_values(by = ['category', 'id'], ascending=[True, False])
plot = sns.barplot(x = "id", y = "category", data = category_count, hue = "isveg")
plot.set(xlabel='Number of Views', ylabel='Food Category')
plt.title("#Total Views by Category")
sns.plt.show()


# In[58]:

def getplot(category):
    ax0=''
    category = category
    product_df = category_df[["id", "screenname", "category", "isveg"]].groupby(by = ['screenname', 'category', 'isveg']).count().sort_values(by = ['id'], ascending=False).reset_index()
    ax0 = sns.barplot(x = "id", y = "screenname", data = product_df.loc[product_df.category==category])
    ax0.set(xlabel='Number of Views', ylabel=category)
    return(ax0)


# In[13]:

get_ipython().run_cell_magic(u'HTML', u'', u'<h3>\n<p>In terms of screen views, which dishes are more popular for a given cuisine? Over here we drill down to the product level for two of the most popular categories, continental and pan-asian. We can observe that number of continental food items that are viewed far exceed that of pan-asian food items.</p>\n</h3>')


# In[60]:

getplot('continental')
plt.title("#Product Views by Category")
sns.plt.show()
print "\n"
getplot('pan-asian')
plt.title("#Product Views by Category")
sns.plt.show()

