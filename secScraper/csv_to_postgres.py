#!/usr/bin/env python
# coding: utf-8

# In[1]:


from secScraper import *
import sys

if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    raise Exception("Must be using Python >= 3.6 due to reliance on ordered default dict.")
else:
    version = "[INFO] Running python {}.{}.{}".format(*sys.version_info[:3])
    if display.run_from_ipython():
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
        get_ipython().run_line_magic('matplotlib', 'notebook')
        version += " for ipython" if display.run_from_ipython() else ""
    print("[INFO] Running python {}.{}.{} (>= python 3.6)".format(*sys.version_info[:3]))


# In[2]:


import csv
import psycopg2
from tqdm import tqdm


# # Read the settings in postgres

# In[3]:


connector = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="1")


# In[4]:


s = postgres.retrieve_settings(connector)


# # Dump all the stock data to postgres

# ## Load the time_range of the stock data of interest
# The main CSV file will be read and any data point outside s['time_range'] will be ignored.

# In[5]:


# Load all stock prices - takes a while but one off
stock_data = pre_processing.load_stock_data(s, penny_limit=75000000)


# ## Dump to CSV first
# Why? Because somehow dumping to CSV then importing in postgres from CSV is crazy fast.

# In[6]:


header_stock_data = (('TICKER', 'text'), ('DATE', 'date'),
                    ('ASK', 'float'), ('MARKET_CAP', 'bigint'))
with open(s['path_filtered_stock_data'], 'w') as f:
    out = csv.writer(f, delimiter=';')
    header = ['IDX'] + [name[0] for name in header_stock_data]
    out.writerow(header)
    idx = 0
    for ticker in tqdm(stock_data):
        for ts in stock_data[ticker]:
            #print(ts)
            out.writerow((idx, ticker, ts, stock_data[ticker][ts][0], int(stock_data[ticker][ts][1])))
            idx += 1


# ## copy_from csv to postgres

# In[7]:


path = s['path_filtered_stock_data']
postgres.csv_to_postgres(connector, 'stock_data', header_stock_data, path)


# # Optional: retrieve the data

# In[8]:


sd = postgres.retrieve_all_stock_data(connector, 'stock_data')


# In[9]:


sd.keys()


# # Dump all the index data to postgres
# Essentially the same with the index_data

# In[10]:


index_data = pre_processing.load_index_data(s)
print("[INFO] Loaded the following index data:", list(index_data.keys()))


# In[11]:


index_data['IXIC']


# In[12]:


header_index_data = (('INDEX', 'text'), ('DATE', 'date'), ('ASK', 'float'))
with open(s['path_filtered_index_data'], 'w') as f:
    out = csv.writer(f, delimiter=';')
    header = ['IDX'] + [name[0] for name in header_index_data]
    out.writerow(header)
    idx = 0
    for ticker in tqdm(index_data):
        for ts in index_data[ticker]:
            #print(ts)
            out.writerow((idx, ticker, ts, index_data[ticker][ts]))
            idx += 1


# In[13]:


path = s['path_filtered_index_data']
postgres.csv_to_postgres(connector, 'index_data', header_index_data, path)


# In[14]:


ind = postgres.retrieve_all_stock_data(connector, 'index_data')


# In[15]:


ind.keys()


# # Dump the lookup table to postgres
# Another database that is unfrequently changing, so we load it in postgres

# In[16]:


lookup = pre_processing.load_lookup(s)
print("[INFO] Loaded {:,} CIK/Tickers correspondances.".format(len(lookup)))


# In[17]:


header_lookup = (('CIK', 'integer'), ('TICKER', 'text'))
with open(s['path_filtered_lookup'], 'w') as f:
    out = csv.writer(f, delimiter=';')
    header = ['IDX'] + [name[0] for name in header_lookup]
    out.writerow(header)
    for idx, item in enumerate(tqdm(lookup.items())):
        out.writerow((idx, *item))


# In[18]:


path = s['path_filtered_lookup']
postgres.csv_to_postgres(connector, 'lookup', header_lookup, path)


# In[19]:


lookup, reverse_lookup = postgres.retrieve_lookup(connector)


# # Dump LM dictionary into postgres
# Same, the dictionary is updated infrequently so it should go into postgres. But that will be for later.
# 
# TO DO: Import in postgres

# In[ ]:





# In[ ]:




