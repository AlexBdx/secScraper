#!/usr/bin/env python
# coding: utf-8

# # Configuration

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


# ## Packages to import

# In[2]:


import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()

import os
import csv
from datetime import datetime
import re
from tqdm import tqdm
import multiprocessing as mp
from collections import OrderedDict
import time
import pandas as pd
import argparse
import psycopg2
import ast
import copy

# Spark
# import findspark
# findspark.init('/home/alex/spark-2.4.4-bin-hadoop2.7')
import pyspark


# ### Set the nb of processes to use based on cmd line arguments/setting

# In[3]:


if display.run_from_ipython():
    nb_processes_requested = mp.cpu_count()  # From IPython, fixed setting
    # nb_processes_requested = 1 # From IPython, fixed setting
else:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--processes", type=int, default=mp.cpu_count(), help="Number of processes launched to process the reports.")
    args = vars(ap.parse_args())
    nb_processes_requested = args["processes"]
    if not 1 <= nb_processes_requested <= mp.cpu_count():
        raise ValueError('[ERROR] Number of processes requested is incorrect.                         \n{} CPUs are available on this machine, please select a number of processes between 1 and {}'
                         .format(mp.cpu_count()))


# ## Settings dictionary

# In[4]:


home = os.path.expanduser("~")
_s = {
    'path_stage_1_data': os.path.join(home, 'Desktop/filtered_text_data/nd_data/'),
    'path_stock_database': os.path.join(home, 'Desktop/Insight project/Database/Ticker_stock_price.csv'),
    'path_filtered_stock_data': os.path.join(home, 'Desktop/Insight project/Database/filtered_stock_data.csv'),
    'path_stock_indexes': os.path.join(home, 'Desktop/Insight project/Database/Indexes/'),
    'path_filtered_index_data': os.path.join(home, 'Desktop/Insight project/Database/Indexes/filtered_index_data.csv'),
    'path_lookup': os.path.join(home, 'Desktop/Insight project/Database/lookup.csv'),
    'path_filtered_lookup': os.path.join(home, 'Desktop/Insight project/Database/filtered_lookup.csv'),
    'path_master_dictionary': os.path.join(home, 'Desktop/Insight project/Database/LoughranMcDonald_MasterDictionary_2018.csv'),
    'path_dump_crsp': os.path.join(home, 'Desktop/Insight project/Database/dump_crsp_merged.txt'),
    'path_output_folder': os.path.join(home, 'Desktop/Insight project/Outputs'),
    'path_dump_cik_scores': os.path.join(home, 'Desktop/Insight project/Outputs/dump_cik_scores.csv'),
    'path_dump_pf_values': os.path.join(home, 'Desktop/Insight project/Outputs/dump_pf_values.csv'),
    'path_dump_master_dict': os.path.join(home, 'Desktop/Insight project/Outputs/dump_master_dict.csv'),
    'metrics': ['diff_jaccard', 'diff_sk_cosine_tf_idf', 'diff_gfg_editDistDP'],
    'stop_words': False,
    'lemmatize': False,
    'differentiation_mode': 'quarterly',
    'pf_balancing': 'unbalanced',
    'time_range': [(2012, 1), (2018, 4)],
    'bin_count': 5,
    'tax_rate': 0,
    'histogram_date_span_ratio': 0.5,
    'report_type': ['10-K', '10-Q'],
    'sections_to_parse_10k': [],
    'sections_to_parse_10q': [],
    'type_daily_price': 'closing'
}


# In[5]:


_s['pf_init_value'] = 100.0  # In points
_s['epsilon'] = 0.001  # Rounding error
# Calculated settings
_s['list_qtr'] = qtrs.create_qtr_list(_s['time_range'])

if _s['bin_count'] == 5:
    _s['bin_labels'] = ['Q'+str(n) for n in range(1, _s['bin_count']+1)]
elif _s['bin_count'] == 10:
    _s['bin_labels'] = ['D'+str(n) for n in range(1, _s['bin_count']+1)]
else:
    raise ValueError('[ERROR] This type of bin has not been implemented yet.')

# Create diff metrics and sing metrics
_s['diff_metrics'] = [m for m in _s['metrics'] if m[:4] == 'diff']
_s['sing_metrics'] = [m for m in _s['metrics'] if m[:4] == 'sing']
# Reports considered to calculate the differences
if _s['differentiation_mode'] == 'quarterly':
    _s['lag'] = 1
    _s['sections_to_parse_10k'] = ['1a', '3', '7', '7a', '9a']
    _s['sections_to_parse_10q'] = ['_i_2', '_i_3', '_i_4', 'ii_1', 'ii_1a']
elif _s['differentiation_mode'] == 'yearly':
    _s['lag'] = 4
    _s['sections_to_parse_10k'] = ['1a', '3', '7', '7a', '9a']
    _s['sections_to_parse_10q'] = ['_i_2', '_i_3', '_i_4', 'ii_1', 'ii_1a']

_s['common_quarterly_sections'] = {
        '10-K': ['1a', '3', '7', '7a', '9a'],
        '10-Q': ['ii_1a', 'ii_1', '_i_2', '_i_3', '_i_4']
}  # Exhibits are not taken into account
"""_s['common_yearly_sections'] = {
    '10-K': ['1', '1a', '1b', '2', '3', '4', '5', '6', '7', '7a', '8', '9', '9a', '9b', '10', '11', '12', '13', '14', '15'],
    '10-Q': ['_i_1', '_i_2', '_i_3', '_i_4', 'ii_1', 'ii_1a', 'ii_2', 'ii_3', 'ii_4', 'ii_5', 'ii_6']
}"""
_s['common_yearly_sections'] = {
    '10-K': ['7'],
    '10-Q': ['_i_2']
}  # Take into account 


# In[6]:


# Transfer s to a read only dict
read_only_dict = pre_processing.ReadOnlyDict()
for key in _s:  # Brute force copy
    read_only_dict[key] = _s[key]
s = read_only_dict  # Copy back
s.set_read_state(read_only=True)  # Set as read only


# # Load external tables

# In[7]:


connector = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="1")


# In[ ]:


postgres.settings_to_postgres(connector, s)


# ## Extract the list of CIK for which we have complete data

# The main problem in our case is that we have 3 different database to play with:
# 1. The SEC provides information based on the CIK of the entity
# 2. Given that the CIK is used by no one else, we use a lookup table to transform that into tickers. But we do not have all the correspondances, so the list of useful CIK is shrunk.
# 3. Finally, we only have stock prices for so many tickers. So that shrinks the CIK list even further.
# 
# We end up with a reduced list of CIK that we can play with.

# ### Load the sentiment analysis dictionary

# In[ ]:


lm_dictionary = Load_MasterDictionary.load_masterdictionary(s['path_master_dictionary'], True)


# ### Find all the unique CIK from the SEC filings

# In[ ]:


cik_path = pre_processing.load_cik_path(s)


# ### Get the largest {CIK: ticker} possible given our lookup table

# In[ ]:


lookup, reverse_lookup = postgres.retrieve_lookup(connector)
print("[INFO] Loaded {:,} CIK/Tickers correspondances.".format(len(lookup)))


# In[ ]:


cik_path, lookup = pre_processing.intersection_sec_lookup(cik_path, lookup)
print("[INFO] Intersected SEC & lookup.")
print("cik_path: {:,} CIK | lookup: {:,} CIK"
      .format(len(cik_path), len(lookup)))


# ### Load stock data and drop all CIKs for which we don't have data

# In[ ]:


# Load all stock prices
stock_data = postgres.retrieve_all_stock_data(connector, 'stock_data')


# In[ ]:


lookup, stock_data = pre_processing.intersection_lookup_stock(lookup, stock_data)
print("[INFO] Intersected lookup & stock data.")
print("lookup: {:,} tickers | stock_data: {:,} tickers"
      .format(len(lookup.values()), len(stock_data)))


# ### Load stock indexes - will serve as benchmark later on

# In[ ]:


index_data = postgres.retrieve_all_stock_data(connector, 'index_data')
print("[INFO] Loaded the following index data:", list(index_data.keys()))


# ## Back propagate these intersection all the way to cik_path

# Technically, we have just done it for lookup. So we only need to re-run an intersection for lookup and sec.

# In[ ]:


cik_path, lookup = pre_processing.intersection_sec_lookup(cik_path, lookup)
print("[INFO] Intersected SEC & lookup.")
print("cik_path: {:,} CIK | lookup: {:,} CIK"
      .format(len(cik_path), len(lookup)))


# ## Sanity check

# At this point, cik_path and lookup should have the same number of keys as the CIK is unique in the path database.
# 
# However, multiple CIK can redirect to the same ticker if the company changed its ticker over time. That should be a very limited amount of cases though.

# In[ ]:


assert cik_path.keys() == lookup.keys()
assert len(set(lookup.values())) == len(set(stock_data.keys()))


# At that point, we have a {CIK: ticker} for which the stock is known, which will enable comparison and all down the road.

# ## Review all CIKs: make sure there is only one submission per quarter

# In this section, the goal is to build a list of CIK that will successfully be parsed for the time_range considered.
# It should be trivial for a vast majority of the CIK, but ideally there should be only one document per quarter for each CIK from the moment they are listed to the moment they are delisted.

# In[ ]:


# Create the list of quarters to consider
cik_path = pre_processing.review_cik_publications(cik_path, s)
print("[INFO] Removed all the CIK that did not have one report per quarter.")
print("cik_dict: {:,} CIK".format(len(cik_path)))


# In[ ]:


print("[INFO] We are left with {:,} CIKs that meet our requirements:".format(len(cik_path)))
print("- The ticker can be looked up in the CIK/ticker table")
print("- The stock data is available for that ticker")
print("- There is one and only one report per quarter")


# ## Dump all the data to postgres
# This is done so that the Flask webapp can retrieve the settings that were used at a later time.

# In[ ]:


print(list(cik_path.keys()).index(10456))  # Find BAX


# connector = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="1")

# postgres.settings_to_postgres(connector, s)

# header_lookup = (('CIK', 'integer'), ('TICKER', 'text'))
# postgres.lookup_to_postgres(connector, lookup, header_lookup)

# header = (('TICKER', 'text'), ('TIMESTAMP', 'date'), 
#           ('ASK', 'float'), ('MARKET_CAP', 'float'))
# path = os.path.join(home, 'Desktop/Insight project/Database/stock_data_filtered.csv')
# postgres.stock_data_csv_to_postgres(connector, path, header)

# stock_data_2 = postgres.retrieve_stock_data(connector)

# # Parse files

# Now we have a list of CIK that should make it until the end. It is time to open the relevant reports and start parsing. This step takes a lot of time and can get arbitrarily long as the metrics get fancier.
# 
# You do not want to keep in RAM all the parsed data. However, there are only ~100 quarters for which we have data and the stage 2 files are no more than 1 Mb in size (Apple seems to top out at ~ 325 kb). So 100 Mb per core + others, that's definitely doable. More cores will use more RAM, but the usage remains reasonable.
# 
# We use multiprocessing to go through N CIK at once but a single core is dedicated to going through a given CIK for the specified time_range. Such a core can be running for a while if the company has been in business for the whole time_range and publish a lot of text data in its 10-K.

# In[ ]:


try:
    sc.stop()
except:
    pass
# nb_processes_requested = 8


# In[ ]:


# Processing the reports will be done in parrallel in a random order
# Settings in s are cast to dict for pickling - the custom class is not supported
nb_cik_to_process = len(cik_path.keys())
#nb_cik_to_process = 50
#cik_path = {k: cik_path[k] for k in cik_path.keys() if k in list(cik_path.keys())[:nb_cik_to_process]}
cik_path = {k: cik_path[k] for k in cik_path.keys() if k in list(cik_path.keys())}

# print(list(cik_path.keys()).index(10456))  # Find BAX
cik_scores = {k: 0 for k in cik_path.keys()}  # Organized by ticker
data_to_process = ([k, v, {**s}, lm_dictionary] for k, v in cik_path.items())
assert cik_path.keys() == cik_scores.keys()
#print(data_to_process)
#result = process_cik(data_to_process[0])
#cik_perf[result[0]] = result[1]
#print(cik_perf)
#assert 0
processing_stats = [0, 0, 0, 0, 0, 0]
#qtr_metric_result = {key: [] for key in s['list_qtr']}
if nb_processes_requested > 1:
    with mp.Pool(processes=nb_processes_requested) as p:
    #with mp.Pool(processes=min(mp.cpu_count(), 1)) as p:
        print("[INFO] Starting a pool of {} workers".format(nb_processes_requested))

        with tqdm(total=nb_cik_to_process) as pbar:
            for i, value in tqdm(enumerate(p.imap_unordered(processing.process_cik, data_to_process))):
                pbar.update()
                #qtr = list_qtr[i]
                # Each quarter gets a few metrics
                if value[1] == {}:
                    # The parsing failed
                    del cik_scores[value[0]]
                else:
                    cik_scores[value[0]] = value[1]
                processing_stats[value[2]] += 1

elif nb_processes_requested == 1:
    print("[INFO] Running on {} core (multiprocessing is off)".format(nb_processes_requested))
    # print(list(data_to_process))
    with tqdm(total=nb_cik_to_process) as pbar:
        for i, value in tqdm(enumerate(map(processing.process_cik, data_to_process))):
            pbar.update()
            #qtr = list_qtr[i]
            # Each quarter gets a few metrics
            if value[1] == {}:
                # The parsing failed
                del cik_scores[value[0]]
            else:
                cik_scores[value[0]] = value[1]
            processing_stats[value[2]] += 1

elif nb_processes_requested == 0:
    # Spark mode!!
    print("[INFO] Running with Spark")
    sc = pyspark.SparkContext(appName="model_calculations")
    print("[INFO] Context started")
    spark_result = sc.parallelize(data_to_process).map(processing.process_cik)
    spark_result = spark_result.take(nb_cik_to_process)
    sc.stop()
    
    # Process the result
    with tqdm(total=nb_cik_to_process) as pbar:
        for i, value in tqdm(enumerate(spark_result)):
            pbar.update()
            #qtr = list_qtr[i]
            # Each quarter gets a few metrics
            if value[1] == {}:
                # The parsing failed
                del cik_scores[value[0]]
            else:
                cik_scores[value[0]] = value[1]
            processing_stats[value[2]] += 1
           
        #qtr_metric_result[value['0']['qtr']] = value

print("[INFO] {} CIK were successfully processed - {}/{} CIK failed.".format(len(cik_scores), len(cik_path)-len(cik_scores), len(cik_path)))
print("Detailed stats and error codes:", processing_stats)


# # Post-processing - Welcome to the gettho

# ## Flip the result dictionary to present a per qtr view

# In[ ]:


metric_scores = post_processing.create_metric_scores(cik_scores, lookup, stock_data, s)


# In[ ]:


print("[INFO] Number of companies that do not have data for a given qtr.")
print("This is because they are listed later in the time_range")
for qtr in s['list_qtr'][s['lag']:]:
    print(qtr, "{}/{}".format(len([cik for cik in metric_scores['diff_jaccard'][qtr] 
                    if metric_scores['diff_jaccard'][qtr][cik] == {}]), len(cik_scores)))


# In[ ]:


df = post_processing.metrics_correlation(metric_scores, s)


# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


df.info()


# In[ ]:


# Create the quintiles - do not re-run that cell or it will crash!
for m in s['metrics']:
    for qtr in s['list_qtr'][s['lag']:]:
        metric_scores[m][qtr] = post_processing.make_quintiles(metric_scores[m][qtr], s)


# In[ ]:


# Sanity check: Verify that there are no CIK left for which we do not have stock prices.
pnf = []
for m in s['metrics']:
    for qtr in s['list_qtr'][s['lag']:]:
        for l in s['bin_labels']:
            for cik in metric_scores[m][qtr][l]:
                _, _, flag_price_found = post_processing.get_share_price(cik, qtr, lookup, stock_data)
                if not flag_price_found:
                    print("[WARNING] [{}] No stock data for {} during {}".format(m, cik, qtr))
                    pnf.append(cik)
print("Unique cik", set(pnf))           


# In[ ]:


# metric_scores['diff_jaccard'][(2013, 1)]  # After


# In[ ]:


pf_values = post_processing.initialize_portfolio(metric_scores, s)


# In[ ]:


# pf_values['diff_jaccard'][(2013, 2)]


# In[ ]:


pf_values = post_processing.build_portfolio(pf_values, lookup, stock_data, s)


# In[ ]:


post_processing.check_pf_value(pf_values, s)


# In[ ]:


# pf_values['diff_jaccard'][(2013, 2)]


# ## Export the data to postgres

# In[ ]:


connector = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="1")


# ## cik_scores

# In[ ]:


# cik_scores[851968][(2013, 1)].keys()


# In[ ]:


header_cik_scores = (('CIK', 'integer'), ('QTR', 'text'), 
                     ('METRIC', 'text'), ('SCORE', 'float'),
                     ('TYPE', 'text'), ('PUBLISHED', 'date'))


# In[ ]:


postgres.cik_scores_to_postgres(connector, cik_scores, header_cik_scores, s)


# In[ ]:


# data = postgres.retrieve_cik_scores(connector, 851968, s)


# In[ ]:


# data[851968][(2013, 1)]


# ## metric_scores

# In[ ]:


# I.1. Push to csv
path = s['path_output_folder']
path_metric_scores = os.path.join(path, 'ms.csv')
header_metric_score = (('METRIC', 'text'),  ('QUARTER', 'text'),
                    ('QUINTILE', 'text'), ('CIK', 'integer'), 
                    ('SECTION', 'text'), ('SCORE', 'float'))
with open(path_metric_scores, 'w') as f:
    out = csv.writer(f, delimiter=';')
    out.writerow(['IDX'] + [h[0] for h in header_metric_score])
    c = 0
    for m in metric_scores:
        for qtr in metric_scores[m]:
            for l in metric_scores[m][qtr]:
                for cik in metric_scores[m][qtr][l]:
                    #sections = [section for section in metric_scores[m][qtr][l][cik] if section != '0' and section != 'total']
                    for section in metric_scores[m][qtr][l][cik]:
                        v = metric_scores[m][qtr][l][cik][section]
                        out.writerow([c, m, qtr, l, cik, section, v])
                        c += 1


# In[ ]:


# I.2. Move the csv to postgres
postgres.csv_to_postgres(connector, 'metric_scores', header_metric_score, path_metric_scores)


# In[ ]:


# II. Sanity check: retrieve the data and compare to existing values
ms = postgres.retrieve_ms_values_data(connector, path_metric_scores, s)
assert ms == metric_scores
del ms


# ## pf_values

# In[ ]:


# pf_values['diff_jaccard'][(2013, 1)]['incoming_compo']['Q1'][49196]


# In[ ]:


path = s['path_output_folder']
header_pf_values1 = (('METRIC', 'text'),  ('QUARTER', 'text'),
                    ('SECTION', 'text'), ('QUINTILE', 'text'),
                    ('CIK', 'integer'), ('TICKER', 'text'),
                    ('ASK', 'float'), ('MARKET_CAP', 'bigint'),
                    ('SHARE_COUNT', 'float'), ('VALUE', 'float'),
                    ('RATIO_PF_VALUE', 'float'))
header_pf_values2 = (('METRIC', 'text'),  ('QUARTER', 'text'),
                    ('SECTION', 'text'), ('QUINTILE', 'text'),
                    ('PF_VALUE', 'float'))

path1 = os.path.join(path, 'pf_values1.csv')
# I.1. Dump to csv all the CIK info
with open(path1, 'w') as f:
    out = csv.writer(f, delimiter=';')
    out.writerow(['IDX'] + [h[0] for h in header_pf_values1])
    c = 0  # Primary key counter
    for m in pf_values:
        for qtr in pf_values[m]:
            for section in ['incoming_compo', 'new_compo']:
                for l in pf_values[m][qtr][section]:
                    for cik in pf_values[m][qtr][section][l]:
                        v = pf_values[m][qtr][section][l][cik]
                        out.writerow([c, m, qtr, section, l, cik, *v])
                        c += 1

# I.2. Dump to csv all the pf values 
path2 = os.path.join(path, 'pf_values2.csv')
with open(path2, 'w') as f:
    out = csv.writer(f, delimiter=';')
    out.writerow(['IDX'] + [h[0] for h in header_pf_values2])
    c = 0  # Primary key counter
    for m in pf_values:
        for qtr in pf_values[m]:
            for section in ['incoming_value', 'new_value']:
                for l in pf_values[m][qtr][section]:
                    v = pf_values[m][qtr][section][l]
                    out.writerow([c, m, qtr, section, l, v])
                    c += 1


# In[ ]:


# I.3. CSV -> Postgres
postgres.csv_to_postgres(connector, 'pf_values_compo', header_pf_values1, path1)
postgres.csv_to_postgres(connector, 'pf_values_value', header_pf_values2, path2)


# In[ ]:


# II. Sanity check: retrieve the data and compare to existing values
pf = postgres.retrieve_pf_values_data(connector, path1, path2, s)
assert pf == pf_values
del pf


# # Display the data

# ## Portfolio view

# In[ ]:


ylim = [0.7, 1.5]
fig, ax = plt.subplots(len(s['diff_metrics']), len(index_data), figsize=(15, 10))
for idx_x, m in enumerate(s['diff_metrics']):
    for idx_y, index_name in enumerate(index_data):
        benchmark, bin_data = display.diff_vs_benchmark_ns(pf_values, index_name, index_data, m, s, norm_by_index=True)
        display.update_ax_diff_vs_benchmark(ax[idx_x, idx_y], benchmark, bin_data, index_name, s, ylim, m)

start = s['time_range'][0]   
end = s['time_range'][1]
plt.savefig(os.path.join(s['path_output_folder'], '{}Q{}_{}Q{}_{}_{}_sw-{}_lem-{}.png'
                         .format(str(start[0])[2:], start[1], 
                                 str(end[0])[2:], end[1],
                                 s['differentiation_mode'][0], s['pf_balancing'][0],
                                 int(s['stop_words']), int(s['lemmatize']))))
if display.run_from_ipython():
    plt.show()
else:
    plt.close(fig)


# In[ ]:


index_name = 'RUT'
diff_method = 'diff_sk_cosine_tf_idf'
diff_method = 'diff_jaccard'
# diff_method='diff_gfg_editDistDP'
benchmark, bin_data = display.diff_vs_benchmark_ns(pf_values, index_name, index_data, diff_method, s, norm_by_index=True)
display.plot_diff_vs_benchmark(benchmark, bin_data, index_name, s)


# ## For a given ticker

# ### Metrics vs stock price

# In[ ]:


cik = 851968
ticker = lookup[cik]
start_date = qtrs.qtr_to_day(s['time_range'][0], 'first', date_format='datetime')
stop_date = qtrs.qtr_to_day(s['time_range'][1], 'last', date_format='datetime')

extracted_stock_data = {k: v for k, v in stock_data[ticker].items() if start_date <= k <= stop_date}
#print(extracted_data)
extracted_cik_scores = cik_scores[cik]


# In[ ]:


extracted_stock_data = {k: v for k, v in stock_data[ticker].items() if start_date <= k <= stop_date}


# In[ ]:


benchmark, metric_data = display.diff_vs_stock(extracted_cik_scores, extracted_stock_data, ticker, s, method='diff')
display.plot_diff_vs_stock(benchmark, metric_data, ticker, s)


# ### Sentiment vs stock price

# In[ ]:


benchmark, metric_data = display.diff_vs_stock(extracted_cik_scores, extracted_stock_data, ticker, s, method='sentiment')
display.plot_diff_vs_stock(benchmark, metric_data, ticker, s, method='sentiment')


# In[ ]:




