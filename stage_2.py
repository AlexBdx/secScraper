#!/usr/bin/env python
# coding: utf-8

# # Configuration

# ## Packages to import

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'notebook')

from insight import *

import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import re
from tqdm import tqdm
import multiprocessing as mp
from collections import OrderedDict
import time
import pandas as pd


# ## Settings dictionary

# In[111]:


# [USER SETTINGS]
example = 'apple'  # Debug
# Examples of companies
example_companies = {
    'apple': ['AAPL', 320193],
    'baxter': ['BAX', 10456],
    'facebook': ['FB', 1326801],
    'google': ['GOOGL', 1652044],
    'microsoft': ['MSFT', 789019]
}

_s = {
    'path_stage_1_data': '/home/alex/Desktop/filtered_text_data/nd_data/',
    'path_stock_database': '/home/alex/Desktop/Insight project/Database/Ticker_stock_price.csv',
    'path_cik_ticker_lookup': '/home/alex/Desktop/Insight project/Database/cik_ticker.csv',
    'path_master_dictionary': '/home/alex/Desktop/Insight project/Database/LoughranMcDonald_MasterDictionary_2018.csv',
    'path_dump_crsp': '/home/alex/Desktop/Insight project/Database/dump_crsp_2001_2005.txt',
    'metrics': ['diff_jaccard', 'diff_cosine', 'diff_minEdit', 'diff_simple', 'sing_LoughranMcDonald'],
    'differentiation_mode': 'intersection',
    'ticker': example_companies[example][0],
    'cik': example_companies[example][1],
    'time_range': [(2010, 1), (2012, 4)],
    'bin_count': 5,
    'report_type': ['10-K', '10-Q'],
    'sections_to_parse_10k': [],
    'sections_to_parse_10q': [],
    'type_daily_price': 'closing'
}


# In[112]:


_s['portfolio_init_value'] = 1000000
_s['epsilon'] = 0.0001  # Rounding error
# Calculated settings
_s['list_qtr'] = qtrs.create_qtr_list(_s['time_range'])

if _s['bin_count'] == 5:
    _s['bin_labels'] = ['Q'+str(n) for n in range(1, _s['bin_count']+1)]
elif _s['bin_count'] == 10:
    _s['bin_labels'] = ['D'+str(n) for n in range(1, _s['bin_count']+1)]
else:
    raise ValueError('[ERROR] This type of bin has not been implemented yet.')

# Reports considered to calculate the differences
if _s['differentiation_mode'] == 'intersection':
    _s['lag'] = 1
    _s['sections_to_parse_10k'] = ['1a', '3', '7', '7a', '9a']
    _s['sections_to_parse_10q'] = ['_i_2', '_i_3', '_i_4', 'ii_1', 'ii_1a']
elif _s['differentiation_mode'] == 'yearly':
    _s['lag'] = 4
    _s['sections_to_parse_10k']: []
    _s['sections_to_parse_10q']: []

_s['intersection_table'] = {
        '10-K': ['1a', '3', '7', '7a', '9a'],
        '10-Q': ['ii_1a', 'ii_1', '_i_2', '_i_3', '_i_4']
}  # Exhibits are not taken into account
_s['straight_table'] = {
    '10-K': ['1', '1a', '1b', '2', '3', '4', '5', '6', '7', '7a', '8', '9', '9a', '9b', '10', '11', '12', '13', '14', '15'],
    '10-Q': ['_i_1', '_i_2', '_i_3', '_i_4', 'ii_1', 'ii_1a', 'ii_2', 'ii_3', 'ii_4', 'ii_5', 'ii_6']
}


# In[113]:


# Transfer s to a read only dict
read_only_dict = pre_processing.ReadOnlyDict()
for key in _s:  # Brute force copy
    read_only_dict[key] = _s[key]
s = read_only_dict  # Copy back
s.set_read_state(read_only=True)  # Set as read only


# # Load external tables

# ## Load master dictionary for sentiment analysis

# In[5]:


master_dictionary = pre_processing.load_master_dictionary(s['path_master_dictionary'])


# ## Extract the list of CIK for which we have complete data

# The main problem in our case is that we have 3 different database to play with:
# 1. The SEC provides information based on the CIK of the entity
# 2. Given that the CIK is used by no one else, we use a lookup table to transform that into tickers. But we do not have all the correspondances, so the list of useful CIK is shrunk.
# 3. Finally, we only have stock prices for so many tickers. So that shrinks the CIK list even further.
# 
# We end up with a reduced list of CIK that we can play with.

# ### Find all the unique CIK from the SEC filings

# In[6]:


cik_path = pre_processing.load_cik_path(s)


# ### Get the largest {CIK: ticker} possible given our lookup table

# In[7]:


lookup = pre_processing.load_lookup(s)
print("[INFO] Loaded {:,} CIK/Tickers correspondances.".format(len(lookup)))


# In[8]:


cik_path, lookup = pre_processing.intersection_sec_lookup(cik_path, lookup)
print("[INFO] Intersected SEC & lookup.")
print("cik_path: {:,} CIK | lookup: {:,} CIK"
      .format(len(cik_path), len(lookup)))


# ### Load stock data and drop all CIKs for which we don't have data

# In[9]:


# Load all stock prices
stock_data = pre_processing.load_stock_data(s)


# In[10]:


lookup, stock_data = pre_processing.intersection_lookup_stock(lookup, stock_data)
print("[INFO] Intersected lookup & stock data.")
print("lookup: {:,} tickers | stock_data: {:,} tickers"
      .format(len(lookup.values()), len(stock_data)))


# ## Back propagate these intersection all the way to cik_path

# Technically, we have just done it for lookup. So we only need to re-run an intersection for lookup and sec.

# In[11]:


cik_path, lookup = pre_processing.intersection_sec_lookup(cik_path, lookup)
print("[INFO] Intersected SEC & lookup.")
print("cik_path: {:,} CIK | lookup: {:,} CIK"
      .format(len(cik_path), len(lookup)))


# ## Sanity check

# At this point, cik_path and lookup should have the same number of keys as the CIK is unique in the path database.
# 
# However, multiple CIK can redirect to the same ticker if the company changed its ticker over time. That should be a very limited amount of cases though.

# In[12]:


assert cik_path.keys() == lookup.keys()
assert len(set(lookup.values())) == len(set(stock_data.keys()))


# At that point, we have a {CIK: ticker} for which the stock is known, which will enable comparison and all down the road.

# ## Review all CIKs: make sure there is only one submission per quarter

# In this section, the goal is to build a list of CIK that will successfully be parsed for the time_range considered.
# It should be trivial for a vast majority of the CIK, but ideally there should be only one document per quarter for each CIK from the moment they are listed to the moment they are delisted.

# In[13]:


# Create the list of quarters to consider
cik_path = pre_processing.review_cik_publications(cik_path, s)
print("[INFO] Removed all the CIK that did not have one report per quarter.")
print("cik_dict: {:,} CIK".format(len(cik_path)))


# In[14]:


print("[INFO] We are left with {:,} CIKs that meet our requirements:".format(len(cik_path)))
print("- The ticker can be looked up in the CIK/ticker tabke")
print("- The stock data is available for that ticker")
print("- There is one and only one report per quarter")


# In[15]:


"""
# [DEBUG]: isolate a subset of companies
company = 'apple'
cik_path = {
    example_companies['apple'][1]: cik_path[example_companies['apple'][1]],
    example_companies['microsoft'][1]: cik_path[example_companies['microsoft'][1]]
}
cik_path.keys()
"""


# # Parse files

# Now we have a list of CIK that should make it until the end. It is time to open the relevant reports and start parsing. This step takes a lot of time and can get arbitrarily long as the metrics get fancier.
# 
# You do not want to keep in RAM all the parsed data. However, there are only ~100 quarters for which we have data and the stage 2 files are no more than 1 Mb in size (Apple seems to top out at ~ 325 kb). So 100 Mb per core + others, that's definitely doable. More cores will use more RAM, but the usage remains reasonable.
# 
# We use multiprocessing to go through N CIK at once but a single core is dedicated to going through a given CIK for the specified time_range. Such a core can be running for a while if the company has been in business for the whole time_range and publish a lot of text data in its 10-K.

# In[16]:


#quarterly_submissions = {key: [] for key in list_qtr}
#stg2parser = parser.stage_2_parser(s)

def process_cik(data):
    # This function will be multiprocessed
    # Probably via imap_unordered call as all CIK are independent
    # 0. expand argument list
    cik = data[0]
    file_list = data[1]
    s = data[2]
    
    # 1. Parse all reports
    quarterly_submissions = {key: [] for key in s['list_qtr']}
    stg2parser = parser.stage_2_parser(s)
    file_list = sorted(file_list)
    for path_report in file_list:
        split_path = path_report.split('/')
        qtr = (int(split_path[-3]), int(split_path[-2][3]))  # Ex: (2016, 3)
        """
        try:
            assert qtr in s['list_qtr']  # Should always be True
        except:
            print(qtr)
            print(s['list_qtr'])
            raise
        """
        if qtr in quarterly_submissions.keys():
            published = split_path[-1].split('_')[0]
            published = datetime.strptime(published, '%Y%m%d').date()
            type_report = split_path[-1].split('_')[1]
            if type_report in s['report_type']:
                with open(path_report, errors='ignore') as f:
                    text_report = f.read()
                parsed_report = {}
                parsed_report['0'] = {'type': type_report, 'published': published, 'qtr': qtr}
                parsed_report['input'] = text_report
                #print(path_report)
                
                """Attempt to parse the report"""
                try:
                    parsed_report = stg2parser.parse(parsed_report)
                except:
                    # If it fails, we need to skip the whole CIK as it becomes a real mess otherwise.
                    print("[WARNING] {} failed parsing".format(path_report))
                    #raise
                    return (cik, {}, 1)
                quarterly_submissions[qtr].append(parsed_report)
    
    # Delete empty qtr - because not listed or delisted
    quarterly_submissions = {k: v for k, v in quarterly_submissions.items() if len(v) > 0}
    if len(quarterly_submissions) == 0:  # None of the reports were 10-Q or 10-K
        return (cik, {}, 2)
    idx_first_qtr = s['list_qtr'].index(sorted(list(quarterly_submissions.keys()))[0])
    idx_last_qtr = s['list_qtr'].index(sorted(list(quarterly_submissions.keys()))[-1])

    # Sanity checks: there should not be any issue here, but you never know
    for key in quarterly_submissions.keys():
        if len(quarterly_submissions[key]) == 0:
            print("[WARNING] No report were found for {} in the paths".format(key))
        elif len(quarterly_submissions[key]) > 1:
            print("[WARNING] {} reports were released in {}".format(len(quarterly_submissions[key]), key))
    """
    # Look for the first quarter for that company - might not have been listed at the start of the time_range
    for idx in range(sorted(s['list_qtr'])):
        if s['list_qtr'][idx] in quarterly_submissions.keys():
            idx_first_qtr = idx
            break
    # Look for the last quarter for that company - might have been delisted before the end of the time_range
    for idx in range(sorted(s['list_qtr']))[::-1]:
        if s['list_qtr'][idx] in quarterly_submissions.keys():
            idx_last_qtr = idx
            break
    """
    
    # 2. Process the pair differences
    if idx_last_qtr < idx_first_qtr + s['lag']:
        time.sleep(1)
        print("idx_first_qtr: {} | idx_last_qtr: {} | lag: {}".format(idx_first_qtr, idx_last_qtr, s['lag']))
        #print(cik)
        #print(file_list)
        print("[WARNING] Not enough valid reports for CIK {} in this time_range. Skipping.".format(cik))
        quarterly_results = {}  # This CIK will be easy to remove later on
        return (cik, {}, 3)
    
    quarterly_results = {key: 0 for key in s['list_qtr'][idx_first_qtr+s['lag']:idx_last_qtr+1]}  # Include last index
    assert idx_last_qtr>=idx_first_qtr+s['lag']
    for current_idx in range(idx_first_qtr+s['lag'], idx_last_qtr+1):
        previous_idx = current_idx - s['lag']
        current_qtr = s['list_qtr'][current_idx]
        previous_qtr = s['list_qtr'][previous_idx]
        
        try:
            submissions_current_qtr = quarterly_submissions[current_qtr]
            submissions_previous_qtr = quarterly_submissions[previous_qtr]
        except:
            print("This means that for a quarter, we only had an extra document not a real 10-X")
            return (cik, {}, 4)
        try:
            assert len(submissions_current_qtr) == 1
            assert len(submissions_previous_qtr) == 1
        except:
            print("Damn should not have crashed here...")
            return (cik, {}, 5)
        print("[INFO] Comparing current qtr {} to previous qtr {}"
              .format(s['list_qtr'][current_idx], s['list_qtr'][previous_idx]))
        
        data = [submissions_current_qtr[0], submissions_previous_qtr[0], s]
        #print(submissions_current_qtr)
        final_result = analyze_reports(data)
        quarterly_results[current_qtr] = final_result
    return (cik, quarterly_results, 0)
        


# In[17]:


def calculate_metrics(current_text, previous_text, s):
    """
    Calculate the metrics for a given pair of section text
    """
    section_result = {m: 0 for m in s['metrics']}
    sample = 100
    for m in s['metrics']:
        # Should use a decorator here
        if m == 'diff_jaccard':
            section_result[m] =  metrics.diff_jaccard(current_text, previous_text)
        elif m == 'diff_cosine':
            section_result[m] =  metrics.diff_cosine(current_text, previous_text)
        elif m == 'diff_minEdit':
             section_result[m] = metrics.diff_minEdit(current_text[:sample], previous_text[:sample])
        elif m == 'diff_simple':
            section_result[m] = metrics.diff_simple(current_text[:sample], previous_text[:sample])
        elif m == 'sing_LoughranMcDonald':
            section_result[m] = metrics.sing_sentiment(current_text, master_dictionary)
        else:
            raise ValueError('[ERROR] Requested method has not been implemented!')
    return section_result


# In[18]:


def average_report_scores(result, word_count, s):
    """
    Calculate the weighted average for each metric"""
    final_result = {m: 0 for m in s['metrics']}
    nb_metrics = len(result.keys())
    assert nb_metrics == len(s['metrics'])
    assert result.keys() == word_count.keys()
    
    # Create a few totals for the weighted averages
    stc = {k: v[0] for k, v in word_count.items()}  # stp: section_total_current
    stp = {k: v[1] for k, v in word_count.items()}  # stp: section_total_previous
    sts = sum(stc.values())  # section_total_single, basically nb words in all sections of interest in current text
    _std = {k: v[0] + v[1] for k, v in word_count.items()}  # temp
    std = sum(_std.values())  # section_total_diff, basically nb words in all sections of interest in both text
    
    # Weight each metric by the number of words involved in its computation.
    for section in result.keys():
        for m in final_result.keys():
            if m[:4] == 'sing':
                try:
                    final_result[m] += result[section][m]*(stc[section]/sts)  # Consider only the nb words in current doc
                except:
                    print(result[section][m], (stc[section]/sts))
                    raise
            elif m[:4] == 'diff':
                final_result[m] += result[section][m]*((stc[section]+stp[section])/std)  # Divide by the total nb or words involved in both sections
            else:
                raise ValueError('[ERROR] This type of operation is not supported. How do I average it?')
    
    # Sanity check: make sure the values are meaningful
    epsilon = 0.0001  # Account for rounding errors
    for m in final_result.keys():
        if m[:4] == 'sing':  # else case already handled above
            try:
                assert -1 - s['epsilon'] <= final_result[m] <= 1 + s['epsilon']
            except:
                print(final_result)
                raise
        elif m[:4] == 'diff':
            try:
                assert - s['epsilon'] <= final_result[m] <= 1 + s['epsilon']
            except:
                print(final_result)
                raise
    return final_result


# In[19]:


def analyze_reports(data):
    # This function is designed to be multiprocessed
    # Data is a list of dict containing the data to be used   
    current = data[0]
    previous = data[1]
    s = data[2]
    
    # We need to calculate the same things at the same time for comparison purposes. 
    word_count = dict()  # Counts the number of words in each section
    if s['differentiation_mode'] == 'intersection':  # Reports could be the same or different
        sections_to_consider = s['intersection_table']['10-K']
        result = {section: {} for section in sections_to_consider}  # 10-K notation
        #print("Created", result)
        
        for idx in range(len(sections_to_consider)):            
            current_section = s['intersection_table'][current['0']['type']][idx]
            previous_section = s['intersection_table'][previous['0']['type']][idx]
            #print("Working on {}".format(tuple((current_section, previous_section))))
            
            try:
                current_text, previous_text = pre_processing.normalize_texts(current[current_section], previous[previous_section])
            except KeyError:
                if current_section == 'ii_1a' or previous_section == 'ii_1a':
                    # That means there were no update on the 10-Q
                    # Not great but for now let's give it a similarity of 1
                    print("Typical issue - we will fill the section_result manually")
                    for m in s['metrics']:
                        section_result[m] = 1
                    result[sections_to_consider[idx]] = section_result
                    continue
                else:
                    raise KeyError('[ERROR] Something went wrong')
            word_count[sections_to_consider[idx]] = [len(current_text.split()), len(previous_text.split())]
            result[sections_to_consider[idx]] = calculate_metrics(current_text, previous_text, s)
    
    elif s['differentiation_mode'] == 'yearly':
        assert current['0']['type'] == previous['0']['type']
        report_type = current['0']['type']
        result = {section: {} for section in s['straight_table'][report_type]}  # 10-K notation
        
        for idx in range(len(s['straight_table'][report_type])):            
            current_section = s['straight_table'][report_type][idx]
            previous_section = s['straight_table'][report_type][idx]
            #print("Working on {}".format(tuple((current_section, previous_section))))
            
            try:
                current_text, previous_text = pre_processing.normalize_texts(current[current_section], previous[previous_section])
            except KeyError:
                if 1:
                #if current_section == 'ii_1a' or previous_section == 'ii_1a':
                    # That means there were no update on the 10-Q
                    # Not great but for now let's give it a similarity of 1
                    print("Typical issue - we will fill the section_result manually")
                    for m in s['metrics']:
                        section_result[m] = 1
                    result[s['straight_table'][report_type][idx]] = section_result
                    continue
                else:
                    raise KeyError('[ERROR] Sections {} and {} are not implemented.'.format(current_section, previous_section))
            word_count[sections_to_consider[idx]] = len(current_text.split()) + len(previous_text.split())
            result[s['straight_table'][report_type][idx]] = calculate_metrics(current_text, previous_text, s)
    else:
        raise ValueError('[ERROR] This differentiation mode is unknown!')
    
    
    # Final step: we take the average of each metric
    final_result = average_report_scores(result, word_count, s)    

    # Sanity checks
    assert type(final_result) == dict
    assert len(final_result) == len(s['metrics'])
    for key in final_result.keys():
        assert -1 - s['epsilon']<= final_result[key] <= 1 + s['epsilon']
    # Transfer the metadata
    final_result['0'] = current['0']
    print(final_result)
    
    return final_result  # Simple dictionary


# In[20]:


# Processing the reports will be done in parrallel in a random order
cik_scores = {k: 0 for k in cik_path.keys()}  # Organized by ticker
#debug = [[k, v, {**s}] for k, v in cik_path.items() if k==98338]  # settings are cast to dict for pickling
debug = [[k, v, {**s}] for k, v in cik_path.items()]  # settings are cast to dict for pickling

data_to_process = debug[:1000]  # Debug
#print(data_to_process)
#result = process_cik(data_to_process[0])
#cik_perf[result[0]] = result[1]
#print(cik_perf)
#assert 0
processing_stats = [0, 0, 0, 0, 0, 0]
#qtr_metric_result = {key: [] for key in s['list_qtr']}
with mp.Pool(processes=min(mp.cpu_count(), len(data_to_process))) as p:
#with mp.Pool(processes=min(mp.cpu_count(), 1)) as p:
    print("[INFO] Starting a pool of {} workers".format(min(mp.cpu_count(), len(data_to_process))))

    with tqdm(total=len(data_to_process)) as pbar:
        for i, value in tqdm(enumerate(p.imap_unordered(process_cik, data_to_process))):
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
print("[INFO] {} CIK failed to be processed.".format(sum(processing_stats[1:])))
print("Detailed stats:", processing_stats)


# # Post-processing

# ## Flip the result dictionary to present a per qtr view

# In[35]:


# Reorganize the dict to display the data per quarter instead
qtr_scores = {qtr: {} for qtr in s['list_qtr']}
for c in cik_path.keys():
    if c in cik_scores.keys():
        if cik_scores[c] == 0:
            del cik_scores[c]

for cik in tqdm(cik_scores):
    for qtr in cik_scores[cik]:
        qtr_scores[qtr][cik] = cik_scores[cik][qtr]

assert list(qtr_scores.keys()) == s['list_qtr']


# ## Create a separate dictionary for each metric

# In[36]:


sorted_keys = sorted(qtr_scores[(2010, 2)])
qtr_scores[(2010, 2)][sorted_keys[0]]


# In[37]:


# Create the new empty master dictionary
master_dict = {m: 0 for m in s['metrics']}
for m in s['metrics']:
    master_dict[m] = {qtr: 0 for qtr in s['list_qtr']}
master_dict


# In[51]:


# Populate it
for m in s['metrics']:
    for qtr in s['list_qtr']:
        #master_dict[m][qtr] = {cik: qtr_scores[qtr][cik][m] for cik in qtr_scores[qtr].keys()}
        master_dict[m][qtr] = [(cik, qtr_scores[qtr][cik][m]) for cik in qtr_scores[qtr].keys()]


# In[54]:


# Display the length for all qtr
for qtr in s['list_qtr']:
    print("qtr: {} length: {}".format(qtr, len(master_dict[s['metrics'][0]][qtr])))


# ## For each metric, split each qtr into 5 quintiles
# 
# For each metric and for each quarter, make quintiles containing all the (cik, score) tuples. 
# 
# Now at this point the portfolio is not balanced, it is just the list of companies we would like to invest in. We need to weigh each investment by the relative market cap. 

# In[130]:


# Populate it
# The two zeros are respectively nb shares unbalanced & balanced
for m in s['metrics']:
    for qtr in s['list_qtr']:
        #master_dict[m][qtr] = {cik: qtr_scores[qtr][cik][m] for cik in qtr_scores[qtr].keys()}
        master_dict[m][qtr] = [[cik, qtr_scores[qtr][cik][m], 0, 0] for cik in qtr_scores[qtr].keys()]


# In[131]:


def make_quintiles(x, s):
    # x is (cik, value)
    # Create labels and bins of the same size
    try:
        assert len(x[0]) == 4
    except:
        print(x[0])
        raise
    #labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    quintiles = {l: [] for l in s['bin_labels']}
    _, input_data, _, _ = zip(*x)
    mapping = pd.qcut(input_data, s['bin_count'], labels=False)
    #print(mapping)
    for idx_input, idx_output in enumerate(mapping):
        #idx_qcut = labels.index(associated_label)  # Find label's index
        quintiles[s['bin_labels'][idx_output]].append(x[idx_input])
    return quintiles


# In[132]:


# Reorganize each quarter 
for m in s['metrics'][:-1]:
    for qtr in s['list_qtr'][s['lag']:]:  # There cannot be a report for the first few qtr
        print(m, qtr)
        master_dict[m][qtr] = make_quintiles(master_dict[m][qtr], s)
        assert len(master_dict[m][qtr].keys()) == 5


# In[155]:


def dump_master_dict(path, master_dict):
    with open(path, 'w') as f:
        out = csv.writer(f, delimiter=';')
        header = ['METRIC', 'QUARTER', 'QUINTILE', 'CIK', 'SCORE']
        out.writerow(header)
        
        # Main writing loop
        for m in tqdm(s['metrics'][:-1]):
            for qtr in s['list_qtr'][1:]:
                for l in s['bin_labels']:
                    for entry in master_dict[m][qtr][l]:
                        out.writerow([m, qtr, l, entry[0], entry[1]])


# In[156]:


dump_master_dict(master_dict)

