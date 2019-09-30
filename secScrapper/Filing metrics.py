#!/usr/bin/env python
# coding: utf-8

# # Configuration

# In[1]:


from insight import *

import glob
import os
import csv
from datetime import datetime
import re
from tqdm import tqdm
import multiprocessing as mp

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
    'path_project_root': '/home/alex/Desktop/filtered_text_data/nd_data/',
    'path_ticker_data': '/home/alex/Desktop/Insight project/Database/Stocks',
    'metrics_diff': ['sim_jaccard', 'sim_cosine', 'sim_minEdit', 'sim_simple'],
    'metrics_sentiment': ['LoughranMcDonald'],
    'differentiation_mode': 'intersection',
    'ticker': example_companies[example][0],
    'cik': example_companies[example][1],
    'time_range': [(2006, 1), (2006, 4)],
    'report_type': ['10-K', '10-Q'],
    'sections_to_parse_10k': [],
    'sections_to_parse_10q': [],
    'type_daily_price': 'closing'
}


# Calculated settings
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


# In[4]:


# Make the settings dictionary read only
class ReadOnlyDict(dict):  # Settings should be read only, so the final dict will be read only
    __readonly = False  # Start with a read/write dict

    def set_read_state(self, read_only=True):
        """Allow or deny modifying dictionary"""
        self.__readonly = bool(read_only)

    def __setitem__(self, key, value):
        if self.__readonly:
            raise RuntimeError('This dictionary is currently read only!')
        return dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        if self.__readonly:
            raise RuntimeError('This dictionary is currently read only!')
        return dict.__delitem__(self, key)


# Transfer s to a read only dict
read_only_dict = ReadOnlyDict()
for key in _s:  # Brute force copy
    read_only_dict[key] = _s[key]
s = read_only_dict  # Copy back
s.set_read_state(read_only=True)  # Set as read only


# In[5]:


file_list = glob.glob(s['path_project_root']+'**/*.txt', recursive=True)
print("[INFO] Loaded {:,} files".format(len(file_list)))


# In[6]:


# Filter the file list for a single company
cik = '_' + str(s['cik']) + '_'
file_list = [path for path in file_list if re.search(cik, path)]
print("[INFO] Loaded {} files".format(len(file_list)))


# In[7]:


list_qtr = qtrs.create_qtr_list(s['time_range'])


# In[11]:


quarterly_submissions = {key: [] for key in list_qtr}
stg2parser = parser.stage_2_parser(s)

for path_report in tqdm(file_list):
    split_path = path_report.split('/')
    qtr = (int(split_path[-3]), int(split_path[-2][3]))  # Ex: (2016, 3)
    if qtr in quarterly_submissions.keys():
        published = split_path[-1].split('_')[0]
        published = datetime.strptime(published, '%Y%m%d').date()
        type_report = split_path[-1].split('_')[1]
        if type_report in s['report_type']:
            with open(path_report, errors='ignore') as f:
                text_report = f.read()
            parsed_report = dict()
            parsed_report['0'] = {'type': type_report, 'published': published, 'qtr': qtr}
            parsed_report['input'] = text_report
            print(path_report)
            parsed_report = stg2parser.parse(parsed_report)
            quarterly_submissions[qtr].append(parsed_report)

# Check the submission for quarters with more or less reports - you never know
for key in quarterly_submissions.keys():
    if len(quarterly_submissions[key]) == 0:
        print("[WARNING] No report were found for {} in the paths".format(key))
    elif len(quarterly_submissions[key]) > 1:
        print("[INFO] {} reports were released in {}".format(len(quarterly_submissions[key]), key))

# Build the list of reports to process
# We want a single list so we can easily set up parralel processing of it.
differential_reports = []
for idx in range(s['lag'], len(list_qtr)):
    submissions_current_qtr = quarterly_submissions[list_qtr[idx]]
    submissions_previous_qtr = quarterly_submissions[list_qtr[idx-s['lag']]]
    for sub in submissions_current_qtr:
        if sub['0']['type'] in s['report_type']:
            current_report = sub
    for sub in submissions_previous_qtr:
        if sub['0']['type'] in s['report_type']:
            previous_report = sub
    
    differential_reports.append([current_report, previous_report, s])

assert len(differential_reports) == len(list_qtr) - s['lag']
print(differential_reports[0][0]['0'])


def analyse_reports(data):
    # This function is designed to be multiprocessed
    # Data is a list of dict containing the data to be used
    assert len(data) == 3
    assert type(data[0]) == dict and type(data[1]) == dict

    current = data[0]
    previous = data[1]
    s = data[2]
    final_result = {m: 0 for m in s['metrics_diff']}
    
    # We need to calculate the same things at the same time for comparison purposes.
        
    sample = 10000  # Cap the number of characters to use for text comparisons
    if s['differentiation_mode'] == 'intersection':  # Reports could be the same or different
        result = {section: {} for section in s['intersection_table']['10-K']}  # 10-K notation
        # print("Created", result)
        
        for idx in range(len(s['intersection_table']['10-K'])):
            
            section_result = {m: 0 for m in s['metrics_diff']}
            
            current_section = s['intersection_table'][current['0']['type']][idx]
            previous_section = s['intersection_table'][previous['0']['type']][idx]
            # print("Working on {}".format(tuple((current_section, previous_section))))
            
            try:
                current_text = current[current_section]
                current_text = current_text.strip()
                previous_text = previous[previous_section]
                previous_text = previous_text.strip()
            except KeyError:
                if current_section == 'ii_1a' or previous_section == 'ii_1a':
                    # That means there were no update on the 10-Q
                    # Not great but for now let's give it a similarity of 1
                    print("Typical issue - we will fill the section_result manually")
                    for m in s['metrics_diff']:
                        section_result[m] = 1
                    result[s['intersection_table']['10-K'][idx]] = section_result
                    continue
                else:
                    raise KeyError('[ERROR] Something went wrong')

            for m in s['metrics_diff']:
                # Should use a decorator here
                if m == 'sim_jaccard':
                    section_result[m] =  metrics.sim_jaccard(current_text, previous_text)
                elif m == 'sim_cosine':
                    section_result[m] =  metrics.sim_cosine(current_text, previous_text)
                elif m == 'sim_minEdit':
                     section_result[m] = metrics.sim_minEdit(current_text[:sample], previous_text[:sample])
                elif m == 'sim_simple':
                    section_result[m] = metrics.sim_simple(current_text[:sample], previous_text[:sample])
                elif m == 'sentiment':
                    section_result[m] = metrics.sim_sentiment(current_text)
                else:
                    raise ValueError('[ERROR] Requested method has not been implemented!')

            result[s['intersection_table']['10-K'][idx]] = section_result  # Store the dictionary that comes out of it
    
    elif s['differentiation_mode'] == 'yearly':
        assert current['0']['type'] == previous['0']['type']
        report_type = current['0']['type']
        result = {section: {} for section in s['straight_table'][report_type]}  # 10-K notation
        
        for idx in range(len(s['straight_table'][report_type])):
            
            section_result = {m: 0 for m in s['metrics_diff']}
            
            current_section = s['straight_table'][report_type][idx]
            previous_section = s['straight_table'][report_type][idx]
            # print("Working on {}".format(tuple((current_section, previous_section))))
            
            try:
                current_text = current[current_section]
                current_text = current_text.strip()
                previous_text = previous[previous_section]
                previous_text = previous_text.strip()
            except KeyError:
                if 1:
                # if current_section == 'ii_1a' or previous_section == 'ii_1a':
                    # That means there were no update on the 10-Q
                    # Not great but for now let's give it a similarity of 1
                    print("Typical issue - we will fill the section_result manually")
                    for m in s['metrics_diff']:
                        section_result[m] = 1
                    result[s['straight_table'][report_type][idx]] = section_result
                    continue
                else:
                    raise KeyError('[ERROR] Sections {} and {} are not implemented.'.format(current_section, previous_section))

        for m in s['metrics_diff']:
            # Should use a decorator here
            if m == 'sim_jaccard':
                section_result[m] = metrics.sim_jaccard(current_text, previous_text)
            elif m == 'sim_cosine':
                section_result[m] = metrics.sim_cosine(current_text, previous_text)
            elif m == 'sim_minEdit':
                section_result[m] = metrics.sim_minEdit(current_text[:sample], previous_text[:sample])
            elif m == 'sim_simple':
                section_result[m] = metrics.sim_simple(current_text[:sample], previous_text[:sample])
            elif m == 'sentiment':
                section_result[m] = metrics.sim_sentiment(current_text)
            else:
                raise ValueError('[ERROR] Requested method has not been implemented!')

        result[s['straight_table'][report_type][idx]] = section_result  # Store the dictionary that comes out of it
    else:
        raise ValueError('[ERROR] This differentiation mode is unknown!')

    # Final step: we take the average of each metric
    nb_metrics = len(result.keys())
    
    # print("Result to sum", result)
    try:
        for section in result.keys():
            for m in final_result.keys():
                # print(final_result)

                final_result[m] += result[section][m]
        for m in final_result.keys():
            final_result[m] /= nb_metrics
    except:
        print(section)
        print(m)
        print(result[section])

    # Sanity checks
    assert type(final_result) == dict
    assert len(final_result) == len(s['metrics_diff'])
    for key in final_result.keys():
        assert -1 <= final_result[key] <= 1
    # Transfer the metadata
    final_result['0'] = current['0']
    print(final_result)
    
    return final_result  # Simple dictionary


# Processing the reports will be done in parallel in a random order
qtr_metric_result = {key: [] for key in list_qtr}
with mp.Pool(processes=min(mp.cpu_count(), len(differential_reports))) as p:
# with mp.Pool(processes=min(mp.cpu_count(), 1)) as p:
    with tqdm(total=len(differential_reports)) as pbar:
        for i, value in tqdm(enumerate(p.imap_unordered(analyse_reports, differential_reports))):
            pbar.update()
            # qtr = list_qtr[i]
            # Each quarter gets a few metrics
            qtr_metric_result[value['0']['qtr']] = value

# Sanity check
print(qtr_metric_result[s['time_range'][1]])


def load_stock_data(ticker, selected_span=None):
    path_ticker = os.path.join(s['path_ticker_data'], ticker.lower()+'.us.txt')
    with open(path_ticker) as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        data = {}
        selected_span = [(1900, 1), (2100, 1)] if selected_span is None else selected_span
        for row in reader:
            date = row[0].split('-')
            qtr = int(date[1])//3 + 1
            # print(tuple([int(date[0]), qtr]))
            if selected_span[0] <= tuple([int(date[0]), qtr]) <= selected_span[1]: 
                ts = datetime.strptime(row[0], '%Y-%m-%d').date()
                # ts = tuple(row[0].split('-'))
                if s['type_daily_price'] == 'opening':
                    data[ts] = float(row[1])  # Daily opening value
                elif s['type_daily_price'] == 'high':
                    data[ts] = float(row[2])  # Daily highest value
                elif s['type_daily_price'] == 'low':
                    data[ts] = float(row[3])  # Daily lowest value
                elif s['type_daily_price'] == 'closing':
                    data[ts] = float(row[4])  # Daily closing value
                elif s['type_daily_price'] == 'average':
                    data[ts] = (float(row[1]) + float(row[4]))/2  # Average of the opening and closing values
                else:
                    raise ValueError('[ERROR] Unknown type of price for this stock')
    return data


ticker_data = load_stock_data(s['ticker'], selected_span=s['time_range'])
display.diff_vs_stock(qtr_metric_result, ticker_data, s)