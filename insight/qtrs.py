import os
import requests, zipfile, io
import urllib
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook, tqdm
import multiprocessing as mp
from datetime import datetime
import time
import csv
import itertools
import matplotlib
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import argparse
from termcolor import colored


def create_qtr_list(time_range):
    # Sanity checks
    assert len(time_range) == 2
    assert 1994 <= time_range[0][0] and 1994 <= time_range[1][0]
    assert 1 <= time_range[0][1] <= 4 and 1 <= time_range[1][1] <= 4
    assert time_range[1][0] >= time_range[0][0]
    if time_range[1][0] == time_range[0][0]:  # Same year
        assert time_range[1][1] >= time_range[0][1]  # Need different QTR
    
    list_qtr = []
    for year in range(time_range[0][0], time_range[1][0]+1):
        for qtr in range(1, 5):
            # Manage the start and end within a year
            if year == time_range[0][0]:
                if qtr < time_range[0][1]:
                    continue
            if year == time_range[1][0]:
                if qtr > time_range[1][1]:
                    break
            
            # Common case
            list_qtr.append((year, qtr))
    
    # Sanity checks
    assert list_qtr[0] == time_range[0]
    assert list_qtr[-1] == time_range[1]
    return list_qtr

def test_create_qtr_list():
    test_1 = create_qtr_list([(2018, 1), (2018, 4)])
    assert test_1 == [(2018, 1), (2018, 2), (2018, 3), (2018, 4)]
    test_2 = create_qtr_list([(2016, 2), (2017, 3)])
    assert test_2 == [(2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3)]
    return True
#test_create_qtr_list()

def yearly_qtr_list(time_range):
    year_list = []
    if time_range[0][0] == time_range[1][0]:
        year_list = create_qtr_list(time_range)
    else:
        for year in range(time_range[0][0], time_range[1][0]+1):
            if year == time_range[0][0]:
                year_list.append(create_qtr_list([(year, time_range[0][1]), (year, 4)]))
            elif year == time_range[1][0]:
                year_list.append(create_qtr_list([(year, 1), (year, time_range[1][1])]))
            else:
                year_list.append(create_qtr_list([(year, 1), (year, 4)]))
    return year_list

def test_yearly_qtr_list():
    test_1 = yearly_qtr_list([(2016, 2), (2016, 2)])
    assert test_1 == [(2016, 2)]
    test_2 = yearly_qtr_list([(2015, 2), (2016, 3)])
    assert test_2 == [[(2015, 2), (2015, 3), (2015, 4)], [(2016, 1), (2016, 2), (2016, 3)]]
    return True
#test_yearly_qtr_list()

# Build the URL for the master index of a given quarter
def qtr_to_master_url(qtr):
    assert type(qtr) == tuple
    url = r"https://www.sec.gov/Archives/edgar/full-index"
    return '{}/{}/QTR{}/master.zip'.format(url, qtr[0], qtr[1])

def master_url_to_filepath(url):
    qtr = url.split('/')
    return os.path.join(path_master_indexes, qtr[6], qtr[7], 'master.zip')

def create_list_url_master_zip(list_qtr):
    # Sanity checks
    assert len(list_qtr)
    
    list_master_idx = []
    for qtr in list_qtr:
        list_master_idx.append(qtr_to_master_url(qtr))
    return list_master_idx

def is_downloaded(filepath):
    #expected_path = master_url_to_filepath(url_idx)
    if os.path.isfile(filepath):
        return True
    else:  # Build the folder architecture if needed
        if not os.path.isdir(os.path.split(filepath)[0]):
            os.makedirs(os.path.split(filepath)[0])
        return False

def test_is_downloaded():
    test_1 = is_downloaded("/ahsbxaksjhbxhjx.txt")
    assert test_1 == False
    path_temp_file = os.path.join(home, "temp_test_is_downloaded.temp")
    with open(path_temp_file, 'w') as f:
        pass
    test_2 = is_downloaded(path_temp_file)
    os.remove(path_temp_file)
    assert test_2 == True
    return True
#test_is_downloaded()

# Create an unzipping function to be run by a pool of workers
def unzip_file(path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(os.path.split(path)[0])


def parse_index(path, doc_types):
    # This method is bound to be run in parallel
    # Parses one master index and returns the URL of all the interesting documents in a dictionary
    docs = {key: [] for key in doc_types}  # Initialize an empty partial dictionary
    with open(path, 'r', encoding="utf8", errors='ignore') as f:
        data = csv.reader(f, delimiter='|')
        
        for _ in range(11):  # Jump to line 12
            next(data)
        
        for row in data:
            if row[2] in doc_types:
                date = "".join(row[3].split('-'))  # Format is YYYYMMDD
                end_url = row[4]  # You get the end of the URL for that file
                # The CIK can be accessed by parsing the end_url so no need to store it
                docs[row[2]].append((date, end_url))    
    return docs


def doc_url_to_filepath(submission_date, end_url):
    # entry is a tuple containing the date and the end_url. 
    # The CIK can be found from the end_url
    # The submission ID can be found from there too
    
    cik = end_url.split('/')[2]
    submission_id = "".join(end_url.split('/')[3][:-4].split('-'))
    #print(entry, cik, submission_id)
    return os.path.join(path_daily_data, submission_date, cik, submission_id+'.html')


def doc_url_to_FilingSummary_url(end_url):
    # WARNING: Not all files have a filing summary. 10-Q and 10-K do.
    # Convert a document url to the url of its xml summary
    cik_folder = end_url.split('/')[:3]
    submission_id = "".join(end_url.split('/')[3][:-4].split('-'))
    final_url = "/".join([base_url.rstrip('/'), *cik_folder, submission_id, 'FilingSummary.xml'])
    return final_url


def display_download_stats(stats):
    """
    Just a better way to display the downloading stats rather than dumping the dict"""
    text = []
    try:
        if stats['free_space'] < 10*2**30:  # Display text in bold red if less than 10 Gb left
            for key in stats:
                to_append = key + ": {:,}".format(stats[key])
                text.append("{}".format(colored(to_append, 'red', attrs=['bold'])))
        else:
            for key in stats:
                text.append(key + ": {:,}".format(stats[key])) 
    except:  
        for key in stats:
            text.append(key + ": {:,}".format(stats[key]))   
    print("[INFO] " + " | ".join(text))

