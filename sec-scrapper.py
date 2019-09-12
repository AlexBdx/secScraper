#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import our libraries
import os
import requests, zipfile, io
import urllib
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook
import multiprocessing as mp
from datetime import datetime
import time
import csv
import itertools
import matplotlib
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np


# In[2]:


home = os.path.expanduser("~")
path_master_indexes = os.path.join(home, 'Desktop/data/master_indexes')
path_daily_data = os.path.join(home, 'Desktop/data/daily_data')
base_url = "https://www.sec.gov/Archives/"
verbose = False


# ## Set the parameters of interest

# ### Time range

# In[3]:


# Specify the start and finish of the data collection
# The range is specified in QTR (1 to 4 each year)
# The range is inclusive of the min the max
time_range = [(2018, 1), (2018, 4)]


# ### Documents of interests

# In[4]:


doc_types = [
    '10-K',
    '10-Q',
    '8-K'
]


# In[5]:


info = {}


# ## Create the list of master indexes to download

# In[6]:


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


# In[7]:


info['quarters'] = create_qtr_list(time_range)
info['quarters']


# In[8]:


# Build the URL for the master index of a given quarter
def qtr_to_master_url(qtr):
    assert type(qtr) == tuple
    url = r"https://www.sec.gov/Archives/edgar/full-index"
    return '{}/{}/QTR{}/master.zip'.format(url, qtr[0], qtr[1])


# In[9]:


def master_url_to_filepath(url):
    qtr = url.split('/')
    return os.path.join(path_master_indexes, qtr[6], qtr[7], 'master.zip')


# In[10]:


def create_list_url_master_zip(list_qtr):
    # Sanity checks
    assert len(list_qtr)
    
    list_master_idx = []
    for qtr in list_qtr:
        list_master_idx.append(qtr_to_master_url(qtr))
    return list_master_idx


# In[11]:


info['url_master_zip'] = create_list_url_master_zip(info['quarters'])
info['url_master_zip']


# ## Download all the master indexes as zip files

# In[12]:


def is_downloaded(filepath):
    #expected_path = master_url_to_filepath(url_idx)
    if os.path.isfile(filepath):
        return True
    else:  # Build the folder architecture if needed
        if not os.path.isdir(os.path.split(filepath)[0]):
            os.makedirs(os.path.split(filepath)[0])
        return False


# In[13]:


info['path_master_zip'] = []
for url in info['url_master_zip']:
    info['path_master_zip'].append(master_url_to_filepath(url))
info['path_master_zip']  # list of all the zips we need


# def is_downloaded(url_idx):
#     expected_path = master_url_to_filepath(url_idx)
#     if os.path.isfile(expected_path):
#         return True
#     else:  # Build the folder architecture if needed
#         if not os.path.isdir(os.path.split(expected_path)[0]):
#             os.makedirs(os.path.split(expected_path)[0])
#         return False

# In[14]:


"""Verify that the master index zip are present. If not, download it."""
download_stats = {
    'bytes_downloaded': 0,
    'count_downloaded': 0
}  # Number of files, bytes

for n, filepath in enumerate(tqdm_notebook(info['path_master_zip'])):
    # Check if that zip has already been downloaded
    # print(filepath, is_downloaded(filepath))
    # assert 0
    if not is_downloaded(filepath):
        #raw_idx = requests.get(url_idx)
        (filename, headers) = urllib.request.urlretrieve(info['url_master_zip'][n], filepath)
        if verbose:
            print(filename, headers)
        download_stats['bytes_downloaded'] += os.path.getsize(filepath)
        download_stats['count_downloaded'] += 1
    else:
        if verbose:
            print("[WARNING] Skipping {}: already downloaded."
                  .format(filepath))

print(download_stats)


# ## Unzip all the master indexes

# In[15]:


# Create an unzipping function to be run by a pool of workers
def unzip_file(path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(os.path.split(path)[0])


# In[16]:


# 1. Build the list of zip files to extract
files_to_unzip = []
info['path_master_idx'] = []
for path_master_zip in info['path_master_zip']:
    path_master_index = path_master_zip[:-3]+'idx'  # Convert to index
    info['path_master_idx'].append(path_master_index)
    if not os.path.isfile(path_master_index):
        files_to_unzip.append(path_master_zip)
        #unzip_file(file)
    else:
        if verbose:
            print("[WARNING] Skipping unzip of {}: already unzipped"
                  .format(path_master_zip))

# 2. Unzip all zips using a pool
if len(files_to_unzip):
    t0 = time.perf_counter()
    # Launch a pool of workers
    pool = mp.Pool(min(mp.cpu_count(), len(files_to_unzip)))
    #pool.map(unzip_file, files_to_unzip, chunksize=1)
    r = list(tqdm_notebook(pool.imap(unzip_file, files_to_unzip, chunksize=1), total=len(files_to_unzip)))
    pool.close()
    t1 = time.perf_counter()
    size_unzipped = download_stats['bytes_downloaded']//2**20
    print("[INFO] Unzipped {} Mb in {:.3f} s ({:.1f} Mb/s)"
          .format(size_unzipped, t1-t0, size_unzipped/((t1-t0))))
else:
    print("[WARNING] Nothing to unzip. Is this normal?.")

info['path_master_idx']


# ## Parse all the indexes

# In[17]:


def parse_index(path, doc_types):
    # This method is bound to be run in parallel
    # Parses one master index and returns the URL of all the interesting documents in a dictionary
    docs = {key: [] for key in doc_types}  # Initialize an empty partial dictionary
    with open(path) as f:
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


# In[18]:


# 1. Build the list of files to download based on the parsing
"""[TBR]
for file in info['path_master_zip']:
    path_index = file[:-3]+'idx'  # Switch to the idx file w/o saving it
    
list_master_idx = []
for path in download_stats['file_path']:
    list_master_idx.append(path[:-3]+'idx')
print(list_master_idx)
"""

# 1. Create the list of URL that have the documents of importance
doc_of_interest = {key: [] for key in doc_types}
if len(info['path_master_idx']):
    t0 = time.perf_counter()
    # Launch a pool of workers
    #pool = mp.Pool(min(mp.cpu_count(), len(list_master_idx)))
    #pool.map(parse_index, list_master_idx, chunksize=1)
    for master_idx in info['path_master_idx']:
        parsed_index = parse_index(master_idx, doc_types)  # Returns a dict with all doc types
        # Add the lists that came in the parsed_index to a general dictionary
        for key in parsed_index:
            doc_of_interest[key].append(parsed_index[key])
        # Now, doc_of_interest has a lists of lists of data
        
    #r = list(tqdm_notebook(pool.imap(parse_index, list_master_idx, chunksize=1), total=len(list_master_idx)))
    #pool.close()
    #print(r)
    
    # Merging all the dict via comprehension
    general_url = {key: list(itertools.chain.from_iterable(doc_of_interest[key])) for key in doc_of_interest}
    nb_url = {key: len(value) for key, value in general_url.items()}
    
    t1 = time.perf_counter()
    print()
    print("[INFO] Parsed {} indexes and merged {} URL in {:.3f} s ({:,.1f} URL/s)"
          .format(len(info['path_master_idx']), sum(nb_url.values()), t1-t0, sum(nb_url.values())/(t1-t0)))
else:
    print("[WARNING] No URL to merge. This is unusual.")


# 2. Display sample URLs
print(nb_url)
max_display = 10
print("\n\n[INFO] Displaying {} sample local paths:".format(max_display))
for key in general_url:
    print("\nReport type:", key, "({}/{:,})".format(max_display, nb_url[key]))
    for k in range(max_display//2):
        # print(general_url[key][k])
        print(general_url[key][k][0], general_url[key][k][1])
    print("...")
    for k in range(len(general_url[key]) - max_display//2, len(general_url[key])):
        print(general_url[key][k][0], general_url[key][k][1])


# ## Download the documents of interest

# In[19]:


"""Display how many documents are available for that time period"""
# Do some stats on the document dates as a sanity check
# Calculate the number of documents per date
stats = {}
for key in general_url:
    for entry in general_url[key]:
        datetime_date = datetime.strptime(entry[0], '%Y%m%d')
        try:
            stats[datetime_date] += 1
        except:
            stats[datetime_date] = 1

assert sum(stats.values()) == sum(nb_url.values())


# Sort the list for plotting and display
lists = sorted(stats.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.figure(figsize=(15, 5))
dates = matplotlib.dates.date2num(x)
plt.plot_date(dates, y)
plt.title("Historical repartition of relevant documents\nTotal count: {:,} | {} to {}"
          .format(sum(stats.values()), time_range[0], time_range[1]), fontsize=20)
plt.ylabel('Document count [-]', fontsize=16)
plt.xlabel('Date [-]', fontsize=16)
plt.show()


# def is_downloaded(filepath):
#     #expected_path = master_url_to_filepath(url_idx)
#     if os.path.isfile(filepath):
#         return True
#     else:  # Build the folder architecture if needed
#         if not os.path.isdir(os.path.split(filepath)[0]):
#             os.makedirs(os.path.split(filepath)[0])
#         return False

# In[20]:


def doc_url_to_filepath(submission_date, end_url):
    # entry is a tuple containing the date and the end_url. 
    # The CIK can be found from the end_url
    # The submission ID can be found from there too
    
    cik = end_url.split('/')[2]
    submission_id = "".join(end_url.split('/')[3][:-4].split('-'))
    #print(entry, cik, submission_id)
    return os.path.join(path_daily_data, submission_date, cik, submission_id+'.html')


# In[21]:


def doc_url_to_FilingSummary_url(end_url):
    # WARNING: Not all files have a filing summary. 10-Q and 10-K do.
    # Convert a document url to the url of its xml summary
    cik_folder = end_url.split('/')[:3]
    submission_id = "".join(end_url.split('/')[3][:-4].split('-'))
    final_url = "/".join([base_url.rstrip('/'), *cik_folder, submission_id, 'FilingSummary.xml'])
    return final_url


# In[22]:


# Generate the list of local path
general_path = {key: [] for key in doc_types}
for file_type in doc_types:
    for entry in tqdm_notebook(general_url[file_type]):
        general_path[file_type].append(doc_url_to_filepath(*entry))
    assert len(general_path[file_type]) == len(general_url[file_type])

max_display = 10
print("[INFO] Displaying {} sample local paths:".format(max_display))
for key in general_path:
    print("\nReport type:", key, "({}/{:,})".format(max_display, nb_url[key]))
    for k in range(max_display//2):
        print(general_path[key][k])
    print("...")
    for k in range(len(general_path[key]) - max_display//2, len(general_path[key])):
        print(general_path[key][k])


# In[23]:


max_download = 3  # Of each type
min_time_between_requests = 0.1  # [s]
download_stats = {
    'bytes_downloaded': 0,
    'count_downloaded': 0
}  # Number of files, bytes


last_request = time.perf_counter()  # Initialize this counter
for file_type in doc_types:
    counter = 0
    for entry in tqdm_notebook(general_url[file_type]):
        if counter < max_download:
            path_doc_old = doc_url_to_filepath(*entry)  # [TBR]
            path_doc = general_path[file_type][counter]
            assert path_doc == path_doc_old
            #general_path[file_type].append(path_doc)
            
            """[OPTIONAL] Download the FilingSummary when it exists"""
            """
            if file_type == '10-K' or file_type == '10-Q':
                # A filing summary *might* be available
                # 1. Check if the FilingSummary has already been downloaded
                path_filing_summary = path_doc[:-3] + '-index.xml'
                if not is_downloaded(path_filing_summary):
                    url_filing_summary = doc_url_to_FilingSummary_url(entry[1])
                    print("Final URL:", url_filing_summary)
                    (filename, headers) = urllib.request.urlretrieve(url_file_index, filepath)
                    print((filename, headers))
                else:
                    if verbose:
                        print("[WARNING] Document at {} already downloaded".format(url_file_index))
            """
            
            # Check if the file has already been downloaded
            if not is_downloaded(path_doc):
                url_doc = base_url + entry[1]
                elapsed_since_last_requests = time.perf_counter() - last_request
                if elapsed_since_last_requests < min_time_between_requests:
                    print("[WARNING] Will wait for {:.3f} s"
                          .format(min_time_between_requests - elapsed_since_last_requests))
                    time.sleep(min_time_between_requests - elapsed_since_last_requests)
                if verbose:
                    print("[INFO] Time since last request: {:.3f} s".format(elapsed_since_last_requests))
                last_request = time.perf_counter()
                (filename, headers) = urllib.request.urlretrieve(url_doc, path_doc)
                download_time = time.perf_counter() - last_request
                #print((filename, headers))
                downloaded_size = os.path.getsize(path_doc)
                download_stats['bytes_downloaded'] += downloaded_size
                download_stats['count_downloaded'] += 1
                print("[INFO] [{}] Latest download speed: {:,} kb in {:.3f} s ({:,.1f} kb/s)"
                      .format(file_type, downloaded_size//2**10, download_time, downloaded_size/(2**10*download_time)))
            else:
                if verbose:
                    print("[WARNING] Document at {} already downloaded".format(path_doc))
        else:
            break
        counter += 1
download_stats


# ## Parse a doc and run it through the NLP

# ### Parse a doc with Vader - test

# In[24]:


analyser = SentimentIntensityAnalyzer()
text = """Punctuation: The use of an exclamation mark(!), increases the magnitude of the intensity without modifying the semantic orientation. For example, “The food here is good!” is more intense than “The food here is good.” and an increase in the number of (!), increases the magnitude accordingly. Sign in
Analytics Vidhya

    About Us
    Machine Learning
    Deep Learning
    Hackathons
    Contribute
    Courses

Simplifying Sentiment Analysis using VADER in Python (on Social Media Text)
An easy to use Python library built especially for sentiment analysis of social media texts.
Parul Pandey
Parul Pandey
Sep 23, 2018 · 8 min read
PC:Pixabay/PDPics

    “If you want to understand people, especially your customers…then you have to be able to possess a strong capability to analyze text. “ — Paul Hoffman, CTO:Space-Time Insight

The 2016 US Presidential Elections were important for many reasons. Apart from the political aspect, the major use of analytics during the entire canvassing period garnered a lot of attention. During the elections, millions of Twitter data points, belonging to both Clinton and Trump, were analyzed and classified with a sentiment of either positive, neutral, or negative. Some of the interesting outcomes that emerged from the analysis were:

    The tweets that mentioned ‘@realDonaldTrump’ were greater than those mentioning ‘@HillaryClinton’, indicating the majority were tweeting about Trump.
    For both candidates, negative tweets outnumbered the positive ones.
    The Positive to Negative Tweet ratio was better for Trump than for Clinton.

This is the power that sentiment analysis brings to the table and it was quite evident in the U.S elections. Well, the Indian Elections are around the corner too and sentiment analysis will have a key role to play there as well.
What is Sentiment Analysis?
source

Sentiment Analysis, or Opinion Mining, is a sub-field of Natural Language Processing (NLP) that tries to identify and extract opinions within a given text. The aim of sentiment analysis is to gauge the attitude, sentiments, evaluations, attitudes and emotions of a speaker/writer based on the computational treatment of subjectivity in a text.
Why is sentiment analysis so important?

Businesses today are heavily dependent on data. Majority of this data however, is unstructured text coming from sources like emails, chats, social media, surveys, articles, and documents. The micro-blogging content coming from Twitter and Facebook poses serious challenges, not only because of the amount of data involved, but also because of the kind of language used in them to express sentiments, i.e., short forms, memes and emoticons.

Sifting through huge volumes of this text data is difficult as well as time-consuming. Also, it requires a great deal of expertise and resources to analyze all of that. Not an easy task, in short.

Sentiment Analysis is also useful for practitioners and researchers, especially in fields like sociology, marketing, advertising, psychology, economics, and political science, which rely a lot on human-computer interaction data.

Sentiment Analysis enables companies to make sense out of data by being able to automate this entire process! Thus they are able to elicit vital insights from a vast unstructured dataset without having to manually indulge with it.
Why is Sentiment Analysis a Hard to perform Task?

Though it may seem easy on paper, Sentiment Analysis is actually a tricky subject. There are various reasons for that:

    Understanding emotions through text are not always easy. Sometimes even humans can get misled, so expecting a 100% accuracy from a computer is like asking for the Moon!
    A text may contain multiple sentiments all at once. For instance,

    “The intent behind the movie was great, but it could have been better”.

The above sentence consists of two polarities, i.e., Positive as well as Negative. So how do we conclude whether the review was Positive or Negative?

    Computers aren’t too comfortable in comprehending Figurative Speech. Figurative language uses words in a way that deviates from their conventionally accepted definitions in order to convey a more complicated meaning or heightened effect. Use of similes, metaphors, hyperboles etc qualify for a figurative speech. Let us understand it better with an example.

    “The best I can say about the movie is that it was interesting.”

Here, the word ’interesting’ does not necessarily convey positive sentiment and can be confusing for algorithms.

    Heavy use of emoticons and slangs with sentiment values in social media texts like that of Twitter and Facebook also makes text analysis difficult. For example a “ :)” denotes a smiley and generally refers to positive sentiment while “:(” denotes a negative sentiment on the other hand. Also, acronyms like “LOL“, ”OMG” and commonly used slangs like “Nah”, “meh”, ”giggly” etc are also strong indicators of some sort of sentiment in a sentence.

These are few of the problems encountered not only with sentiment analysis but with NLP as a whole. In fact, these are some of the Open-ended problems of the Natural Language Processing field.
VADER Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labelled according to their semantic orientation as either positive or negative.

VADER has been found to be quite successful when dealing with social media texts, NY Times editorials, movie reviews, and product reviews. This is because VADER not only tells about the Positivity and Negativity score but also tells us about how positive or negative a sentiment is.

It is fully open-sourced under the MIT License. The developers of VADER have used Amazon’s Mechanical Turk to get most of their ratings, You can find complete details on their Github Page.
methods and process approach overview
Advantages of using VADER

VADER has a lot of advantages over traditional methods of Sentiment Analysis, including:

    It works exceedingly well on social media type text, yet readily generalizes to multiple domains
    It doesn’t require any training data but is constructed from a generalizable, valence-based, human-curated gold standard sentiment lexicon
    It is fast enough to be used online with streaming data, and
    It does not severely suffer from a speed-performance tradeoff.

    The source of this article is a very easy to read paper published by the creaters of VADER library.You can read the paper here.

Enough of talking. Let us now see practically how does VADER analysis work for which we will have install the library first.
Installation

The simplest way is to use the command line to do an installation from [PyPI] using pip. Check their Github repository for the detailed explanation.

> pip install vaderSentiment

Once VADER is installed let us call the SentimentIntensityAnalyser object,

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzeranalyser = SentimentIntensityAnalyzer()

Working & Scoring

Let us test our first sentiment using VADER now. We will use the polarity_scores() method to obtain the polarity indices for the given sentence.

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))

Let us check how VADER performs on a given review:

sentiment_analyzer_scores("The phone is super cool.")The phone is super cool----------------- {'neg': 0.0, 'neu': 0.326, 'pos': 0.674, 'compound': 0.7351}

Putting in a Tabular form:

    The Positive, Negative and Neutral scores represent the proportion of text that falls in these categories. This means our sentence was rated as 67% Positive, 33% Neutral and 0% Negative. Hence all these should add up to 1.
    The Compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1(most extreme negative) and +1 (most extreme positive). In the case above, lexicon ratings for andsupercool are 2.9and respectively1.3. The compound score turns out to be 0.75 , denoting a very high positive sentiment.

compound score metric

read here for more details on VADER scoring methodology.

VADER analyses sentiments primarily based on certain key points:

    Punctuation: The use of an exclamation mark(!), increases the magnitude of the intensity without modifying the semantic orientation. For example, “The food here is good!” is more intense than “The food here is good.” and an increase in the number of (!), increases the magnitude accordingly.

See how the overall compound score is increasing with the increase in exclamation marks.

    Capitalization: Using upper case letters to emphasize a sentiment-relevant word in the presence of other non-capitalized words, increases the magnitude of the sentiment intensity. For example, “The food here is GREAT!” conveys more intensity than “The food here is great!”

    Degree modifiers: Also called intensifiers, they impact the sentiment intensity by either increasing or decreasing the intensity. For example, “The service here is extremely good” is more intense than “The service here is good”, whereas “The service here is marginally good” reduces the intensity.

    Conjunctions: Use of conjunctions like “but” signals a shift in sentiment polarity, with the sentiment of the text following the conjunction being dominant. “The food here is great, but the service is horrible” has mixed sentiment, with the latter half dictating the overall rating.

    Preceding Tri-gram: By examining the tri-gram preceding a sentiment-laden lexical feature, we catch nearly 90% of cases where negation flips the polarity of the text. A negated sentence would be “The food here isn’t really all that great”.

Handling Emojis, Slangs and Emoticons.

VADER performs very well with emojis, slangs and acronyms in sentences. Let us see each with an example.

    Emojis

print(sentiment_analyzer_scores('I am 😄 today'))
print(sentiment_analyzer_scores('😊'))
print(sentiment_analyzer_scores('😥'))
print(sentiment_analyzer_scores('☹️'))#OutputI am 😄 today---------------------------- {'neg': 0.0, 'neu': 0.476, 'pos': 0.524, 'compound': 0.6705}😊--------------------------------------- {'neg': 0.0, 'neu': 0.333, 'pos': 0.667, 'compound': 0.7184}😥--------------------------------------- {'neg': 0.275, 'neu': 0.268, 'pos': 0.456, 'compound': 0.3291}☹️-------------------------------------- {'neg': 0.706, 'neu': 0.294, 'pos': 0.0, 'compound': -0.34}💘--------------------------------------- {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

    Slangs

print(sentiment_analyzer_scores("Today SUX!"))
print(sentiment_analyzer_scores("Today only kinda sux! But I'll get by, lol"))#outputToday SUX!------------------------------ {'neg': 0.779, 'neu': 0.221, 'pos': 0.0, 'compound': -0.5461}Today only kinda sux! But I'll get by, lol {'neg': 0.127, 'neu': 0.556, 'pos': 0.317, 'compound': 0.5249}

    Emoticons

print(sentiment_analyzer_scores("Make sure you :) or :D today!"))Make sure you :) or :D today!----------- {'neg': 0.0, 'neu': 0.294, 'pos': 0.706, 'compound': 0.8633}

We saw how VADER can easily detect sentiment from emojis and slangs which form an important component of the social media environment.
Conclusion

The results of VADER analysis are not only remarkable but also very encouraging.The outcomes highlight the tremendous benefits that can be attained by use of VADER in cases of micro-blogging sites wherein the text data is a complex mix of a variety of text.

I hope you found this article useful. Let me know if you have any doubts or suggestions in the comments section below.

    Machine Learning
    Artificial Intelligence
    NLP
    Sentiment Analysis
    Data Science

Parul Pandey

Written by
Parul Pandey
Data Science+Community+Evangelism @H2O.ai | Linkedin: https://www.linkedin.com/in/parul-pandey-a5498975/
Analytics Vidhya
Analytics Vidhya
Analytics Vidhya is a community of Analytics and Data Science professionals. We are building the next-gen data science ecosystem https://www.analyticsvidhya.com
See responses (22)
More From Medium
More from Analytics Vidhya
How to apply data augmentation to deal with unbalanced datasets in 20 lines of code
Arnaldo Gualberto
Arnaldo Gualberto in Analytics Vidhya
Aug 30 · 4 min read
269
More from Analytics Vidhya
One-way Analysis of Variance (ANOVA) with Python
Valentina Alto
Valentina Alto in Analytics Vidhya
Sep 4 · 5 min read
187
More from Analytics Vidhya
Dimension Manipulation using Autoencoder in Pytorch on MNIST dataset
Garima Nishad
Garima Nishad in Analytics Vidhya
Sep 5 · 5 min read
515
Discover Medium
Welcome to a place where words matter. On Medium, smart voices and original ideas take center stage - with no ads in sight. Watch
Make Medium yours
Follow all the topics you care about, and we’ll deliver the best stories for you to your homepage and inbox. Explore
Become a member
Get unlimited access to the best stories on Medium — and support writers while you’re at it. Just $5/month. Upgrade
About
Help
Legal
"""
t0 = time.perf_counter()
score = analyser.polarity_scores(text)
t1 = time.perf_counter()
print(len(text), t1-t0, score)
print(len(text)/(t1-t0))


# In[28]:


for file_type in doc_types:
    counter = 0
    for entry in tqdm_notebook(general_path[file_type]):
        # print(entry)
        if counter < max_download:
            t0 = time.perf_counter()
            with open(entry) as f:  # Load the file
                html_doc = f.read()
            soup = BeautifulSoup(html_doc, 'html.parser')
            text_only = soup.get_text()
            t1 = time.perf_counter()
            print("Before: {:,} byte | After: {:,} byte | Done in: {:.1f} s"
                  .format(len(html_doc), len(text_only), t1-t0))

            # Get a few subset of the text for analysis
            average_score = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
            length_subset = 10000
            iterations = 100
            
            for _ in range(iterations):
                if len(text_only) > length_subset:
                    start_index = np.random.randint(len(text_only)-length_subset)
                    score = analyser.polarity_scores(text_only[start_index:start_index+length_subset])
                else:
                    score = analyser.polarity_scores(text_only)
                for key in average_score:
                    average_score[key] += score[key]

            for key in average_score:
                    average_score[key] /= iterations
            t2 = time.perf_counter()
            print(average_score)
            """
            print("Took {:.3f} s to analyze {} byte ({} byte/s) | Result: {}"
                  .format(t2-t1, length_subset*iterations, length_subset*iterations/(t2-t1), average_score))
            """
        else:
            break
        counter += 1


# In[ ]:




