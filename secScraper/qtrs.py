import os
import zipfile
from datetime import datetime
import csv
from termcolor import colored


def create_qtr_list(time_range):
    """
    From a given time_range, create the list of qtr contained in it. Includes both qtr in the time_range. time_range
    is of the form [(year, QTR), (year, QTR)].

    :param time_range: a list of two tuples representing the start and finish qtr
    :return: a list of all the qtr included in the time_range.
    """
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


def yearly_qtr_list(time_range):
    """
    Give all the qtr of each year included in a given time_range. Both start and end qtr are included.

    :param time_range: start and finish qtr
    :return: list of lists
    """
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


def qtr_to_master_url(qtr):
    """
    Build the URL for the master index of a given quarter.

    :param qtr: given qtr
    :return: string that represents the URL
    """
    assert type(qtr) == tuple
    url = r"https://www.sec.gov/Archives/edgar/full-index"
    return '{}/{}/QTR{}/master.zip'.format(url, qtr[0], qtr[1])


def master_url_to_filepath(url):
    """
    Transforms a master URL into a local file path. This needs to be refactored to be driven from a settings dict.

    :param url: Initial EDGAR URL
    :return: local path
    """
    qtr = url.split('/')
    return os.path.join(path_master_indexes, qtr[6], qtr[7], 'master.zip')


def create_list_url_master_zip(list_qtr):
    """
    Generates the URLs for the master indexes for a list of qtr.

    :param list_qtr: list of qtr of interest
    :return: list of URLs
    """
    # Sanity checks
    assert len(list_qtr)
    
    list_master_idx = []
    for qtr in list_qtr:
        list_master_idx.append(qtr_to_master_url(qtr))
    return list_master_idx


def is_downloaded(filepath):
    """
    Checks if a file at a given path already exists or not.

    :param filepath: string that represents a local path
    :return: bool
    """
    if os.path.isfile(filepath):
        return True
    else:  # Build the folder architecture if needed
        if not os.path.isdir(os.path.split(filepath)[0]):
            os.makedirs(os.path.split(filepath)[0])
        return False


def unzip_file(path):
    """
    Create an unzipping function to be run by a pool of workers.

    :param path: string representing the path of a zip file
    :return: void
    """
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(os.path.split(path)[0])


def parse_index(path, doc_types):
    """
    Parses one master index and returns the URL of all the interesting documents in a dictionary.

    :param path: string representing the path of the master index
    :param doc_types: types of documents we are interested in, as a list
    :return: dict containing the end url for each type of doc we are interested in.
    """
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
    """
    This one is about onverting a URL into a local file path for download.

    :param submission_date: date the document was submitted
    :param end_url: end url as found in the master index
    :return: local download path for the html file
    """
    # entry is a tuple containing the date and the end_url. 
    # The CIK can be found from the end_url
    # The submission ID can be found from there too
    
    cik = end_url.split('/')[2]
    submission_id = "".join(end_url.split('/')[3][:-4].split('-'))
    return os.path.join(path_daily_data, submission_date, cik, submission_id+'.html')


def doc_url_to_FilingSummary_url(end_url):
    """
    Convert a document url to the url of its xml summary.
    WARNING: Not all files have a filing summary. 10-Q and 10-K do.

    :param end_url: end url as found in the master index
    :return: URL of the xml document that has the section info about the file.
    """
    cik_folder = end_url.split('/')[:3]
    submission_id = "".join(end_url.split('/')[3][:-4].split('-'))
    final_url = "/".join([base_url.rstrip('/'), *cik_folder, submission_id, 'FilingSummary.xml'])
    return final_url


def display_download_stats(stats):
    """
    A better way to display the downloading stats and make sure there is enough space on the disk for the download.
    If not, consider increasing your EBS size on AWS. If you fill up the disk, that sucks. Make sure there are some
    sacrificial files in your current terminal folder so you can easily regain control. Anyway.

    :param stats: download stats
    :return: void
    """
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


def previous_qtr(qtr, s):
    """
    For once, a self-explanatory function name! Calculate what the previous qtr is in s['list_qtr']

    :param qtr: given qtr
    :param s: Settings dictionary
    :return: previous qtr in s['list_qtr']
    """
    idx = s['list_qtr'].index(qtr)
    if idx == 0:
        raise ValueError('[ERROR] This is the first quarter in the time_range')
    else:
        return s['list_qtr'][idx-1]


def qtr_to_day(qtr, position, date_format='string'):
    """
    Dumb function that returns the first or last day in a quarter. Two options for the output type: string or datetime.

    :param qtr: given qtr
    :param position: specify 'first' or 'last' day of the qtr. By default last is the 31st so it might not exist.
    :param date_format: 'string' or 'datetime'. Specifies the output type.
    :return: result. Read the above
    """
    # Dumb function that returns the first or last day in a quarter.
    # WARNING: Last day is not necessarily a real calendar day.
    # WARNING: On that day the stock exchange was not necessarily open
    if position == 'first':
        result = "{}{}{}".format(str(qtr[0]), str((qtr[1]-1)*3+1).zfill(2), '01')
    elif position == 'last':
        result = "{}{}{}".format(str(qtr[0]), str(qtr[1]*3).zfill(2), '31')
    else:
        raise ValueError('[ERROR] Only first and last day of quarter are supported')
    
    if date_format == 'datetime':
        result = datetime.strptime(result, '%Y%m%d').date()
    return result

