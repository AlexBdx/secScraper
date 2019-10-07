import csv
from tqdm import tqdm
from datetime import datetime
import glob
import multiprocessing as mp

class ReadOnlyDict(dict):
    """
    Simple dictionary class that makes it read-only. This applies to the settings dictionary most likely.
    """
    __readonly = False  # Start with a read/write dict

    def set_read_state(self, read_only=True):
        """
        Allow or deny modifying dictionary.

        :param read_only: bool to set the state of the dictionary
        :return:
        """
        self.__readonly = bool(read_only)

    def __setitem__(self, key, value):
        """
        Prevents modification of an item when read only.

        :param key: A key
        :param value: A value
        :return: void
        """
        if self.__readonly:
            raise RuntimeError('This dictionary is currently read only!')
        return dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        """
        Prevents deletion of an item when read only.

        :param key: A key
        :return: void
        """
        if self.__readonly:
            raise RuntimeError('This dictionary is currently read only!')
        return dict.__delitem__(self, key)


def unique_cik(path_list):
    """
    Identify all unique CIK in a path list.

    :param path_list: list of path, most likely obtain from a recursive glob.glob
    :return: list of unique CIK found
    """
    all_cik = [int(e.split('/')[-1].split('_')[4]) for e in path_list]
    return set(all_cik)


def paths_to_cik_dict(file_list, unique_sec_cik):
    """
    Organizes a list of file paths into a dictionary, the keys being the CIKs. unique_sec_cik is used to initialize
    the cik_dict.

    :param file_list: unorganized list of paths
    :param unique_sec_cik: set of all unique CIK found
    :return: a dictionary containing all the paths, organized by CIKs
    """
    cik_dict = {k: [] for k in unique_sec_cik}
    for path in tqdm(file_list):
        split_path = path.split('/')
        cik = int(split_path[-1].split('_')[4])  # Cast to an int
        cik_dict[cik].append(path)
    return cik_dict


def load_cik_path(s):
    """
    Find all the file paths and organize them by CIK.

    :param s: Settings dictionary
    :return: Dictionary of paths with the keys being the CIK.
    """
    file_list = glob.glob(s['path_stage_1_data']+'**/*.txt', recursive=True)
    print("[INFO] Loaded {:,} 10-X".format(len(file_list)))
    file_list = filter_cik_path(file_list, s)
    print("[INFO] Shrunk to {:,} {}".format(len(file_list), s['report_type']))
    # print("[INFO] Example:", file_list[0])
    unique_sec_cik = unique_cik(file_list)  # Get unique CIKs
    print("[INFO] Found {:,} unique CIK in master index".format(len(unique_sec_cik)))
    cik_path = paths_to_cik_dict(file_list, unique_sec_cik)  # Create a dict based on that
    print("[INFO] cik_path contains data on {:,} CIK numbers".format(len(cik_path)))
    return cik_path


def filter_cik_path(file_list, s):
    """
    Filter out all the reports that are not of the considered type. The considered type is available in the settings
    dictionary.

    :param file_list:
    :param s:
    :return:
    """
    filtered_file_list = [f for f in file_list if f.split('/')[-1].split('_')[1] in s['report_type']]
    return filtered_file_list 


def load_lookup(s):
    """
    Load the CIK -> Lookup table.

    :param s: Settings dictionary
    :return: Lookup table in the form of a dictionary.
    """
    # Load the lookup table
    with open(s['path_lookup']) as f:
        cik_to_ticker = dict()
        reader = csv.reader(f, delimiter='|')
        next(reader)  # Skip header
        for row in reader:
            cik_to_ticker[int(row[0])] = row[1]
    return cik_to_ticker


def intersection_sec_lookup(cik_path, lookup):
    """
    Finds the intersection of the set of CIKs contained in the cik_path dictionary and the CIKs contained in the lookup
    table. This is part of the steps taken to ensure that we have bijections between all the sets of CIKs for all
    external databases.

    :param cik_path: Dictionary of paths organized by CIKs
    :param lookup: lookup table CIK -> ticker
    :return: both dictionaries with only the intersection of CIKs left as keys.
    """
    # 1. Create unique list of keys
    unique_cik = set(cik_path.keys())
    unique_lookup = set(lookup.keys())
    
    # 2. Intersection
    intersection_cik = list(unique_cik & unique_lookup)
    
    # 3. Update both dictionaries (fwd and backwd propagation)
    inter_cik = {cik: cik_path[cik] for cik in intersection_cik}
    inter_lookup = {cik: lookup[cik] for cik in intersection_cik}
    
    return inter_cik, inter_lookup


def load_stock_data(s, penny_limit=0, verbose=True):
    """
    Load all the stock data and pre-processes it.
    WARNING: Despite all (single process) efforts, this still takes a while. Using map seems to be the fastest
    way in python for that O(N) operation but it still takes ~ 60 s on my local machine (1/3rd reduction)

    :param s: Settings dictionary
    :return: dict stock_data[ticker][time stamp] = (closing, market cap)
    """
    with open(s['path_stock_database']) as f:
        header = next(f).split(',')
        header[-1] = header[-1].strip()
        idx_date = header.index("date")
        idx_ticker = header.index("TICKER")
        idx_closing = header.index("ASK")
        idx_outstanding_shares = header.index("SHROUT")

        start = s['time_range'][0]
        finish = s['time_range'][-1]
        print("[INFO] Loading data from {} to {}".format(start, finish))

        def process_line(line):
            row = line.split(',')
            date = row[idx_date]
            qtr = tuple((int(date[:4]), int(date[4:6]) // 3 + 1))

            if start <= qtr <= finish:  # Only data in time range
                row[-1] = row[-1].strip()
                ticker = row[idx_ticker]
                closing_price = row[idx_closing]
                outstanding_shares = row[idx_outstanding_shares]
                if ticker == '' or closing_price == '' or outstanding_shares == '':
                    return '0', 1, 0, 0
                # 2. Process the row
                closing_price = float(closing_price)
                market_cap = 1000 * closing_price * int(outstanding_shares)
                if market_cap < penny_limit:
                    return '0', ticker, 0, 0
                return ticker, datetime.strptime(date, '%Y%m%d').date(), closing_price, market_cap
            else:
                return '0', 3, 0, 0

        print("[INFO] Starting the mapping")
        result = map(process_line, f)
        stock_data = dict()
        # previous_ticker = '0'
        counter_incomplete_line = 0
        counter_line_out_of_range = 0
        penny_stocks = []
        nb_lines = 0
        for e in tqdm(result, total=30563446):
            nb_lines += 1
            if e[0] != '0':
                # if e[0] != previous_ticker:  # Not faster and less flexible
                if e[0] not in stock_data.keys():
                    stock_data[e[0]] = dict()
                    # previous_ticker = e[0]
                stock_data[e[0]][e[1]] = (e[2], e[3])
            else:
                if e[1] == 1:  # Incomplete line
                    counter_incomplete_line += 1
                elif type(e[1]) == str:
                    penny_stocks.append(e[1])
                elif e[1] == 3:
                    counter_line_out_of_range += 1

        # Remove all the penny stocks
        penny_stocks = set(penny_stocks)
        stock_data = {k: v for k, v in stock_data.items() if k not in penny_stocks}

        if verbose:
            print("[INFO] stock_data load statistics:")
            print("Incomplete lines: {:,}/{:,}".format(counter_incomplete_line, nb_lines))
            print("Penny stocks found (at least one entry below threshold): {}/{}"
            .format(len(penny_stocks), len(penny_stocks) + len(stock_data.keys())))
            print("Lines out of range: {:,}/{:,}".format(counter_line_out_of_range, nb_lines))
        return stock_data


def load_index_data(s):
    """
    Loads the csv files containing the daily historical data for the stock market indexes that were selected in s.

    :param s: Settings dictionary
    :return: dictionary of the index data.
    """
    # 1. Find all the indexes in the folder
    file_list = glob.glob(s['path_stock_indexes']+'**/*.csv', recursive=True)
    file_list = [f for f in file_list if f.split('/')[-1] != 'filtered_index_data.csv']
    index_names = [f.split('/')[-1][14:-4] for f in file_list]
    paths = zip(file_list, index_names)
    
    # 2. Open all these files and add the data to a dictionary
    index_data = {k: {} for k in index_names}
    for path in paths:
        with open(path[0]) as f:
            reader = csv.reader(f)
            header = next(reader)
            idx_date = header.index("Date")
            idx_closing = header.index("Close")
            for row in reader:
                date = datetime.strptime(row[idx_date], '%Y-%m-%d').date()
                index_data[path[1]][date] = float(row[1])  # Load all
    return index_data


def intersection_lookup_stock(lookup, stock):
    """
    Finds the intersection of the set of CIKs contained in the lookup dictionary and the CIKs contained in the stock
    database. This is part of the steps taken to ensure that we have bijections between all the sets of CIKs for all
    external databases.

    :param lookup: lookup dictionary
    :param stock: stock data, organized in a dictionary with tickers as keys.
    :return: both dictionaries with only the intersection of CIKs left as keys.
    """
    # 1. Create unique lists to compare
    unique_lookup = set(list(lookup.values()))
    unique_stock = set(list(stock.keys()))

    # 2. Create intersection of tickers
    intersection_tickers = list(unique_lookup & unique_stock)
    print(len(intersection_tickers))
    
    # 3. Return a new intersection dictionary
    inter_lookup = {k: v for k, v in lookup.items() if v in intersection_tickers}
    inter_stock = {k: stock[k] for k in stock.keys() if k in intersection_tickers}
    
    return inter_lookup, inter_stock


def review_cik_publications(cik_path, s):
    """Filter the CIK based on how many publications there are per quarter
    This function reviews all the CIK to make sure there is only 1 publication per qtr
    It provides a few hooks to correct issues but these have not been implemented.
    Around 10 % of the CIK seem to have problems at one point or another.

    :param cik_path:
    :param s: Settings dictionary
    :return: A filtered version of the cik_path dictionary - only has the keys that passed the test.
    """
    cik_to_delete = []
    for cik, paths in tqdm(cik_path.items()):
        # Make sure there are enough reports to enable diff calculations
        if not len(paths) > s['lag']:  # You need more reports than the lag
            cik_to_delete.append(cik)
            continue
        
        quarterly_submissions = {key: [] for key in s['list_qtr']}
        for path_report in paths:  # For each report for that CIK
            split_path = path_report.split('/')
            qtr = (int(split_path[-3]), int(split_path[-2][3]))  # Ex: (2016, 3)
            if qtr in quarterly_submissions.keys():
                published = split_path[-1].split('_')[0]
                published = datetime.strptime(published, '%Y%m%d').date()
                type_report = split_path[-1].split('_')[1]
                if type_report in s['report_type']:  # Add to the dict
                    metadata = {'type': type_report, 'published': published, 'qtr': qtr}
                    quarterly_submissions[qtr].append(metadata)

        # Check for continuity of the reports
        flag_continuity = check_report_continuity(quarterly_submissions, s)
        if not flag_continuity:  # Will ignore that CIK
            cik_to_delete.append(cik)
            continue
    
    # Create a subset of cik_dict based on the cik not faulty
    print()
    print("[INFO] {} CIKs caused trouble".format(len(cik_to_delete)))
    cik_dict = {k: v for k, v in cik_path.items() if k not in cik_to_delete}
    
    return cik_dict


def check_report_type(quarterly_submissions, qtr):
    """
    Verify that all the reports in quarterly_submissions were published at the right time based on their type. A 10-K
    is supposed to be published and only published in Q1. A 10-Q is supposed to be published and only published in
    Q2, Q3 or Q4.

    :param quarterly_submissions: dictionary of reports published, by qtr. There should only be one report per qtr
    :param qtr: A given qtr
    :return: void but will raise if the report in [0] was not published at the right time.
    """
    if quarterly_submissions[qtr][0]['type'] == '10-K':
        if qtr[1] == 1:
            return True
        else:
            return False
    elif quarterly_submissions[qtr][0]['type'] == '10-Q':
        if qtr[1] == 2 or qtr[1] == 3 or qtr[1] == 4:
            return True
        else:
            return False
    else:
        raise ValueError('[ERROR] Only 10-K and 10-Q supported.')


def check_report_continuity(quarterly_submissions, s):
    """
    Verify that the sequence of reports for the various qtr is 0-...0-1-...-1-0-...-0. In other words, once you are
    listed you only have one and only one report per quarter until you are delisted.

    :param quarterly_submissions:
    :param s:
    :return:
    """
    flag_success, qtr = find_first_listed_qtr(quarterly_submissions, s)
    # print("First quarter is", qtr)
    if not flag_success:
        # print('Returned False. Could not find the first quarter, they seem all empty.')
        return False
        # raise ValueError('Could not find the first quarter, they seem all empty.')
    
    # Now we start going through the submissions for each qtr. There shall only be one.
    idx = s['list_qtr'].index(qtr)
    for qtr in s['list_qtr'][idx:]:
        if len(quarterly_submissions[qtr]) == 1:
            # Verify that 10-K are published in Q1 only and 10-Q in Q2-3-4
            if check_report_type(quarterly_submissions, qtr):
                continue
            else:
                return False
        elif len(quarterly_submissions[qtr]) == 0:  # Has it been delisted?
            flag_is_delisted = is_permanently_delisted(quarterly_submissions, qtr, s)
            # print("Returned {} because flag_is_delisted is {}".format(flag_is_delisted, flag_is_delisted))
            return True if flag_is_delisted else False
        else:  # More than one report -> failed
            # print("Returned False because there is more than one report")
            return False
    # print("Returned True and everything is good")
    return True


def find_first_listed_qtr(quarterly_submissions, s):
    """
    Finds the first qtr for which the company published at least one report.

    :param quarterly_submissions: dictionary of submissions indexes by qtr
    :param s: Settings dictionary
    :return: bool for success and first qtr when the company was listed.
    """
    flag_listed = False
    for qtr in s['list_qtr']:
        if len(quarterly_submissions[qtr]) == 0:
            continue
        else:
            flag_listed = True
            break
    return flag_listed, qtr


def is_permanently_delisted(quarterly_submissions, qtr, s):
    """
    Check if a company is permanently delisted starting from a given qtr. This function is not great, I should have made
    a single function that finds the first qtr for which a company is listed and the qtr for which it became delisted,
    if ever.

    :param quarterly_submissions:
    :param qtr: a given qtr
    :param s: Settings dictionary
    :return: bool assessing whether or not it is permanently delisted after the given qtr
    """
    flag_permanently_delisted = True
    idx = s['list_qtr'].index(qtr)  # Index of the quarter that is empty
    for qtr in s['list_qtr'][idx:]:  # Check again and check the rest
        if len(quarterly_submissions[qtr]):
            flag_permanently_delisted = False
            break
    return flag_permanently_delisted


def dump_tickers_crsp(path_dump_file, tickers):
    """
    Dump all tickers to a file - should not be useful anymore.

    :param path_dump_file: path for csv dump
    :param tickers: all the tickers to dump.
    :return: void
    """
    with open(path_dump_file, 'w') as f:
        out = csv.writer(f)
        for ticker in tickers:
            out.writerow([ticker])
