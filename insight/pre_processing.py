import csv
import Load_MasterDictionary as LM
from tqdm import tqdm
from datetime import datetime
import time
import glob

"""I. Settings processing"""
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

"""II. Loading external database & curate the data"""
# 1. List of paths
def unique_cik(path_list):
    # Create a list of all the CIK
    all_cik = [int(e.split('/')[-1].split('_')[4]) for e in path_list]
    return list(set(all_cik))

def paths_to_cik_dict(file_list, cik):
    # Keys: CIK
    # Values: unorganized list of all paths for that CIK
    cik_dict = {key: [] for key in cik}
    for path in tqdm(file_list):
        split_path = path.split('/')
        cik_dict[int(split_path[-1].split('_')[4])].append(path)
    return cik_dict

def load_cik_path(s):
    file_list = glob.glob(s['path_stage_1_data']+'**/*.txt', recursive=True)
    print("[INFO] Loaded {:,} 10-X".format(len(file_list)))
    file_list = filter_cik_path(file_list, s)
    print("[INFO] Shrunk to {:,} {}".format(len(file_list), s['report_type']))
    #print("[INFO] Example:", file_list[0])
    unique_sec_cik = unique_cik(file_list)  # Get unique CIKs
    print("[INFO] Found {:,} unique CIK in master index".format(len(unique_sec_cik)))
    cik_path = paths_to_cik_dict(file_list, unique_sec_cik)  # Create a dict based on that
    print("[INFO] cik_path contains data on {:,} CIK numbers".format(len(cik_path)))
    
    return cik_path

def filter_cik_path(file_list, s):
    # Drop all the reports that are not of the considered type.
    filtered_file_list = [f for f in file_list if f.split('/')[-1].split('_')[1] in s['report_type']]
    return filtered_file_list 

# 2. CIK -> Ticker lookup table
def load_lookup(s):
    # Load the lookup table
    with open(s['path_cik_ticker_lookup']) as f:
        cik_to_ticker = dict()
        reader = csv.reader(f, delimiter='|')
        next(reader)  # Skip header
        for row in reader:
            cik_to_ticker[int(row[0])] = row[1]
    return cik_to_ticker

def intersection_sec_lookup(cik_path, lookup):
    # 1. Create unique list of keys
    unique_cik = set(cik_path.keys())
    unique_lookup = set(lookup.keys())
    
    # 2. Intersection
    intersection_cik = list(unique_cik & unique_lookup)
    
    # 3. Update both dictionaries (fwd and backwd propagation)
    inter_cik = {cik: cik_path[cik] for cik in intersection_cik}
    inter_lookup = {cik: lookup[cik] for cik in intersection_cik}
    
    return inter_cik, inter_lookup

# 3. Stock data
"""[TBR]
def load_path_stock_data(path_folder, lookup_tickers=None):
    # Load the ticker names in the stock database
    file_list = glob.glob(path_folder+'**/*.txt', recursive=True)
    list_tickers = [file_name[:-7] for file_name in file_list]
    unique_tickers = set(list_tickers)
    assert len(unique_tickers) == len(list_tickers)
    
    # Calculate intersection
    if ticker_lookup:
        unique_ticker_lookup = set(ticker_lookup)
        intersection = list(unique_tickers & unique_ticker_lookup)
"""

def load_stock_data(s, verbose=False):
    # Loading the data can take a while so better do it once only
    # Load the whole spreadsheet in RAM basically. Then one can count the unique tickers
    # And then their count.
    #path_ticker = os.path.join(s['path_ticker_data'], ticker.lower()+'.us.txt')
    
    # Count total number of data rows for tqdm
    with open(s['path_stock_database']) as f:
        nb_lines = sum(1 for line in f) -1  # Brutally fast, > 15 MM rows/second

    with open(s['path_stock_database']) as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_date = header.index("date")
        idx_ticker = header.index("TICKER")
        idx_closing = header.index("ASK")
        idx_outstanding_shares = header.index("SHROUT")
        data = dict()
        #selected_span = [(1900, 1), (2100, 1)] if selected_span == None else selected_span
        for row in tqdm(reader, total=nb_lines):  # Could change
            # 1. Check for emtpy rows (??)
            ticker = row[idx_ticker]
            closing_price = row[idx_closing]
            outstanding_shares = row[idx_outstanding_shares]
            if ticker == '' or closing_price == '' or outstanding_shares == '':
                continue
            
            # 2. Process the row
            closing_price = float(closing_price)
            market_cap = 1000*closing_price*int(outstanding_shares)
            date = row[idx_date]
            qtr = tuple((int(date[:4]), int(date[4:6])//3 + 1))
            
            #print(tuple([int(date[0]), qtr]))
            if s['time_range'][0] <= qtr <= s['time_range'][1]:  # Only data in time range
                if ticker not in data.keys():
                    data[ticker] = dict()
                ts = datetime.strptime(date, '%Y%m%d').date()
                #ts = tuple(row[0].split('-'))
                if s['type_daily_price'] == 'opening':
                    raise ValueError('[ERROR] Not supported. Get more data.')
                elif s['type_daily_price'] == 'high':
                    raise ValueError('[ERROR] Not supported. Get more data.')
                elif s['type_daily_price'] == 'low':
                    raise ValueError('[ERROR] Not supported. Get more data.')
                elif s['type_daily_price'] == 'closing':
                    data[ticker][ts] = [closing_price, market_cap]
                elif s['type_daily_price'] == 'average':
                    raise ValueError('[ERROR] Not supported. Get more data.')
                else:
                    raise ValueError('[ERROR] Unknown type of price for this stock')
    return data

def intersection_lookup_stock(lookup, stock):
    # 1. Create unique lists to compare
    unique_lookup = set(list(lookup.values()))
    unique_stock = set(list(stock.keys()))
    #assert len(unique_lookup) == len(lookup)
    
    # 2. Create intersection of tickers
    intersection_tickers = list(unique_lookup & unique_stock)
    print(len(intersection_tickers))
    
    # 3. Return a new intersection dictionary
    inter_lookup = {k: v for k, v in lookup.items() if v in intersection_tickers}
    inter_stock = {k: stock[k] for k in stock.keys() if k in intersection_tickers}
    
    return inter_lookup, inter_stock

# 4. Filter the CIK based on how many publications there are per quarter
def review_cik_publications(cik_path, s, verbose=False):
    """This function reviews all the CIK to make sure there is only 1 publication per qtr
    It provides a few hooks to correct issues but these have not been implemented.
    Around 10 % of the CIK seem to have problems at one point or another.
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

        """[TBR]
        # Analyse these submissions: are there holes?
        flag_not_listed = True  # Assume it is not listed first
        for idx in range(len(s['list_qtr'])):
        #for qtr, submissions in quarterly_submissions.items():
            qtr = s['list_qtr'][idx]
            submissions = quarterly_submissions[qtr]
            # 1. Spin the wheel as long as it not listed
            if len(submissions) == 0 and flag_not_listed:
                continue  # The company is not yet listed
            elif flag_not_listed:
                if qtr != s['list_qtr'][0] and verbose:
                    print("[INFO] CIK {} started to be listed in {}".format(cik, qtr))
                flag_not_listed = False

            # 2. Now ready to go.
            if len(submissions) == 0:  # No submissions!
                if idx < len(s['list_qtr'])-1:
                    next_qtr = s['list_qtr'][idx+1]
                    next_submissions = quarterly_submissions[next_qtr]
                    if len(next_submissions) == 0:
                        if verbose:
                            print("[WARNING] CIK {}'s last quarter was {}".format(cik, qtr))
                        break  # Got delisted, we are done here but we keep it!
                    elif len(next_submissions) == 1:  # 
                        print("[ERROR] CIK {} is missing a report in {}".format(cik, qtr))
                        cik_to_delete.append(cik)
                        break
                    elif len(next_submissions) == 2:
                        if verbose:
                            print("[WARNING] CIK {} published late in quarter {}. Moving to the previous quarter"
                              .format(cik, qtr))
                        cik_to_delete.append(cik)
                        break
                        #<move first report to previous qtr>  # Move file in the database? Maybe, if not too many of them
                    else:
                        print("[ERROR] Too many upcoming reports for CIK {} in {}.".format(cik, qtr))
                        cik_to_delete.append(cik)
                        break
                else:
                    pass  # We cannot verify further, so we pass
            elif len(submissions) == 1:
                # Should make sure it is from the right quarter though
                pass
            elif len(submissions) == 2:
                if idx < len(s['list_qtr'])-1:
                    next_qtr = s['list_qtr'][idx+1]
                    next_submissions = quarterly_submissions[next_qtr]
                    if len(next_submissions) == 0:  # We are just early
                        if verbose:
                            print("[WARNING] CIK {} published early in quarter {}. Moving to next quarter".format(cik, qtr))
                        #<move first report to next qtr>
                        cik_to_delete.append(cik)
                        break
                    else:
                        print("[ERROR] Too many reports for CIK {} in {} - unresolvable.".format(cik, qtr))
                        cik_to_delete.append(cik)
                        break
                else:
                    pass  # We cannot verify further, so we pass

            else:
                print("[ERROR] Too many reports for CIK {} in {}: {}.".format(cik, qtr, len(submissions)))
                # print(submissions)
                #assert 0
                cik_to_delete.append(cik)
                break
        """
        
        
        
    
    # Create a subset of cik_dict based on the cik not faulty
    print()
    print("[INFO] {} CIKs caused trouble".format(len(cik_to_delete)))
    cik_dict = {k: v for k, v in cik_path.items() if k not in cik_to_delete}
    
    return cik_dict

def check_report_continuity(quarterly_submissions, s):
    # Verify that the sequence is 0-...0-1-...-1-0-...-0
    flag_success, qtr = find_first_listed_qtr(quarterly_submissions, s)
    #print("First quarter is", qtr)
    if not flag_success:
        #print('Returned False. Could not find the first quarter, they seem all empty.')
        return False
        #raise ValueError('Could not find the first quarter, they seem all empty.')
    
    # Now we start going through the reports. There shall only be one
    idx = s['list_qtr'].index(qtr)
    for qtr in s['list_qtr'][idx:]:
        if len(quarterly_submissions[qtr]) == 1:
            continue
        elif len(quarterly_submissions[qtr]) == 0:  # Has it been delisted?
            flag_is_delisted = is_permanently_delisted(quarterly_submissions, qtr, s)
            #print("Returned {} because flag_is_delisted is {}".format(flag_is_delisted, flag_is_delisted))
            return True if flag_is_delisted else False
        else:  # More than one report -> failed
            #print("Returned False because there is more than one report")
            return False
    #print("Returned True and everything is good")
    return True

def find_first_listed_qtr(quarterly_submissions, s):
    flag_listed = False
    for qtr in s['list_qtr']:
        if len(quarterly_submissions[qtr]) == 0:
            continue
        else:
            flag_listed = True
            break
    return flag_listed, qtr

def is_permanently_delisted(quarterly_submissions, qtr, s):
    flag_permanently_delisted = True
    idx = s['list_qtr'].index(qtr)  # Index of the quarter that is empty
    for qtr in s['list_qtr'][idx:]:  # Check again and check the rest
        if len(quarterly_submissions[qtr]):
            flag_permanently_delisted = False
            break
    return flag_permanently_delisted

def test_check_report_continuity():
    s = {'list_qtr': [
    (2010, 1),
    (2010, 2),
    (2010, 3),
    (2010, 4),
    (2011, 1),
    (2011, 2),
    (2011, 3),
    (2011, 4),
    (2012, 1),
    (2012, 2),
    (2012, 3),
    (2012, 4)
    ]}
    qs1 = {
        (2010, 1): [{'type': '10-K', 'published': (2010, 2, 26), 'qtr': (2010, 1)}], 
        (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}], 
        (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}], 
        (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}], 
        (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}], 
        (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}], 
        (2011, 3): [{'type': '10-Q', 'published': (2011, 8, 5), 'qtr': (2011, 3)}], 
        (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}], 
        (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}], 
        (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}], 
        (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}], 
        (2012, 4): [{'type': '10-Q', 'published': (2012, 11, 2), 'qtr': (2012, 4)}]
    }
    qs2 = {
        (2010, 1): [],
        (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}],
        (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}], 
        (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}], 
        (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}], 
        (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}], 
        (2011, 3): [{'type': '10-Q', 'published': (2011, 8, 5), 'qtr': (2011, 3)}], 
        (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}], 
        (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}], 
        (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}], 
        (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}], 
        (2012, 4): [{'type': '10-Q', 'published': (2012, 11, 2), 'qtr': (2012, 4)}]
    }
    qs3 = {
        (2010, 1): [{'type': '10-K', 'published': (2010, 2, 26), 'qtr': (2010, 1)}], 
        (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}], 
        (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}], 
        (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}], 
        (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}], 
        (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}], 
        (2011, 3): [{'type': '10-Q', 'published': (2011, 8, 5), 'qtr': (2011, 3)}], 
        (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}], 
        (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}], 
        (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}], 
        (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}],
        (2012, 4): []
    }
    qs4 = {
        (2010, 1): [{'type': '10-K', 'published': (2010, 2, 26), 'qtr': (2010, 1)}], 
        (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}], 
        (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}], 
        (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}], 
        (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}], 
        (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}], 
        (2011, 3): [], 
        (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}], 
        (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}], 
        (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}], 
        (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}],
        (2012, 4): [{'type': '10-Q', 'published': (2012, 11, 2), 'qtr': (2012, 4)}]
    }
    qs5 = {
        (2010, 1): [{'type': '10-K', 'published': (2010, 2, 26), 'qtr': (2010, 1)}], 
        (2010, 2): [{'type': '10-Q', 'published': (2010, 5, 3), 'qtr': (2010, 2)}], 
        (2010, 3): [{'type': '10-Q', 'published': (2010, 8, 6), 'qtr': (2010, 3)}], 
        (2010, 4): [{'type': '10-Q', 'published': (2010, 11, 5), 'qtr': (2010, 4)}], 
        (2011, 1): [{'type': '10-K', 'published': (2011, 3, 1), 'qtr': (2011, 1)}], 
        (2011, 2): [{'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}, {'type': '10-Q', 'published': (2011, 5, 6), 'qtr': (2011, 2)}], 
        (2011, 3): [{'type': '10-Q', 'published': (2011, 8, 5), 'qtr': (2011, 3)}], 
        (2011, 4): [{'type': '10-Q', 'published': (2011, 11, 4), 'qtr': (2011, 4)}], 
        (2012, 1): [{'type': '10-K', 'published': (2012, 2, 29), 'qtr': (2012, 1)}], 
        (2012, 2): [{'type': '10-Q', 'published': (2012, 5, 4), 'qtr': (2012, 2)}], 
        (2012, 3): [{'type': '10-Q', 'published': (2012, 8, 3), 'qtr': (2012, 3)}], 
        (2012, 4): [{'type': '10-Q', 'published': (2012, 11, 2), 'qtr': (2012, 4)}]
    }
    qs6 = {
        (2010, 1): [], 
        (2010, 2): [], 
        (2010, 3): [], 
        (2010, 4): [], 
        (2011, 1): [], 
        (2011, 2): [], 
        (2011, 3): [], 
        (2011, 4): [], 
        (2012, 1): [], 
        (2012, 2): [], 
        (2012, 3): [], 
        (2012, 4): []
    }
    list_qs = [(True, qs1), (True, qs2), (True, qs3), (False, qs4), (False, qs5), (False, qs6)]
    for qs in list_qs:
        assert check_report_continuity(qs[1], s) == qs[0]
    return True
# test_check_report_continuity()

"""[DEBUG]"""
# Dump all tickers to a file - should not be useful anymore
def dump_tickers_crsp(path_dump_file, tickers):
    with open(path_dump_file, 'w') as f:
        out = csv.writer(f)
        for ticker in tickers:
            out.writerow([ticker])

def normalize_texts(current_text, previous_text):
    """Remove all extra spaces, \n and \t that could be left and substitute by a single whitespace.
    """
    return " ".join(current_text.split()), " ".join(previous_text.split())

def load_master_dictionary(path):
    lm_dictionary = LM.load_masterdictionary(path, True)
    return lm_dictionary
