import pandas as pd
from datetime import datetime
from secScraper import qtrs
import csv
from tqdm import tqdm
from scipy.stats.mstats import winsorize


def make_quintiles(x, s, winsorize=0.01):
    """
    Winsorize input data (default is 1% on each end). Create quintiles based on a list of values.
    Used to create quintiles based on the scores of each report.

    :param x: list of lists. Only interested in a single column though.
    :param s: Settings dictionary
    :return: quintiles as a list of list
    """
    # x is (cik, score, nb_share_unbalanced, nb_share_balanced)
    # Create labels and bins of the same size
    # labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']  # Not using that anymore
    quintiles = {l: [] for l in s['bin_labels']}
    
    # _, input_data, _, _ = zip(*x)
    input_data = x
    input_data = pd.Series(input_data)
    mapping = pd.qcut(input_data.rank(method='first'), s['bin_count'], labels=False)
    print(mapping)
    for idx_input, idx_output in enumerate(mapping):
        quintiles[s['bin_labels'][idx_output]].append(x[idx_input])
    return quintiles


def get_share_price(cik, qtr, lookup, stock_data, verbose=False):
    """
    Get the price of a share.

    :param cik: CIK
    :param qtr: qtr
    :param lookup: lookup dict
    :param stock_data: dict of the stock data
    :param verbose: self explanatory
    :return: share_price, market_cap, flag_price_found
    """
    ticker = lookup[cik]
    # print("cik/ticker", cik, ticker)
    qtr_start_date = "{}{}{}".format(str(qtr[0]), str((qtr[1]-1)*3+1).zfill(2), '01')
    qtr_start_date = datetime.strptime(qtr_start_date, '%Y%m%d').date()
    
    # Find the first trading day after the beginning of the quarter.
    # Sanity check: is there a price available?
    try:
        time_range = list(stock_data[ticker].keys())
    except:
        return 1, 1, False
    
    if verbose:
        print("Prices cover {} to {} and we are looking for {}".format(time_range[0], time_range[-1], qtr_start_date))
    
    share_price = 1
    market_cap = 1
    for _ in range(7):
        try:
            share_price, market_cap = stock_data[ticker][qtr_start_date]
            if verbose:
                print("[INFO] Settled for", qtr_start_date)
            break
        except KeyError:
            qtr_start_date = qtr_start_date.strftime('%Y%m%d')
            day = str(int(qtr_start_date[7]) + 1)
            qtr_start_date = qtr_start_date[:7] + day
            qtr_start_date = datetime.strptime(qtr_start_date, '%Y%m%d').date()
    
    flag_price_found = False if share_price == 1 and market_cap == 1 else True
    return share_price, market_cap, flag_price_found


def remove_cik_without_price(pf_scores, lookup, stock_data, s, verbose=False):
    """
    So far, we have not checked if we had a stock price available for that all CIK.
    This function removes the CIK for which we have no price. < 10% of them are dropped.

    :param pf_scores: dict
    :param lookup: lookup dict
    :param stock_data: dict of the stock data
    :param s: Settings dictionary
    :param verbose:
    :return: outputs more stuff
    """
    for m in s['metrics'][:-1]:
        for mod_bin in s['bin_labels']:
            for qtr in s['list_qtr'][s['lag']:]:
                cik_not_found = []
                for entry in pf_scores[m][mod_bin][qtr]:
                    cik = entry[0]
                    _, _, flag_price_found = get_share_price(cik, qtr, lookup, stock_data)
                    if not flag_price_found:
                        cik_not_found.append(cik)
                pf_scores[m][mod_bin][qtr] = [e for e in pf_scores[m][mod_bin][qtr] if e[0] not in cik_not_found]
                if verbose:
                    print("[INFO] Removed {}/{} CIK".format(len(cik_not_found), len(pf_scores[m][mod_bin][qtr])))
                if len(pf_scores[m][mod_bin][qtr]) == 0:
                    raise ValueError("[ERROR] Nothing is left!")
                # elif len(pf_scores[m][mod_bin][qtr]) <= 20:
                    # print(m, mod_bin, qtr)
    return pf_scores


def get_pf_value(pf_scores, m, mod_bin, qtr, lookup, stock_data, s):
    """
    Get the value of a portfolio.

    :param pf_scores: dict containing all the scores for all companies
    :param m: metric
    :param mod_bin: bin considered
    :param qtr: qtr
    :param lookup: lookup dict
    :param stock_data: dict of the stock data
    :param s: Settings dictionary
    :return:
    """
    # Whole bin to sum -> need the balanced and unbalanced value
    unbalanced_value = 0
    balanced_value = 0
    for share in pf_scores[m][mod_bin][qtrs.previous_qtr(qtr, s)]:  # Previous pf...
        cik = share[0]
        share_price, market_cap, flag_price_found = get_share_price(cik, qtr, lookup, stock_data)
            
        unbalanced_value += share_price*share[2]
        balanced_value += share_price*share[3]
    return unbalanced_value, balanced_value


def calculate_portfolio_value(pf_scores, pf_values, lookup, stock_data, s, balancing='balanced', verbose=False):
    """
    Calculate the value of a portfolio, in equal weight and balanced weight (by market cap) mode. The value is written
    to pf_scores (in the inputs).

    :param pf_scores: dict containing all the scores for all companies
    :param pf_values: dict containing the value of a portfolio
    :param lookup: lookup dict
    :param stock_data: dict of the stock data
    :param s: Settings dictionary
    :return: dict pf_scores
    """
    for m in s['metrics'][:-1]:
        for mod_bin in s['bin_labels']:
            for qtr in s['list_qtr'][s['lag']:]: 
                # Here we have an array of arrays [cik, score, nb_shares_unbalanced, nb_shares_balanced]
                # 1. Unbalanced portfolio: everyone get the same amount of shares
                # 1.1 Get number of CIK
                #print(pf_scores)
                nb_cik = len(pf_scores[m][mod_bin][qtr])  # Nb of CIK in that bin
                total_mc = 0
                
                # Update pf value!
                if qtr == s['list_qtr'][s['lag']]:
                    pf_value = s['pf_init_value']
                else:
                    pf_value_unbalanced, pf_value_balanced = get_pf_value(pf_scores, m, mod_bin, qtr, lookup, stock_data, s)
                    if balancing == 'balanced':
                        pf_value = pf_value_balanced
                    elif balancing == 'unbalanced':
                        pf_value = pf_value_unbalanced
                    else:
                        raise ValueError('[ERROR] Balancing method unknown.')
                    
                    # print(pf_value_unbalanced, pf_value_balanced)
                    pf_values[m][mod_bin][qtr][0] = pf_value
                    pf_value *= (1 - pf_values[m][mod_bin][qtr][1])  # Apply a tax rate
                    pf_values[m][mod_bin][qtr][2] = pf_value  # This is what will be used to buy new shares
                
                # 1.2 With that amount, re-populate the pf with the new recommendation 
                # (including last qtr even if useless)
                nb_errors = 0
                for idx in range(nb_cik):
                    cik = pf_scores[m][mod_bin][qtr][idx][0]
                    price, market_cap, flag_price_found = get_share_price(cik, qtr, lookup, stock_data)
                    nb_errors += 0 if flag_price_found else 1
                if nb_errors:
                    if verbose:
                        print("Found", nb_errors, "errors out of", nb_cik)
                nb_cik -= nb_errors
                if nb_cik < 0:
                    raise ValueError("WTF - No CIK left after checking for the stock data availability?")
                    
                for idx in range(nb_cik):
                    cik = pf_scores[m][mod_bin][qtr][idx][0]
                    price, market_cap, flag_price_found = get_share_price(cik, qtr, lookup, stock_data)
                    if not flag_price_found:
                        continue  # We skip it
                    total_mc += market_cap
                    pf_scores[m][mod_bin][qtr][idx][2] = (pf_value/nb_cik)/price  # Unbalanced nb of shares
                    pf_scores[m][mod_bin][qtr][idx][3] = (pf_value*market_cap)/price  # Balanced nb shares
                
                # 1.3 Normalize the balanced value by the total market cap
                for idx in range(nb_cik):
                    pf_scores[m][mod_bin][qtr][idx][3] /= total_mc
    return pf_scores


def dump_master_dict(master_dict, s):
    # path = '/home/alex/Desktop/Insight project/Database/dump_master_dict.csv'
    with open(s['path_dump_master_dict'], 'w') as f:
        out = csv.writer(f, delimiter=';')
        header = ['METRIC', 'QUARTER', 'QUINTILE', 'CIK', 'SCORE']
        out.writerow(header)
        
        # Main writing loop
        for m in tqdm(s['metrics'][:-1]):
            for qtr in s['list_qtr'][s['lag']:]:
                for l in s['bin_labels']:
                    for entry in master_dict[m][qtr][l]:
                        out.writerow([m, qtr, l, entry[0], entry[1]])


def dump_pf_values(pf_values, s):
    # path = '/home/alex/Desktop/Insight project/Database/dump_master_dict.csv'
    with open(s['path_dump_pf_values'], 'w') as f:
        out = csv.writer(f, delimiter=';')
        header = ['METRIC',  'QUINTILE', 'QUARTER', 'PF_VALUE', 'TAX_RATE', 'PF_VALUE_POST_TAX']
        out.writerow(header)
        
        # Main writing loop
        for m in tqdm(s['metrics'][:-1]):
            for l in s['bin_labels']:
                for qtr in s['list_qtr'][s['lag']:]:
                    out.writerow([m, qtr, l, *pf_values[m][l][qtr]])


def dump_cik_scores(cik_scores, s):
    # path = '/home/alex/Desktop/Insight project/Database/dump_master_dict.csv'
    with open(s['path_dump_cik_scores'], 'w') as f:
        out = csv.writer(f, delimiter=';')
        header = ['CIK',  'QTR', 'METRIC', 'SCORE']
        out.writerow(header)
        
        # Main writing loop
        for cik in tqdm(cik_scores.keys()):
            for qtr in s['list_qtr'][s['lag']:]:
                for m in s['metrics']:
                    try:
                        out.writerow([cik, qtr, m, cik_scores[cik][qtr][m]])
                    except KeyError:  # There is no data for this qtr, CIK not listed/delisted
                        continue
