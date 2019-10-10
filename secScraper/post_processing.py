import pandas as pd
from datetime import datetime
from secScraper import qtrs
import csv
from tqdm import tqdm
from scipy.stats.mstats import winsorize
import numpy as np
import copy


"""[TBR] Legacy version that did not work so well
def make_quintiles(x, s, winsorize=0.01):
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
"""
def make_quintiles(qtr_data, s, winsorize = 0.01, verbose=False):
    # 1. Isolate the non zero CIKs
    non_zero_ciks = {cik: v for cik, v in qtr_data.items() if v != {}}
    if verbose:
        print("[INFO] Non zero ciks: {}/{}".format(len(non_zero_ciks), len(qtr_data)))
    sorted_ciks = sorted(non_zero_ciks, key=lambda x: non_zero_ciks[x]['total'] if non_zero_ciks[x] != {} else 0)

    # 2. Winsorize to remove outliers
    start = round(len(sorted_ciks)*winsorize)  # Output an int
    end = len(sorted_ciks) - start  # Stays an int
    sorted_ciks = sorted_ciks[start:end]
    if verbose:
        print("[INFO] Left with {}/{} elements after winsorizing".format(len(sorted_ciks), len(qtr_data)))

    # 3. Make quintiles/deciles
    splits = np.linspace(0, len(sorted_ciks), s['bin_count']+1, endpoint=True, dtype=np.int)
    quintiles = dict()
    # Make sure bins are in increasing order: Q1 -> Q5. Otherwise, sorted_ciks' order needs to be reversed.
    assert int(s['bin_labels'][-1][1:]) > int(s['bin_labels'][0][1:])
    for idx, l in enumerate(s['bin_labels']):
        quintiles[l] = {cik: qtr_data[cik] for cik in sorted_ciks[splits[idx]:splits[idx+1]]}
    
    # Sanity check: Verify that the quintiles worked as expected. O(N**2).
    for idx in range(1, len(s['bin_labels'])):
        for cik in quintiles[s['bin_labels'][idx]]:
            for cik_previous in quintiles[s['bin_labels'][idx-1]]:
                try:
                    assert qtr_data[cik]['total'] >= qtr_data[cik_previous]['total']
                except:
                    print(cik, qtr_data[cik])
                    print(cik_previous, qtr_data[cik_previous])
                    raise
     
    return quintiles


def metrics_correlation(metric_scores, s):
    data = []
    for m in s['diff_metrics']:
        flattened_metric = []
        for qtr in s['list_qtr'][s['lag']:]:
            for cik in metric_scores[m][qtr]:
                if metric_scores[m][qtr][cik] != {}:
                    flattened_metric.append(metric_scores[m][qtr][cik]['total'])
        data.append(flattened_metric)
    df = pd.DataFrame(zip(*data), columns=s['diff_metrics'])
    return df


def create_metric_scores(cik_scores, lookup, stock_data, s):
    pnf = []
    metric_scores = {m: {qtr: {cik: {} for cik in cik_scores} for qtr in s['list_qtr'][s['lag']:]} for m in s['metrics']}
    for cik in tqdm(cik_scores):
        for qtr in cik_scores[cik]:
            _, _, flag_price_found = get_share_price(cik, qtr, lookup, stock_data)
            if not flag_price_found:
                print("[WARNING] There is no stock data for {} during {}".format(cik, qtr))
                pnf.append(cik)
                break  # That CIK from cik_scores will be left unpopulated and subsequently discarded
            sections = [section for section in cik_scores[cik][qtr] if section != '0' and section != 'total']
            for section in sections:
                for m in s['metrics']:
                    metric_scores[m][qtr][cik][section] = cik_scores[cik][qtr][section][m]
                    metric_scores[m][qtr][cik]['total'] = cik_scores[cik][qtr]['total'][m]
                    # metric_scores[m][qtr][cik]['0'] = cik_scores[cik][qtr]['0']
    print("Unique cik", set(pnf))
    return metric_scores


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


def buy_all_pf(qtr, funds, pf, lookup, stock_data, method):
    """
    Allocate a given amount of money to a quarterly portfolio. Method can be balanced (weighted by market cap) or unbalanced
    (each stock gets the same amount of money).
    """
    assert type(funds) == float
    nb_cik = len(pf)  # Nb of CIK in that bin
    sum_market_caps = 0
    
    # 1. Update the share price/market_cap for everyone
    for cik in pf:
        ticker = lookup[cik]
        share_price, market_cap, flag_price_found = get_share_price(cik, qtr, lookup, stock_data)
        pf[cik] = [ticker, share_price, market_cap, 0, 0, 0]  # [..., share_count, $, $/funds]
        sum_market_caps += market_cap  # Needed to balance pf
    
    # 2. Second pass where we split the funds accordingly
    for cik in pf:
        # 2.1. Calculate the $/shares to purchase
        share_price = pf[cik][1]
        market_cap = pf[cik][2]
        if method == 'balanced':
            value = funds*(market_cap/sum_market_caps)  # $ amount depends on your mc
        elif method == 'unbalanced':
            value = funds*(1/nb_cik)  # $ amount is equal for all stocks
        share_count = value/share_price
        
        # 2.2. Buy the shares -> populate the pf line with the new values
        # pf[cik][3:] = [share_count, $, $/funds]
        # $ is how much funds we have put in that stock
        # $/funds is the ratio of funds in that stock to the total value of the pf
        pf[cik][3:] = share_count, value, value/funds
    
    return pf


def sell_all_pf(qtr, pf, lookup, stock_data):
    """
    Sell all the stocks in a portfolio. In practice, we just collect the value of the pf with the new updated share
    prices."""
    sum_stock_values = 0
    sum_market_caps = 0
    # I. First pass to update the stock price and the consequent value held 
    for cik in pf:
        # 1. Update the share price for that CIK/ticker
        share_price, market_cap, flag_price_found = get_share_price(cik, qtr, lookup, stock_data)
        
        # 2. Update the value of that line given the new share_price
        share_count = pf[cik][3]  # This is invariant at that stage!
        updated_value = share_price*share_count
        pf[cik][1] = share_price  # Update stock price
        pf[cik][2] = market_cap  # Update market cap
        pf[cik][4] = updated_value  # Update value of that line
        sum_market_caps += market_cap
        
        # 3. Add that line to the total
        sum_stock_values += updated_value
    
    # II. Re-update all the market_cap ratios with the new value
    for cik in pf:
        updated_value = pf[cik][4]  # Updated value of that line in our pf
        pf[cik][5] = updated_value/sum_stock_values  # This ratio might have gone up or done.
        # pf[cik][5] can be useful to see what is our biggest exposure after a qtr. Might have changed.
    
    return pf, sum_stock_values


def initialize_portfolio(metric_scores, s):
    # Introduce the pf_values
    pf_values = {m: {qtr: {} for qtr in metric_scores[m]} for m in metric_scores}
    # This first pass populates the pf with the relevant ciks.
    for m in s['metrics']:
        for idx, qtr in enumerate(s['list_qtr'][s['lag']:]):
            if idx == 0:  # qtr == s['list_qtr'][s['lag']]
                data = {l: {cik: [] for cik in metric_scores[m][qtr][l]} for l in s['bin_labels']}
                pf_values[m][qtr]['incoming_compo'] = {}  # Not useful. Will be copied later
                pf_values[m][qtr]['incoming_value'] = {l: s['pf_init_value'] for l in s['bin_labels']}  # Not really useful
                pf_values[m][qtr]['new_value'] = {l: s['pf_init_value'] for l in s['bin_labels']}
                pf_values[m][qtr]['new_compo'] = data

            elif 0 < idx:  # not the first qtr
                data = {l: {cik: [] for cik in metric_scores[m][qtr][l]} for l in s['bin_labels']}
                previous_qtr = s['list_qtr'][s['lag']:][idx-1]
                # No need to populate the incoming_value. They will be calculated gradually
                # pf_values[m][qtr]['incoming_compo'] = copy.deepcopy(pf_values[m][previous_qtr]['new_compo'])
                pf_values[m][qtr]['incoming_compo'] = {}
                pf_values[m][qtr]['incoming_value'] = {l: 0 for l in s['bin_labels']}
                pf_values[m][qtr]['new_value'] = {l: 0 for l in s['bin_labels']}
                pf_values[m][qtr]['new_compo'] = data  # Common with idx == 0 case
    return pf_values


def build_portfolio(pf_values, lookup, stock_data, s):
    # Populate the pf for each cik and get overall values
    for m in s['metrics']:
        for idx, qtr in enumerate(s['list_qtr'][s['lag']:]):
            if qtr == s['list_qtr'][s['lag']]:  # First quarter
                # Perform a new_compo only. Then copy to incoming_compo (not really necessary though)
                for l in s['bin_labels']:
                    quintile_funds = pf_values[m][qtr]['new_value'][l]
                    assert quintile_funds == 100.0
                    pf = buy_all_pf(qtr, quintile_funds, pf_values[m][qtr]['new_compo'][l], lookup, stock_data, s['pf_balancing'])
                    pf_values[m][qtr]['new_compo'][l] = pf
                    pf_values[m][qtr]['incoming_compo'][l] = copy.deepcopy(pf)
                #print(pf_values[m][qtr]['new_compo'])
                # assert 0
            else:
                for stage in ['incoming_compo', 'new_compo']:
                    if stage == 'incoming_compo':
                        for l in s['bin_labels']:
                            previous_qtr = s['list_qtr'][s['lag']:][idx-1]
                            pf_values[m][qtr][stage][l] = copy.deepcopy(pf_values[m][previous_qtr]['new_compo'][l])
                            pf, quintile_funds = sell_all_pf(qtr, pf_values[m][qtr][stage][l], lookup, stock_data)
                            pf_values[m][qtr][stage][l] = pf  # Update the incoming compo with the new prices
                            pf_values[m][qtr]['incoming_value'][l] = quintile_funds
                            pf_values[m][qtr]['new_value'][l] = quintile_funds*(1-s['tax_rate'])
                    elif stage == 'new_compo':  # Take all the new_values and buy yourself a pf
                        for l in s['bin_labels']:
                            quintile_funds = pf_values[m][qtr]['new_value'][l]
                            pf = buy_all_pf(qtr, quintile_funds, pf_values[m][qtr][stage][l], lookup, stock_data, s['pf_balancing'])
                    else:
                        raise ValueError('[ERROR] Stage {} undefined.'.format(stage))
    return pf_values


def check_pf_value(pf_values, s):
    # Sanity checks
    # 1. Sum of all money invested should equate the funds
    for m in s['metrics']:
        for qtr in s['list_qtr'][s['lag']:]:
            for stage in ['incoming_compo', 'new_compo']:
                for l in s['bin_labels']:
                    if stage == 'incoming_compo':
                        declared_value = pf_values[m][qtr]['incoming_value'][l]
                    elif stage == 'new_compo':
                        declared_value = pf_values[m][qtr]['new_value'][l]
                    calculated_pf_value = 0
                    for cik in pf_values[m][qtr][stage][l]:
                        calculated_pf_value += pf_values[m][qtr][stage][l][cik][4]
                    try:
                        assert declared_value - s['epsilon'] < calculated_pf_value < declared_value + s['epsilon']
                    except:
                        print([m],[qtr],[stage],[l])
                        print(pf_values[m][qtr]['incoming_value'])
                        print(declared_value, calculated_pf_value)
                        raise
    return True


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
