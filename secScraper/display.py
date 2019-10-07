import matplotlib
# matplotlib.use('Qt5Agg')  # Use another backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


def run_from_ipython():
    """
    Check if the script is run from command line or from a Jupyter Notebook.

    :return: bool that is True if run from Jupyter Notebook
    """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def histogram_width(qtr_metric_result, metrics, s):
    center = []
    
    for qtr in s['list_qtr'][s['lag']:]:
        # print("Values", qtr_metric_result)
        center.append(matplotlib.dates.date2num(qtr_metric_result[qtr]['0']['published']))
    center = np.array(center)
    diff_center = np.diff(center)  # Distance between two groups of histograms
    r = s['histogram_date_span_ratio']
    return int(min(diff_center)*r/len(metrics))  # Based on the min of that distance


def diff_vs_stock(qtr_metric_result, ticker_data, ticker, s, method='diff'):
    """
    Display the calculated data for a given ticker across the time_range that was specified.

    :param qtr_metric_result: Dictionary containing the data to plot
    :param ticker_data: Daily stock value for the ticker considered
    :param ticker: Company ticker on the US stock exchange
    :param s: Settings dictionary
    :param method: Specify if a difference between two reports or an analysis of each report.
    :return: void
    """
    # 0. Select the type of plot
    if method == 'diff':
        metrics = s['diff_metrics']
    elif method == 'sentiment':
        metrics = s['sing_metrics']
    else:
        raise ValueError('[ERROR] Method unknown')
    

    # 1. Display the stock data
    lists = sorted(ticker_data.items())  # sorted by key, return a list of tuples
    benchmark_x, data_y = zip(*lists)  # unzips the dates & financial data
    #benchmark_x = matplotlib.dates.date2num(benchmark_x)
    # At this point, y is a list of lists. We need to extract the price from it.
    benchmark_y, market_cap = zip(*data_y)  # Will crash if len(y) > 2? Or ignore the rest?
    benchmark = zip(benchmark_x, benchmark_y)
    
    # 2. Display the histogram
    width = histogram_width(qtr_metric_result, metrics, s)

    # Now we plot all the histograms, one metric at a time
    
    metric_data = list()
    for ii, m in enumerate(metrics):
        # Get the data for each quarter on that metric
        x = list()
        y = list()
        for idx, qtr in enumerate(qtr_metric_result):
            if len(qtr_metric_result[qtr]) == 0:
                print("[ERROR] No data for qtr {}?".format(qtr))
                continue
            else:
                center = matplotlib.dates.date2num(qtr_metric_result[qtr]['0']['published'])
                if method == 'diff':
                    position = center - width*(len(metrics))/2 + width*ii
                elif method == 'sentiment':
                    position = center
                x.append(matplotlib.dates.num2date(position).date())
                y.append(qtr_metric_result[qtr][m])

        metric_data.append(zip(x, y))
    
    return benchmark, metric_data
    


def plot_diff_vs_stock(benchmark, metric_data, ticker, s, method='diff'):
    # 0. Select the type of plot
    if method == 'diff':
        metrics = s['diff_metrics']
    elif method == 'sentiment':
        metrics = s['sing_metrics']
    else:
        raise ValueError('[ERROR] Method unknown')
        
    fig, ax1 = plt.subplots(figsize=(15, 5))
    #benchmark_x, benchmark_y = zip(*benchmark)
    
    ax1.plot_date(*zip(*benchmark), ms=1)
    ax1.set_ylabel('Stock price [$]', fontsize=16)
    ax1.set_xlabel('Historical data', fontsize=16)
    ax1.set_title('Similarity scores vs daily {} price for ticker {}'
                  .format(s['type_daily_price'], ticker), fontsize=20)
    
    # Make second axis plot
    ax2 = ax1.twinx()
    for idx, data in enumerate(metric_data):
        x, y = zip(*data)
        plt.bar(x, y, label=metrics[idx], width=6, linestyle='-')
    
    if method == 'diff':
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Metric similarity [0-1]', fontsize=16)
    elif method == 'sentiment':
        abs_max = max(abs(max(y)), abs(min(y)))
        ax2.set_ylim([-abs_max, abs_max])
        ax2.set_ylabel('Composite sentiment [0-1]', fontsize=16)

    ax2.get_xaxis().set_visible(False)
    
    plt.legend()
    plt.savefig(os.path.join(s['path_output_folder'], '{}_{}_View_{}.png'.format(x[0].strftime('%Y%m%d'), x[-1].strftime('%Y%m%d'), ticker)))
    if run_from_ipython():
        plt.show()
    else:
        plt.close(fig)


def diff_vs_benchmark(pf_values, index_name, index_data, diff_method, s, norm_by_index=False):
    """
    Plot a portfolio vs an index.

    :param pf_values: Value of the portfolio over time.
    :param index_name: Name of the index.
    :param index_data: Daily value of the index.
    :param s: Settings dictionary.
    :return: void
    """
    # fig = plt.figure(figsize=(10, 5))

    """Display an index"""
    benchmark_x = []
    benchmark_y = []
    for qtr in s['list_qtr'][s['lag']:]:
        qtr_start_date = "{}{}{}".format(str(qtr[0]), str((qtr[1]-1)*3+1).zfill(2), '01')
        qtr_start_date = datetime.strptime(qtr_start_date, '%Y%m%d').date()
        # days, _ = zip(*index_data[index_name])
        days, prices = zip(*index_data[index_name].items())
        for _ in range(7):
            try:
                idx = days.index(qtr_start_date)
                break
            except ValueError:  # The stock exchange was closed that day. Move to the next one.
                qtr_start_date = qtr_start_date.strftime('%Y%m%d')
                day = str(int(qtr_start_date[7]) + 1)
                qtr_start_date = qtr_start_date[:7] + day
                qtr_start_date = datetime.strptime(qtr_start_date, '%Y%m%d').date()
        try:
            benchmark_x.append(qtr_start_date)
            benchmark_y.append(prices[idx][0])  # Only one entry per timestamp
            
        except KeyError:
            raise KeyError('[ERROR] The stock exchange should not have been shut down for more than 7 days.')
    benchmark_y = [value*s['pf_init_value']/benchmark_y[0] for value in benchmark_y]
    
    """Norm by index or not?"""
    if norm_by_index:
        norm = benchmark_y
        benchmark_y = [-s['pf_init_value']]*len(norm)  # Nullify the index data
    else:
        norm = [1]*len(benchmark_y)
        # plt.plot_date(benchmark_x, benchmark_y, label=index_name, linestyle='-.', linewidth=2, ms=10, marker=',')
    benchmark = zip(benchmark_x, benchmark_y)  # Zip for plotting
    
    """Display all the quintiles/deciles"""
    # bin_data = list()
    bin_data = dict()
    for l in s['bin_labels']:
        x = list()
        y = list()
        for qtr in s['list_qtr'][s['lag']:]:
            start = "{}{}{}".format(str(qtr[0]), str(((qtr[1])-1)*3+1).zfill(2), '01')
            x.append(datetime.strptime(start, '%Y%m%d').date())
            y.append(pf_values[diff_method][l][qtr][0])
        y = [qx_value/benchmark_value for qx_value, benchmark_value in zip(y, norm)]
        # plt.plot_date(x, y, label=l, linestyle='-')
        #single_bin_data = zip(x, y)
        #bin_data.append(single_bin_data)
        bin_data[l] = zip(x, y)
    
    # Actually plot now that all the data is available
    return benchmark, bin_data

def plot_diff_vs_benchmark(benchmark, bin_data, index_name, s):
    # bin_data is a list
    
    nb_bins = len(bin_data)
    if nb_bins == 5:
        prefix = 'Q'
    elif nb_bins == 10:
        prefix = 'D'
    else:
        raise ValueError('[ERROR] Found {} bins. This is not supported yet'.format(nb_bins))
    
    fig = plt.figure(figsize=(10, 5))
    benchmark_x, benchmark_y = zip(*benchmark)

    if benchmark_y[0] != -s['pf_init_value']:  # No benchmark displayed
        plt.plot_date(benchmark_x, benchmark_y, label=index_name, linestyle='-.', linewidth=2, ms=10, marker=',')

    for idx, l in enumerate(bin_data):
        x, y = zip(*bin_data[l])
        # bin_name = prefix + str(idx+1)
        plt.plot_date(x, y, label=l, linestyle='-')  # Label is given by the key
    
    plt.legend()
    plt.title('Portfolio benchmark against {} for different bins'.format(index_name), fontsize=20)
    plt.xlabel('Historical data', fontsize=16)
    plt.ylabel('Portfolio value', fontsize=16)

    plt.savefig(os.path.join(s['path_output_folder'], '{}_{}_Benchmark_{}.png'.format(x[0].strftime('%Y%m%d'), x[-1].strftime('%Y%m%d'), index_name)))
    if run_from_ipython():
        plt.show()
    else:
        plt.close(fig)

def update_ax_diff_vs_benchmark(ax, benchmark, bin_data, index_name, s, ylim, m):
    # bin_data is a list
    
    nb_bins = len(bin_data)
    if nb_bins == 5:
        prefix = 'Q'
    elif nb_bins == 10:
        prefix = 'D'
    else:
        raise ValueError('[ERROR] Found {} bins. This is not supported yet'.format(nb_bins))
    
    # fig = plt.figure(figsize=(10, 5))
    benchmark_x, benchmark_y = zip(*benchmark)

    if benchmark_y[0] != -s['pf_init_value']:  # No benchmark displayed
        ax.plot_date(benchmark_x, benchmark_y, label=index_name, linestyle='-.', linewidth=2, ms=10, marker=',')

    for idx, l in enumerate(bin_data):
        x, y = zip(*bin_data[l])
        # bin_name = prefix + str(idx+1)
        ax.plot_date(x, y, label=l, linestyle='-')  # Label is given by the key
    
    ax.legend()
    ax.set_title('{} against {}'.format(m, index_name))
    #ax.set_xlabel('Historical data', fontsize=16)
    #ax.set_ylabel('Portfolio value', fontsize=16)
    if ylim:
        ax.set_ylim(ylim)

    #plt.savefig(os.path.join(s['path_output_folder'], '{}_{}_Benchmark_{}.png'.format(x[0].strftime('%Y%m%d'), x[-1].strftime('%Y%m%d'), index_name)))
    #if run_from_ipython():
        #plt.show()
    #else:
        #plt.close(fig)
