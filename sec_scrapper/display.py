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


def diff_vs_stock(qtr_metric_result, ticker_data, ticker, s, method='diff'):
    """
    Display the calculated data for a given ticker across the time_range that was specified.

    :param qtr_metric_result: Dictionary containing the data to plot
    :param ticker_data: Daily stock value for the ticker considered
    :param ticker: Company ticker on the US stock exchange
    :param s: Settings dictionary
    :param method: Specify if a difference between two reports or an analysis of each report.
    :return: Void
    """
    # 0. Select the type of plot
    if method == 'diff':
        metrics = [m for m in s['metrics'] if m[:4] == 'diff']
    elif method == 'sentiment':
        metrics = [m for m in s['metrics'] if m[:4] == 'sing']
    else:
        raise ValueError('[ERROR] Method unknown')
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 1. Display the stock data
    lists = sorted(ticker_data.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unzips the dates & financial data
    # At this point, y is a list of lists. We need to extract the price from it.
    price, market_cap = zip(*y)  # Will crash if len(y) > 2? Or ignore the rest?
    
    dates = matplotlib.dates.date2num(x)
    ax1.plot_date(dates, price, ms=1)
    ax1.set_ylabel('Stock price [$]', fontsize=16)
    ax1.set_xlabel('Historical data', fontsize=16)
    ax1.set_title('Similarity scores vs daily {} price for ticker {}'
                  .format(s['type_daily_price'], ticker), fontsize=20)

    # 2. Display the histogram
    ax2 = ax1.twiny()

    xmin, xmax = ax1.get_xlim()  # Copy the start and finish of the scale
    ymin, ymax = ax1.get_ylim()
    width = 5
    
    # First pass to get all the center positions of the bar charts
    publication_dates = []
    xtick_label = []
    for idx, qtr in enumerate(sorted(qtr_metric_result)):
        """
        if idx < s['lag']:
            continue
        """
        if len(qtr_metric_result[qtr]) == 0:
            # This should not happen given all the pre-processing done.
            continue
        else:
            date = matplotlib.dates.date2num(qtr_metric_result[qtr]['0']['published'])
            print("[INFO] Publication dates:", qtr_metric_result[qtr]['0']['published'])
            publication_dates.append(date)
            xtick_label.append(qtr_metric_result[qtr]['0']['published'])
    # print("[INFO] Publication dates:", x)
    publication_dates = np.array(publication_dates)

    # Now we plot all the histograms, one metric at a time
    for ii, m in enumerate(metrics):
        # Get the data for each quarter on that metric
        data = []
        for idx, qtr in enumerate(sorted(qtr_metric_result)):
            if len(qtr_metric_result[qtr]) == 0:
                # Is that still possible?
                continue
            else:
                data.append(qtr_metric_result[qtr][m])
        data = np.array(data)
        data *= ymax/2  # Scaling factor
        if method == 'sentiment':
            data *= 1000
        ax2.bar(publication_dates - width/(len(metrics)) + width*ii, data, width, label=m)
    
    ax2.set_xlim([xmin, xmax])
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylabel('Metric similarity []', fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(s['path_output_folder'], '{}_{}_View_{}.png'
                             .format(x[0].strftime('%Y%m%d'), x[-1].strftime('%Y%m%d'), ticker)))
    if run_from_ipython():
        plt.show()
    else:
        plt.close(fig)


def diff_vs_benchmark(pf_values, index_name, index_data, s):
    """
    Plot a portfolio vs an index.

    :param pf_values: Value of the portfolio over time.
    :param index_name: Name of the index.
    :param index_data: Daily value of the index.
    :param s: Settings dictionary.
    :return: Void
    """
    fig = plt.figure(figsize=(10, 5))
    for l in s['bin_labels']:
        x = []
        y = []
        for qtr in s['list_qtr'][s['lag']:]:
            start = "{}{}{}".format(str(qtr[0]), str(((qtr[1])-1)*3+1).zfill(2), '01')
            x.append(datetime.strptime(start, '%Y%m%d').date())
            y.append(pf_values['diff_cosine_tf_idf'][l][qtr][0])
        plt.plot_date(x, y, label=l, linestyle='-')

    """Overlay an index"""
    benchmark_x = []
    benchmark_y = []
    for qtr in s['list_qtr'][s['lag']:]:
        qtr_start_date = "{}{}{}".format(str(qtr[0]), str((qtr[1]-1)*3+1).zfill(2), '01')
        qtr_start_date = datetime.strptime(qtr_start_date, '%Y%m%d').date()
        days, _ = zip(*index_data[index_name])
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
            benchmark_y.append(index_data[index_name][idx][1])
        except KeyError:
            raise KeyError('[ERROR] The stock exchange should not have been shut down for more than 7 days.')

    benchmark_y = np.array(benchmark_y)  # Allow the division on next line
    benchmark_y *= s['pf_init_value']/benchmark_y[0]
    plt.plot_date(benchmark_x, benchmark_y, label=index_name, linestyle='-.', linewidth=2, ms=10, marker=',')
    plt.legend()
    plt.title('Portfolio benchmark against {} for different bins'.format(index_name), fontsize=20)
    plt.xlabel('Historical data', fontsize=16)
    plt.ylabel('Portfolio value [$]', fontsize=16)

    plt.savefig(os.path.join(s['path_output_folder'], '{}_{}_Benchmark_{}.png'
                             .format(x[0].strftime('%Y%m%d'), x[-1].strftime('%Y%m%d'), index_name)))
    if run_from_ipython():
        plt.show()
    else:
        plt.close(fig)
