import matplotlib
#matplotlib.use('Qt5Agg')  # Use another backend
import matplotlib.pyplot as plt
import numpy as np


def diff_vs_stock(qtr_metric_result, ticker_data, ticker, s, method='diff'):
    """Display the data for a given ticker accross the time_range that was specified"""
    #print(ticker_data)
    # 0. Select the type of plot
    if method == 'diff':
        metrics = [m for m in s['metrics'] if m[:4] == 'diff']
    elif method == 'sentiment':
        metrics = [m for m in s['metrics'] if m[:4] == 'sing']
    else:
        raise ValueError('[ERROR] Method unknown')
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 1. Display the stock data
    lists = sorted(ticker_data.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unzips the dates & financial data
    # At this point, y is a list of lists. We need to extract the price from it.
    price, market_cap = zip(*y)  # Will crash if len(y) > 2? Or ignore the rest?
    
    dates = matplotlib.dates.date2num(x)
    ax1.plot_date(dates, price, ms=1)
    ax1.set_ylabel('Stock price [$]', fontsize=16)
    ax1.set_xlabel('Historical data', fontsize=16)
    ax1.set_title('Similarity scores vs daily {} price for ticker {}'.format(s['type_daily_price'], ticker), fontsize=20)

    # 2. Display the histogram
    ax2 = ax1.twiny()

    xmin, xmax = ax1.get_xlim()  # Copy the start and finish of the scale
    ymin, ymax = ax1.get_ylim()
    width = 5
    
    # First pass to get all the center positions of the bar charts
    x = []
    xtick_label = []
    for idx, qtr in enumerate(sorted(qtr_metric_result)):
        """
        if idx < s['lag']:
            continue
        """
        if len(qtr_metric_result[qtr]) == 0:
            # It is rare, but it is possible that no report were filed that qtr
            continue
        else:
            date = matplotlib.dates.date2num(qtr_metric_result[qtr]['0']['published'])
            print("[INFO] Publication dates:", qtr_metric_result[qtr]['0']['published'])
            x.append(date)
            xtick_label.append(qtr_metric_result[qtr]['0']['published'])
    #print("[INFO] Publication dates:", x)
    x = np.array(x)

    # Now we plot all the histograms, one metric at a time
    for ii, m in enumerate(metrics):
        # Get the data for each quarter on that metric
        data = []
        for idx, qtr in enumerate(sorted(qtr_metric_result)):
            """
            if idx < s['lag']:
                continue
            """
            if len(qtr_metric_result[qtr]) == 0:
                # Is that still possible?
                continue
            else:
                data.append(qtr_metric_result[qtr][m])
        data = np.array(data)
        data *= ymax/2  # Scaling factor
        if method == 'sentiment':
            data *= 1000
        ax2.bar(x - width/(len(metrics)) + width*ii, data, width, label=m)
    
    ax2.set_xlim([xmin, xmax])
    #ax2.set_ylim([0, 1])
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylabel('Metric similarity []', fontsize=16)
    plt.legend()
    plt.show()
