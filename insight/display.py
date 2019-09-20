import matplotlib
matplotlib.use('Qt5Agg')  # Use another backend
import matplotlib.pyplot as plt
import numpy as np


def diff_vs_stock(qtr_metric_result, ticker_data, s, method='diff'):
    # Select the appropriate stuff to plot
    if method == 'diff':
        metrics = s['metrics_diff']
    elif method == 'sentiment':
        metrics = s['metrics_sentiment']
    else:
        raise ValueError('[ERROR] Method unknown')
    
    
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # 1. Display the stock data
    lists = sorted(ticker_data.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    dates = matplotlib.dates.date2num(x)
    ax1.plot_date(dates, y, ms=1)
    ax1.set_ylabel('Stock price [$]', fontsize=16)
    ax1.set_xlabel('Historical data', fontsize=16)
    ax1.set_title('Similarity scores vs daily {} price for {}'.format(s['type_daily_price'], s['ticker']), fontsize=20)

    # 2. Display the histogram
    ax2 = ax1.twiny()

    xmin, xmax = ax1.get_xlim()  # Copy the start and finish of the scale
    ymin, ymax = ax1.get_ylim()
    width = 10
    
    # First pass to get all the center positions of the bar charts
    x = []
    xtick_label = []
    for idx, qtr in enumerate(sorted(qtr_metric_result)):
        if idx < s['lag']:
            continue
        if len(qtr_metric_result[qtr]) == 0:
            # It is rare, but it is possible that no report were filed that qtr
            continue
        else:
            date = matplotlib.dates.date2num(qtr_metric_result[qtr]['0']['published'])
            x.append(date)
            xtick_label.append(qtr_metric_result[qtr]['0']['published'])
    x = np.array(x)

    # Now we plot all the histograms, one metric at a time
    for ii, m in enumerate(metrics):
        # Get the data for each quarter on that metric
        data = []
        for idx, qtr in enumerate(sorted(qtr_metric_result)):
            if idx < s['lag']:
                continue
            if len(qtr_metric_result[qtr]) == 0:
                # It is rare, but it is possible that no report were filed that qtr
                continue
            else:
                data.append(qtr_metric_result[qtr][m])
        data = np.array(data)
        data *= (ymax-ymin)  # Scaling factor
        ax2.bar(x - width/(len(metrics)) + width*ii, data, width, label=m)
    
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylabel('Metric similarity []', fontsize=16)
    plt.legend()
    plt.show()
