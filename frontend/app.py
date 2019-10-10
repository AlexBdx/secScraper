# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from secScraper import *
import psycopg2
import matplotlib
import numpy as np
from datetime import datetime
import os

# Retrieve the settings that were used to perform the last simulation
connector = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="1")
s = postgres.retrieve_settings(connector)
metric_options = [{'label': name, 'value': name} for name in s['diff_metrics']]
lookup, reverse_lookup = postgres.retrieve_lookup(connector)
stock_data = postgres.retrieve_all_stock_data(connector, 'stock_data')
index_data = postgres.retrieve_all_stock_data(connector, 'index_data')
path = s['path_output_folder']
path1 = os.path.join(path, 'pf_values1.csv')
path2 = os.path.join(path, 'pf_values2.csv')
pf_values = postgres.retrieve_pf_values_data(connector, path1, path2, s)
path_metric_scores = os.path.join(path, 'ms.csv')
metric_scores = postgres.retrieve_ms_values_data(connector, path_metric_scores, s)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Layout
app.layout = html.Div(children=[
    html.Div(children='The secScraper project', style={'textAlign': 'center', 'fontSize': 40}),
    
    dcc.Graph(
        id='main_graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'Metric 1'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Metric 2'},
            ],
            'layout': go.Layout(
            xaxis={'title': 'Historical records', },
            yaxis={'title': 'Stock price [$]'},
            yaxis2={'title': 'Score [0-1]', 'overlaying': 'y', 'side': 'right', 'range': [0, 1]},
            title={'text': 'Selection pending'}
            )
        }
    ),
    dcc.RangeSlider(id='range_slider',
    marks={i: '{}'.format(i) for i in range(s['time_range'][0][0], s['time_range'][1][0]+1)},
    min=s['time_range'][0][0],
    max=s['time_range'][1][0],
    value=[s['time_range'][0][0], s['time_range'][1][0]]
    ), 
    
    dcc.Markdown('''
    ### Visualization mode
    Select the view of the database you would like to have.
    
    **Company view** needs a ticker while **Portfolio view** displays the best portfolio found in the database over the time range calculated.
    '''),

    dcc.RadioItems(id='view_mode',
        options=[
            {'label': 'Company View', 'value': 'company_view'},
            {'label': 'Portfolio View', 'value': 'pf_view'}
        ],
        value='company_view'
    ),
    dcc.Checklist(id='normalize_pf',
        options=[
            {'label': 'Normalize portfolios by index?', 'value': 'norm'},
        ],
        value=['norm']
    ),
    
    dcc.Markdown('#### Differentiation methods'),
    dcc.Checklist(id='metrics',
        options=metric_options,
        value=['diff_jaccard', 'diff_cosine_tf']
    ),
    
    dcc.Markdown('#### Apply your selection'),
    html.Label('Please enter a ticker'),
    dcc.Input(id='input_ticker', value='MHK', type='text'),
    html.Label(id='does_ticker_exist'),
    dcc.Dropdown(id='index_name',
    options=[
        {'label': 'DJI', 'value': 'DJI'},
        {'label': 'IXIC', 'value': 'IXIC'},
        {'label': 'RUT', 'value': 'RUT'},
        {'label': 'SPX', 'value': 'SPX'}
    ],
    value='SPX'
    ),
    html.Button('Submit', id='button'),
    
    dcc.Markdown('## Display'),
    dcc.Markdown('''
    The following parameters were found to have been used to compute the database:
    
    **differentiation method:** {}
    
    **time range:** {} to {}
    
    **trade tax rate:** {}
    
    **stock price considered:** {}
    
    '''.format(s['differentiation_mode'], *s['time_range'], s['tax_rate'], s['type_daily_price'])
    ),
    
    
    dcc.Markdown('''
    #### Description
    Investing strategies can be extremely complex. Using advanced mathematical models, investors try relentlessly to "Beat the Street" and gain on edge on each other. One piece of this puzzle are the 10-X filings that listed companies submit at the end of each quarter to report on their operations. They contain a wealth of quantitative information on their financial health.

    Unfornately, they have gotten longer and longer over time, which makes reading one a daunting task. From the boilerplate text to incomplete sections, their usefulness has substantially decreased. But has it?

    In this project, I attempt to disregard all financial data in the 10-X and build a virtual portfolio of companies based only on text. Let's see how that performs!
    
    #### Useful information
    
    **GitHub:** [secScraper](https://github.com/AlexBdx/secScraper)
    
    **Documentation:** [Read The Docs!](https://sec-scrapper.readthedocs.io/en/latest/)
    
    **PyPi package:** [PyPi](https://pypi.org/project/secScraper/) (install via `pip3 install secScraper`)
    ''')
    ])


# Callbacks 
"""
@app.callback(
    [Output('main_graph', 'figure')],
    [Input('range_slider', 'value')],
    [State('main_graph', 'figure')]
)
def update_plot_range(time_range, graph_figure):
    new_start = time_range[0]
    new_end = time_range[1]
    graph_figure['xaxis']['range'] = [new_start, new_end]
    return graph_figure
"""

# When the user clicks on Submit, the data for this ticker is retrieved.
@app.callback(
    [Output('does_ticker_exist', 'children'), Output('main_graph', 'figure')],
    [Input('button', 'n_clicks'), Input('range_slider', 'value')],
    [State('input_ticker', 'value'), State('metrics', 'value'), 
    State('main_graph', 'figure'), State('view_mode', 'value'),
    State('index_name', 'value'), State('normalize_pf', 'value')]
)
def update_display(input_button, tr, input_ticker, metrics, graph_figure, visualization_mode, index_name, norm):
    # Check if that input_ticker exists in our stock database
    connector = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="1")
    
    if visualization_mode == 'company_view':
        user_message, graph_figure = update_company_view(input_ticker, metrics, graph_figure, tr)
    elif visualization_mode == 'pf_view':
        user_message, graph_figure = update_pf_view(index_name, metrics, graph_figure, norm, tr)
    return user_message, graph_figure

def update_company_view(input_ticker, metrics, graph_figure, tr):
    input_ticker = input_ticker.upper()
    flag_ticker_exist = postgres.does_ticker_exist(connector, input_ticker)
    try:  # Get the CIK corresponding to this ticker via SQL query
        cik = reverse_lookup[input_ticker]
    except KeyError:
        flag_ticker_exist = False
    if flag_ticker_exist:
        user_message = "Ticker {} found in database. CIK: {}".format(input_ticker, cik)
        # print(user_message)
        
        # Retrieve the cik_scores for that CIK
        cik_scores = postgres.retrieve_cik_scores(connector, cik, s)
        # print("cik_scores", cik_scores)
        extracted_cik_scores = cik_scores[cik]
        # print("extracted_cik_scores", extracted_cik_scores)
        
        # Extract the stock data to be displayed
        extracted_stock_data = stock_data[input_ticker]
        benchmark, metric_data = display.diff_vs_stock(extracted_cik_scores, extracted_stock_data, input_ticker, s, method='diff')
        benchmark_x, benchmark_y = zip(*benchmark)
        #print(benchmark_x)
        #print(benchmark_y)
        # 2. Partially rebuild the graph_figure
        start = str(tr[0])+'0101'
        end = str(tr[1])+'1231'
        start = datetime.strptime(start, '%Y%m%d').date()
        end = datetime.strptime(end, '%Y%m%d').date()
        graph_figure = {
            'data': [
                {'x': benchmark_x, 'y': benchmark_y, 'type': 'scatter', 'name': input_ticker}
            ],
            'layout': go.Layout(
            xaxis={'title': 'Historical records', 'range': [start, end]},
            yaxis={'title': 'Stock price [$]', 'showgrid': False},
            yaxis2={'title': 'Score [0-1]', 'overlaying': 'y', 'side': 'right', 'range': [0, 1], 'showgrid': False},
            title={'text': 'Stock price for {} (CIK: {})'.format(input_ticker, cik)}
            )
        }
        
        metric_names = [m for m in s['metrics'] if m[:4] == 'diff']
        for m in metric_names:  # Go through the requested metrics
            position = metric_names.index(m)  
            data = metric_data[position]
            x, y = zip(*data)
            #x = [matplotlib.dates.num2date(entry).date() for entry in x]
            #print(x)
            #print(y)
            
            graph_figure['data'].append({'x': x, 'y': y, 'type': 'bar', 'name': m, 'yaxis': 'y2'}) 
        
    else:
        user_message = "Ticker {} not found in database.".format(input_ticker)
    return user_message, graph_figure

def update_pf_view(index_name, metrics, graph_figure, norm, tr):
    # 0. Config
    metric = metrics[0]  # You should only have one selected
    norm_by_index = True if len(norm) else False

    # 1. Retrieve
    benchmark, bin_data = display.diff_vs_benchmark_ns(pf_values, index_name, index_data, metric, s, norm_by_index=norm_by_index)
    nb_bins = len(bin_data)
    if nb_bins == 5:
        prefix = 'Q'
    elif nb_bins == 10:
        prefix = 'D'
    else:
        raise ValueError('[ERROR] Found {} bins. This is not supported yet'.format(nb_bins))
    
    benchmark_x, benchmark_y = zip(*benchmark)
    
    start = str(tr[0])+'0101'
    end = str(tr[1])+'1231'
    start = datetime.strptime(start, '%Y%m%d').date()
    end = datetime.strptime(end, '%Y%m%d').date()
    graph_figure = {
    'data': [], 
    'layout': go.Layout(
    xaxis={'title': 'Historical records', 'range': [start, end]},
    yaxis={'title': 'Portfolio value'},
    title={'text': 'Portfolio benchmark against {} for different bins (differentiation: {})'.format(index_name, metric)}
    )}
    # 2. Setup the graph and display the index
    if benchmark_y[0] != -s['pf_init_value']:  # No benchmark displayed
        graph_figure['data'].append({'x': benchmark_x, 'y': benchmark_y, 'type': 'line', 'name': index_name, 'line': {'width': 6, 'color':'rgb(255, 0, 0)'}})
    

    
    # 3. Add all the quintiles/deciles for a given metric. We plot all of of them
    for l in s['bin_labels']:
        x, y = zip(*bin_data[l])
        graph_figure['data'].append({'x': x, 'y': y, 'type': 'scatter', 'name': l})

    # print(graph_figure['data'])
    
    user_message = "Success"
    return user_message, graph_figure

if __name__ == '__main__':
    app.run_server(debug=True)
