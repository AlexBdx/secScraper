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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[
    dcc.Graph(
        id='main_graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [100, 150, 200], 'type': 'scatter', 'name': 'Metric 1'},
                {'x': [1, 2, 3], 'y': [0.5, 0.6, 0.3], 'type': 'scatter', 'name': 'Metric 2', 'yaxis': 'y2'},
            ],
            'layout': go.Layout(
            xaxis={'title': 'Historical records', },
            yaxis={'title': 'Stock price [$]'},
            yaxis2={'title': 'Score [0-1]', 'overlaying': 'y', 'side': 'right'},
            title={'text': 'Selection pending'}
            )
        }
    )]
)

if __name__ == '__main__':
    app.run_server(debug=True)
