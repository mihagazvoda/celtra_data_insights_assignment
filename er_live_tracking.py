import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from celtra_task_functions import er_parameters, er_stats, read_file, \
    resample_calculate_er, get_axis_data, get_attribute_labels, download_file
import numpy as np
from pathlib import Path

# parameters
url = 'https://s3.amazonaws.com/files.celtra-test.com/bdsEventsSample.gz'
filepath = 'data.csv'
er_interval = 20  # interval for calculating ER in minutes

# download file from web if not in folder
if Path(filepath).is_file():
    df = read_file(filepath)
else:
    df = download_file(url, filepath)

# get uniques for sdks and objectClazz
sdk_labels, object_labels = get_attribute_labels(df)

app = dash.Dash()

app.layout = html.Div([
    html.H1('ER Live Tracking'),
    html.Div([
        html.H4('SDK'),
        dcc.Dropdown(
            id='sdk-dropdown',
            options=sdk_labels)
    ],
        style={'width': '49%', 'display': 'inline-block'}
    ),
    html.Div([
        html.H4('Object class'),
        dcc.Dropdown(
            id='object-dropdown',
            options=object_labels)
    ],
        style={'width': '49%', 'display': 'inline-block'}
    ),
    html.Div([
        dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0),
        dcc.Graph(id='live-update-graph')
    ]),
    html.Div([
        html.H4('Number of minutes to compare ER with the ER of the rest'),
        dcc.Slider(
            id='slider',
            min=5,
            max=60,
            step=5,
            value=10,
            marks={i: '{} min'.format(i) for i in np.arange(10, 61, 10).tolist()})
    ]),
    html.Div(
        id='my-div',
        style={'textAlign': 'center'}
    )
])


# when on of the inputs change the program updates the graph
@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals'),
               Input('sdk-dropdown', 'value'),
               Input('object-dropdown', 'value')])
def update_graph(n, sdk, object):
    df = read_file(filepath, sdk, object)
    df_output = resample_calculate_er(df, er_interval)
    x, y, y_error, y_smoothed = get_axis_data(df_output, er_interval, 0.75)

    # plotting
    trace1 = go.Scatter(x=x, y=y, error_y=dict(type='data', array=y_error, visible=True),
                        name='ER with standard devation', mode='markers')
    trace2 = go.Scatter(x=x, y=y_smoothed, line=dict(color='black'), opacity=0.2, name='Smoothed ER')
    layout = go.Layout(title=f'ER and standard deviation for {er_interval} minute intervals',
                       xaxis=dict(title='time'), yaxis=dict(title='ER [%]'))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig


# when on of the inputs change the program updates ER stats
@app.callback(
    Output('my-div', 'children'),
    [Input('slider', 'value'),
     Input('sdk-dropdown', 'value'),
     Input('object-dropdown', 'value')])
def update_er_stats(slider_minutes, sdk, object):
    df = read_file(filepath, sdk, object)
    border_time = pd.Timestamp(2015, 4, 16, 20, 0, 0) - pd.Timedelta(minutes=slider_minutes)
    er_diff, p_value, k1, k2 = er_stats(df, border_time)

    if (k1 > 5) & (k2 > 5):
        return [html.H2(f"ER(last {slider_minutes} minutes) - ER(rest): {format(er_diff*100, '.1f')}%"),
                html.H2(f"P-value: {format(p_value*100, '.2f')}%")]
    else:
        return [html.H2(f"ER(last {slider_minutes} minutes) - ER(rest): {format(er_diff*100, '.1f')}%"),
                html.H2(f"Confidence level can't be determined the distribution is not Gaussian. "
                        f"Please, select longer time interval!")]


if __name__ == '__main__':
    app.run_server(debug=True)
