import json
import dash
import requests
from dash import dcc, html
from dash.dependencies import Input, Output, State
from pyvis.network import Network

# Create a Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("DISCIE Demo", style={'textAlign': 'center'}),

    # Textarea for input
    html.Div([
        html.Label("Input Text:", style={'fontSize': 16}),
        dcc.Textarea(
            id='input-text',
            placeholder='Enter your text here!',
            rows=5,
            style={'width': '100%', 'fontSize': 14}
        ),
    ]),

    # Threshold Sliders
    html.Div([
        html.Label("Mention Threshold:", style={'fontSize': 16, 'marginRight': '10px'}),
        html.Div([
            dcc.Slider(
                id='threshold1-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.1,
                marks={0: '0', 1: '1'},
                tooltip={'placement': 'top', 'always_visible': True},
            ),
        ], style={'width': '80%', 'margin': 'auto'}),
    ], style={'display': 'flex', 'alignItems': 'center'}),

    html.Div([
        html.Label("Property Threshold:", style={'fontSize': 16, 'marginRight': '10px'}),
        html.Div([
            dcc.Slider(
                id='threshold2-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.3,
                marks={0: '0', 1: '1'},
                tooltip={'placement': 'top', 'always_visible': True},
            ),
        ], style={'width': '80%', 'margin': 'auto'}),
    ], style={'display': 'flex', 'alignItems': 'center'}),

    html.Div([
        html.Label("Combined Threshold:", style={'fontSize': 16, 'marginRight': '10px'}),
        html.Div([
            dcc.Slider(
                id='threshold3-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.3,
                marks={0: '0', 1: '1'},
                tooltip={'placement': 'top', 'always_visible': True},
            ),
        ], style={'width': '80%', 'margin': 'auto'}),
    ], style={'display': 'flex', 'alignItems': 'center'}),

    # Update Graph Button
    html.Div([
        html.Button(
            'Update Graph',
            id='button-update-graph',
            n_clicks=0,
            style={'color': 'white', 'backgroundColor': '#009688', 'width': '80%', 'fontSize': 16,
                   'margin': '20px auto'}
        ),
    ]),

    # Graph Output
    dcc.Loading(
        id="loading-graph",
        type="default",
        children=[
            html.Iframe(
                id='graph-output',
                srcDoc=open("graph.html", "r").read(),
                width='80%',
                height='600',
                style={'border': 'none'}
            ),
        ],
    ),
])


@app.callback(Output('graph-output', 'srcDoc'), [Input('button-update-graph', 'n_clicks')],
              [State('input-text', 'value'),
               State('threshold1-slider', 'value'),
               State('threshold2-slider', 'value'),
               State('threshold3-slider', 'value')]
              )
def update_graph(n_clicks, input_text, threshold1, threshold2, threshold3):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'button-update-graph.n_clicks':
        response = requests.post('http://localhost:5001/process_text', json={'text': input_text,
                                                                             'threshold1': threshold1,
                                                                             'threshold2': threshold2,
                                                                             'threshold3': threshold3})
        if response.status_code == 200:
            graph = Network(directed=True)
            graph_data = response.json()
            print(graph_data)
            for node in graph_data['nodes']:
                graph.add_node(node['id'], label=node['id'])
            for edge in graph_data['edges']:
                graph.add_edge(edge['source'], edge['target'], label=edge['label'])

            graph.show("graph.html")

        return open("graph.html", "r").read()


if __name__ == "__main__":
    app.run(debug=True)
