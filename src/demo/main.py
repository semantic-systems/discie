import json
import threading

import dash
import requests
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from pyvis.network import Network

from src.discriminative_cie.discriminative_cie import DiscriminativeCIE

# Create a Dash app
app = dash.Dash(__name__)

# Create a sample graph using pyvis
g = Network()
g.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=200, spring_strength=0.001, damping=0.09,
                           overlap=0.5)
g.add_node("A", label="Node A")
g.add_node("B", label="Node B")
g.add_node("C", label="Node C")
g.add_edge("A", "B", label="Edge AB")
g.add_edge("B", "C", label="Edge BC")
g.show("graph.html")

alt_graph = Network()
alt_graph.add_node("A", label="Node d")
alt_graph.add_node("B", label="Node as")
alt_graph.add_node("C", label="Node Cf")

# Define the app layout
app.layout = html.Div([
html.Link(
        rel='stylesheet',
        href='src/demo/assets/css/bootstrap.min.css'
    ),
    html.H1("DISCIE Demo", style={'textAlign': 'center', 'background-color': '#f2f2f2', 'padding': '10px'}),

# Add Divs to display threshold values
# Textarea for input
html.Label("Input Text:", style={'fontSize': 16, 'margin-left': '3%', }),
html.Div([
dcc.Textarea(
    id='input-text',
    placeholder='Enter your text here!',
    rows=5,
    style={'width': '80%', 'fontSize': 14, 'margin-left': '3%'}
),
]),
html.Div([
html.Label("Mention threshold:", style={'width': '9%', 'margin-right': '1px', 'display': 'inline-block', 'margin-left': '3%'}),
        html.Div([
            dcc.Slider(id='threshold1-slider', min=0, max=1, step=0.01, value=0.1, marks={0: '0', 1: '1'}, tooltip={'placement': 'top', 'always_visible': True}),
        ], style={'display': 'inline-block', 'width': '71%'}),
    ], style={'display': 'flex', 'align-items': 'center'}),
html.Div([
        html.Label("Property threshold:", style={'width': '9%', 'margin-right': '1px', 'display': 'inline-block', 'margin-left': '3%'}),
        html.Div([
            dcc.Slider(id='threshold2-slider', min=0, max=1, step=0.01, value=0.5, marks={0: '0', 1: '1'}, tooltip={'placement': 'top', 'always_visible': True}),
        ], style={'display': 'inline-block', 'width': '71%'}),
    ], style={'display': 'flex', 'align-items': 'center'}),

html.Div([
        html.Label("Combined threshold:", style={'width': '9%', 'margin-right': '1px' ,'display': 'inline-block', 'margin-left': '3%'} ),
        html.Div([
            dcc.Slider(id='threshold3-slider', min=0, max=1, step=0.01, value=0.3, marks={0: '0', 1: '1'}, tooltip={'placement': 'top', 'always_visible': True}),
        ], style={'display': 'inline-block', 'width': '71%'}),
    ], style={'display': 'flex', 'align-items': 'center'}),


    html.Button('Update Graph', id='button-update-graph', n_clicks=0, style={'color': 'white', 'background-color': '#009688', 'width': '10%', 'margin-left': '38%'}),

    html.Iframe(id='graph-output', srcDoc=open("graph.html", "r").read(), width='80%', height='800', style={'margin-left': '3%', 'margin-top': '20px'}),
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
            graph.barnes_hut(gravity=-10000, central_gravity=0.3, spring_length=200, spring_strength=0.001, damping=0.09,
                           overlap=0.5)

            graph_data = response.json()
            print(graph_data)
            for node in graph_data['nodes']:
                graph.add_node(node['id'], label=f"{node['id']}/{node['label']}")
            for edge in graph_data['edges']:
                graph.add_edge(edge['source'], edge['target'], label=f"{edge['property']}/{edge['property_label']}")

            graph.show("graph.html")

        return open("graph.html", "r").read()

if __name__ == "__main__":
    app.run(debug=True)


