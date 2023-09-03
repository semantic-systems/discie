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
    html.H1("Graph Visualization with Dash and pyvis"),
    dcc.Input(id='input-text', type='text', placeholder='Enter graph data here', size='100'),

    html.Button('Update Graph', id='button-update-graph', n_clicks=0),

    html.Iframe(id='graph-output', srcDoc=open("graph.html", "r").read(), width='100%', height='600'),
])


@app.callback(Output('graph-output', 'srcDoc'), [Input('button-update-graph', 'n_clicks')],
              [State('input-text', 'value')])
def update_graph(n_clicks, input_text):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'button-update-graph.n_clicks':
        response = requests.post('http://localhost:5001/process_text', json={'text': input_text})
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

# @app.callback(Output('graph-output', 'srcDoc'), [Input('input', 'value')])
# def update_graph(input_value):
#     if input_value:
#         alt_graph.show("graph.html")
#     return open("graph.html", "r").read()

if __name__ == "__main__":
    app.run(debug=True)


