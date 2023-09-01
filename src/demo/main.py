import json

import dash
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
    dcc.Input(id='input-text', type='text', placeholder='Enter graph data here'),

    html.Button('Update Graph', id='button-update-graph', n_clicks=0),

    html.Iframe(id='graph-output', srcDoc=open("graph.html", "r").read(), width='100%', height='600'),
])


def instantiate_discie():

    types_index = json.load(open(f"models/types_index_relations.json"))

    alt_num_types = len(json.load(open(f"models/types_index_cross.json")))

    return DiscriminativeCIE("models/run_training_bi_encoder_new",
                               "models/model-epoch=06-val_f1=0.85_val_f1.ckpt",
                               "models/model-epoch=13-val_triple_f1=0.85_triple_f1.ckpt",
                               "models/model-epoch=25-val_triple_f1=0.90_triple_f1.ckpt",
                               types_index=types_index,
                               include_mention_scores=True,
                               alt_num_types=alt_num_types,
                               mention_threshold=0.1,
                               property_threshold=0.5,
                               combined_threshold=0.7,
                               num_candidates=10
                               )


discie = instantiate_discie()
@app.callback(Output('graph-output', 'srcDoc'), [Input('button-update-graph', 'n_clicks')],
              [State('input-text', 'value')])
def update_graph(n_clicks, input_text):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'button-update-graph.n_clicks':
        alt_graph.show("graph.html")

        return open("graph.html", "r").read()

# @app.callback(Output('graph-output', 'srcDoc'), [Input('input', 'value')])
# def update_graph(input_value):
#     if input_value:
#         alt_graph.show("graph.html")
#     return open("graph.html", "r").read()

if __name__ == "__main__":
    app.run_server(debug=True)


