import json

import networkx as nx
from flask import jsonify, request, Flask

from src.discriminative_cie.discriminative_cie import DiscriminativeCIE


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
                               property_threshold=0.3,
                               combined_threshold=0.3,
                               num_candidates=30
                               )


app = Flask(__name__)
@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    input_text = data['text']
    disambiguated, _ = discie.run([{"text":input_text}])

    nodes = set()
    edges = []

    triples, all_occuring_entities = disambiguated[0]
    nodes = list(set(all_occuring_entities))
    for triple in triples:

        edges.append((triple[0], triple[2], triple[1]))

    # Convert the graph to a format that can be sent as JSON
    nodes = [{'id': node} for node in nodes]
    edges = [{'source': source, 'target': target, "label": label} for source, target, label in edges]

    response = {
        'nodes': nodes,
        'edges': edges
    }

    return jsonify(response)

def test():
    print("test")
    return "test"

discie = instantiate_discie()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)