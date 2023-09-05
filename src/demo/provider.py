import json

import networkx as nx
from flask import jsonify, request, Flask

from src.discriminative_cie.discriminative_cie import DiscriminativeCIE
from SPARQLWrapper import SPARQLWrapper, JSON


wikidata_endpoint = "https://query.wikidata.org/sparql"
wikidata_sparql = SPARQLWrapper(wikidata_endpoint)

def normalize_identifier(identifier: str):
    if identifier.startswith("Q"):
        return identifier
    elif identifier.startswith("http://www.wikidata.org/entity/"):
        return identifier.split("/")[-1]
    else:
        raise ValueError(f"Invalid identifier: {identifier}")
def get_labels_via_sparql(identifiers: list):
    identifiers = " ".join([f"wd:{x}" for x in identifiers])
    query = f"""
    SELECT ?item ?label
    WHERE {{
    VALUES ?item {{ {identifiers} }}
      ?item rdfs:label ?label .
      FILTER(LANGMATCHES(LANG(?label), "en"))
    }}"""
    wikidata_sparql.setQuery(query)
    wikidata_sparql.setReturnFormat(JSON)
    results = wikidata_sparql.query().convert()
    return {normalize_identifier(x["item"]["value"]): x["label"]["value"] for x in results["results"]["bindings"]}


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
                               num_candidates=30,
                             only_one_relation_per_pair=True,
                               )


app = Flask(__name__)
@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    input_text = data['text']
    threshold1 = data['threshold1']
    threshold2 = data['threshold2']
    threshold3 = data['threshold3']
    discie.mention_threshold = threshold1
    discie.property_threshold = threshold2
    discie.combined_threshold = threshold3
    disambiguated, _ = discie.run([{"text":input_text}])

    nodes = set()
    edges = []

    triples, all_occuring_entities = disambiguated[0]
    nodes = list(set(all_occuring_entities))
    pids = list({triple[1] for triple in triples})
    all_labels = get_labels_via_sparql(nodes + pids)
    for triple in triples:
        edges.append((triple[0], triple[2], triple[1]))

    # Convert the graph to a format that can be sent as JSON
    nodes = [{'id': node, 'label': all_labels[node]} for node in nodes]
    edges = [{'source': source, 'target': target, "property": label, "property_label": all_labels[label]} for source, target, label in edges]

    response = {
        'nodes': nodes,
        'edges': edges
    }

    return jsonify(response)


if __name__ == '__main__':
    discie = instantiate_discie()
    app.run(host='0.0.0.0', port=5001)