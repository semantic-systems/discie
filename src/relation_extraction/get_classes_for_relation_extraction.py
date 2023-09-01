import json
import re
from collections import defaultdict

import jsonlines
from tqdm import tqdm


def get_item_types_and_frequencies(superclass_mapping: dict = None, type_filter: set = None):
    filter_set = {"P31"} #, "P106", "P641", "P17"}
    item_types = defaultdict(lambda: defaultdict(set))
    type_frequencies = defaultdict(int)
    encountered_entity_types = set()
    for item in tqdm(
            jsonlines.open("/data1/moeller/graphbasejointelre/src/wikidata_preprocessing/parsed_wikidata_info.jsonl"),
            total=99004802):
        for p, o in item["triples"]:
            if p in filter_set:
                if superclass_mapping is not None and o in superclass_mapping:
                    all_classes = set(superclass_mapping[o])
                    all_classes.add(o)
                else:
                    all_classes = {o}
                if type_filter is not None:
                    all_classes = all_classes.intersection(type_filter)
                encountered_entity_types.update(all_classes)
                item_types[item["qid"]][p].update(all_classes)
                for class_ in all_classes:
                    type_frequencies[class_] += 1

    never_encountered = type_filter.difference(encountered_entity_types)
    print(never_encountered)
    print(len(encountered_entity_types))
    return item_types, type_frequencies


def get_filter_set():
    content = json.load(open("/export/home/moeller/.cache/refined/wikipedia_data/class_to_idx.json"))
    filter_set = set()
    for k, v in content.items():
        pattern = r"Q\d+"

        # Search for the pattern in the string
        match = re.search(pattern, k)

        if match:
            q_identifier = match.group()
            filter_set.add(q_identifier)
        else:
            raise ValueError("No match")

    return filter_set


filter_set = get_filter_set()
superclass_mapping = json.load(open("data/class_superclasses.json"))

item_types, type_frequencies = get_item_types_and_frequencies(superclass_mapping, filter_set)

type_dict = {key: list({x for v in value.values() for x in v}) for key, value in item_types.items() }

new_types_dictionary = jsonlines.open("data/item_types_relation_extraction_alt.jsonl", "w")
for k, v in tqdm(type_dict.items()):
    new_types_dictionary.write({"item": k, "types": v})


