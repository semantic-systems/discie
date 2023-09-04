import json
from collections import defaultdict

import jsonlines
from tqdm import tqdm


type_dict = {
    "Q5": "PER",
    "Q43229": "ORG",
    "Q2221906": "LOC",
    "Q17334923": "LOC",
    "Q27096213": "LOC",
}


def main(superclass_mapping: dict):
    filter_set = {"P31"} #, "P106", "P641", "P17"}
    item_types = defaultdict(lambda: defaultdict(set))

    type_frequencies = defaultdict(int)
    encountered_entity_types = set()
    for item in tqdm(
            jsonlines.open("data/parsed_wikidata_info.jsonl"),
            total=99004802):
        mapped_types = set()
        for p, o in item["triples"]:
            if p in filter_set:
                if o in superclass_mapping:
                    all_classes = set(superclass_mapping[o])
                    all_classes.add(o)
                else:
                    all_classes = {o}
                encountered_entity_types.update(all_classes)
                if p in filter_set:
                    for class_ in all_classes:
                        if class_ in type_dict:
                            mapped_types.add(type_dict[class_])
        if not mapped_types:
            mapped_types.add("Other")
        item_types[item["qid"]] = mapped_types

    output_file = jsonlines.open("data/mapped_types.jsonl", "w")
    for key, value in item_types.items():
        output_file.write({"item": key, "types": list(value)})

if __name__ == "__main__":
    superclass_mapping = json.load(open("data/class_superclasses.json"))
    main(superclass_mapping)



