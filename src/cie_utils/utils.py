from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON


sparql = SPARQLWrapper(
    "https://dbpedia.org/sparql"
)
sparql.setReturnFormat(JSON)

def separate_texts(texts, spans):
    separated_texts = []
    is_mention_list = []
    for text, text_spans in zip(texts, spans):
        separated_text = []
        prev_end = 0
        is_mention = []
        for span_start, span_end in text_spans:
            if text[prev_end:span_start]:
                separated_text.append(text[prev_end:span_start])
                is_mention.append(False)
            separated_text.append(text[span_start:span_end])
            is_mention.append(True)
            prev_end = span_end
        if text[prev_end:]:
            separated_text.append(text[prev_end:])
            is_mention.append(False)
        is_mention_list.append(is_mention)
        separated_texts.append(separated_text)
    return separated_texts, is_mention_list

def map_page_ids_to_wikidata_ids(page_ids: set, batch_size: int = 100):
    mapping = {}
    batch = []
    for page_id in tqdm(page_ids):
        batch.append(page_id)
        if len(batch) >= batch_size:
            concat = " ".join([str(x) for x in batch])
            sparql.setQuery(f"""
                PREFIX wd: <http://www.wikidata.org/entity/> 
                SELECT ?wikidata_id ?wikipedia_id WHERE {{
                    VALUES ?wikipedia_id {{{concat}}} 
                    ?dbpedia_id owl:sameAs ?wikidata_id  .
                    ?dbpedia_id dbo:wikiPageID ?wikipedia_id .
                    FILTER strstarts(str(?wikidata_id),str(wd:))
                }}
                """
                            )
            ret = sparql.queryAndConvert()
            for binding in ret["results"]["bindings"]:
                qid = binding["wikidata_id"]["value"]
                mapping[binding["wikipedia_id"]["value"]] = qid[qid.rfind("Q"):]
            batch = []
    if batch:
        concat = " ".join(batch)
        sparql.setQuery(f"""
                        PREFIX wd: <http://www.wikidata.org/entity/> 
                        SELECT ?wikidata_id ?wikipedia_id WHERE {{
                            VALUES ?wikipedia_id {{{concat}}} 
                            ?dbpedia_id owl:sameAs ?wikidata_id  .
                            ?dbpedia_id dbo:wikiPageID ?wikipedia_id .
                            FILTER strstarts(str(?wikidata_id),str(wd:))
                        }}
                        """
                        )
        ret = sparql.queryAndConvert()
        for binding in ret["results"]["bindings"]:
            mapping[binding["wikipedia_id"]["value"]] = binding["wikidata_id"]["value"]
    return mapping