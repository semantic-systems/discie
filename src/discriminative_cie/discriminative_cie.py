import argparse
import enum
import json
import time

import math
import os
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import faiss
import jsonlines
import networkx
import numpy as np
import torch
import wandb
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from src.candidate_generation.candidate_generator import init_model, create_faiss_index_for_entity_descriptions, \
    load_property_indices, load_entity_descriptions, get_mention_input, CustomExample, get_type_dictionary, \
    get_boundaries
from src.candidate_generation.pl_candidate_generator import PLCrossEncoder
from src.mention_recognizer.pairwise_mention_recognizer import PairwiseMentionRecognizer
import hashlib
import matplotlib.pyplot as plt

from src.relation_extraction.relation_extraction import PairwiseRelationRecognizer


class DisambiguationMode(enum.Enum):
    """
    Enum class for disambiguation mode
    """
    GRAPH = "graph"
    SIMPLE = "simple"


def load_restrictions(restrictions: str) -> Optional[set]:
    if restrictions is None:
        return None
    return set(json.load(open(restrictions)))


class DiscriminativeCIE:
    def __init__(self, path_to_bi_encoder, path_to_mention_recognizer, path_to_cross_encoder,
                 path_to_relation_extractor = None,num_candidates=10,
                 mention_threshold=0.5, combined_threshold=0.5, property_threshold=0.5, all_qids=None,
                 types_index=None, disambiguation_mode=DisambiguationMode.SIMPLE,
                 include_property_scores=False,
                 include_mention_scores=False,
                 alt_num_types: int = None,
                 entity_restrictions: str=None,
                 property_restrictions: str=None,
                 spoof_boundaries: bool = False,
                 spoof_candidates: bool = False,
                 spoof_linking: bool = False,
                 alternative_relation_extractor: bool = False,
                 alternative_relation_extractor_use_types: bool = True,
                 alternative_relation_extractor_deactivate_text: bool = False,
                 only_one_relation_per_pair: bool = False,):

        self.index_name = "indices/" + hashlib.md5(Path(path_to_bi_encoder).name.encode('utf-8')).hexdigest()[0:10]
        print(self.index_name)
        relation_counts = json.load(open("data/relation_counts.json"))
        rel_id2surface_form = {item["information"]["en_title"]: item["wikidata_id"] for item in jsonlines.open("data/rel_id2surface_form.jsonl")}
        rel_id2surface_form_inverse = {v: k for k, v in rel_id2surface_form.items()}
        self.relation_counts = {rel_id2surface_form[k]: v for k, v in relation_counts.items()}
        self.rel_names_descending = [i[0] for i in sorted(list(self.relation_counts.items()), key=lambda x: -x[1])]
        self.bin_edges = self.get_bin_edges(max(self.relation_counts.values()))
        self.bucket_ids = list(range(len(self.bin_edges) - 1))
        self.bucket_id2num_ref_rels = Counter(self.get_bucket_ids(self.rel_names_descending))

        self.entity_restrictions = load_restrictions(entity_restrictions)
        self.property_restrictions = load_restrictions(property_restrictions)
        self.property_indices = load_property_indices()
        self.only_one_relation_per_pair = only_one_relation_per_pair
        self.spoof_boundaries = spoof_boundaries
        self.spoof_candidates = spoof_candidates
        self.spoof_linking = spoof_linking
        self.num_bootstrap_samples = 50
        self.property_indices_inverse = {v: k for k, v in self.property_indices.items()}
        relations_to_mask = None
        if self.property_restrictions is not None:
            relations_to_mask = set()
            for key, value in self.property_indices.items():
                if key not in self.property_restrictions:
                    relations_to_mask.add(value)
            relations_to_mask = list(relations_to_mask)
        self.entity_descriptions = load_entity_descriptions()
        self.bi_encoder = init_model(path_to_bi_encoder)
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        self.mention_recognizer = PairwiseMentionRecognizer.load_from_checkpoint(path_to_mention_recognizer,
                                                                                 model_name="distilbert-base-cased",
                                                                                 map_location=torch.device('cpu'))
        self.alternative_relation_extractor = alternative_relation_extractor
        self.alternative_relation_extractor_use_types = alternative_relation_extractor_use_types
        self.alternative_relation_extractor_deactivate_text = alternative_relation_extractor_deactivate_text
        if not os.path.exists(self.index_name):
            os.makedirs(self.index_name)

        index_path = self.index_name + "/faiss.index"
        indices_path = self.index_name + "/faiss.indices"
        if not os.path.exists(index_path):
            create_faiss_index_for_entity_descriptions(self.bi_encoder, True, index_path, indices_path)
        print("Loading faiss index")
        self.faiss_index = faiss.read_index(index_path)
        self.entity_indices = json.load(open(indices_path))
        print("Done loading faiss index")

        if types_index is not None:
            print("Loading types index")
            self.types_dictionary, self.types_index = get_type_dictionary(all_qids, types_index)
            print("Done loading types index")
        else:
            self.types_index = None
            self.types_dictionary = {}
        num_types = len(self.types_index) if alt_num_types is None else alt_num_types

        self.cross_encoder = PLCrossEncoder.load_from_checkpoint(path_to_cross_encoder,
                                                                 model_name="sentence-transformers/all-MiniLM-L12-v2",
                                                                 num_properties=len(self.property_indices),
                                                                 entity_descriptions=self.entity_descriptions,
                                                                 epochs_of_training_before_regeneration=-1,
                                                                 batch_size=32,
                                                                 number_of_types=num_types,
                                                                 relations_to_mask=relations_to_mask,
                                                                 map_location=torch.device('cpu')
                                                                 )
        self.relation_extractor = None
        if path_to_relation_extractor is not None:
            if not self.alternative_relation_extractor:
                self.relation_extractor =  PLCrossEncoder.load_from_checkpoint(path_to_relation_extractor,
                                                                     model_name="sentence-transformers/all-MiniLM-L12-v2",
                                                                     num_properties=len(self.property_indices),
                                                                     entity_descriptions=self.entity_descriptions,
                                                                     epochs_of_training_before_regeneration=-1,
                                                                     batch_size=32,
                                                                    number_of_types=len(self.types_index) if self.types_index is not None else None,
                                                                               relations_to_mask=relations_to_mask,
                                                                               map_location=torch.device('cpu')
                                                                     )
            else:
                if not self.alternative_relation_extractor_use_types:
                    types_index = None
                self.relation_extractor = PairwiseRelationRecognizer.load_from_checkpoint(path_to_relation_extractor,
                                                                                         model_name="distilbert-base-cased",
                                                                                         num_relations=len(self.property_indices),
                                                                                         number_of_types=len(
                                                                                             types_index) if types_index else None,
                                                                                         deactivate_text=self.alternative_relation_extractor_deactivate_text,
                                                                                          map_location=torch.device(
                                                                                              'cpu')
                                                                                         )


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cross_encoder.cuda()
            self.bi_encoder.cuda()
            self.mention_recognizer.cuda()
            if self.relation_extractor is not None:
                self.relation_extractor.cuda()
        self.cross_encoder.eval()
        self.cross_encoder.freeze()
        if self.relation_extractor is not None:
            self.relation_extractor.eval()
            self.relation_extractor.freeze()
        self.bi_encoder.eval()
        self.mention_recognizer.eval()

        self.num_candidates = num_candidates
        self.mention_threshold = mention_threshold
        self.combined_threshold = combined_threshold
        self.property_threshold = property_threshold
        self.include_property_scores = include_property_scores
        self.include_mention_scores = include_mention_scores
        self.first_property_threshold = 0.1
        self.disambiguation_mode = disambiguation_mode

    def get_mention_tokens(self, mention_matrices: torch.Tensor, attention_matrices: torch.Tensor) -> List[List[Tuple[int, int, float]]]:
        mention_tokens = []
        for mention_matrix, attention_matrix in zip(mention_matrices, attention_matrices):
            scores = torch.sigmoid(mention_matrix)
            attention_matrix = attention_matrix.triu(diagonal=0)
            scores = scores * attention_matrix
            scores = scores.cpu().detach().numpy()
            mention_tokens_for_text = []
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    if scores[i][j] > self.mention_threshold:
                        mention_tokens_for_text.append((i if i != 0 else i + 1, j, scores[i][j]))
            mention_tokens.append(mention_tokens_for_text)
        return mention_tokens

    def get_candidates(self, bi_encoder_input: List[List[str]]) -> Dict[int, dict]:
        elongated_bi_encoder_input = []
        example_ids = []
        mention_ids = []
        for idx, context_representations in enumerate(bi_encoder_input):
            for ment_idx, context_representation in enumerate(context_representations):
                elongated_bi_encoder_input.append(context_representation)
                example_ids.append(idx)
                mention_ids.append(ment_idx)

        encoded_sentences = self.bi_encoder.encode(elongated_bi_encoder_input, batch_size=256)
        # if self.bi_encoder.normalize:
        encoded_sentences = encoded_sentences / np.linalg.norm(encoded_sentences, axis=1).reshape(-1, 1)

        # Run the following batch wise
        n = len(encoded_sentences)
        k = self.num_candidates if self.entity_restrictions is None else 1000
        scores = np.empty((n, k), dtype=np.float32)
        indices = np.empty((n, k), dtype=np.int64)

        chunk_size = 10000  # Adjust this value based on the size of your dataset and available memory

        for i in range(0, n, chunk_size):
            start = i
            end = min(i + chunk_size, n)
            chunk_scores, chunk_indices = self.faiss_index.search(encoded_sentences[start:end], k)

            scores[start:end] = chunk_scores
            indices[start:end] = chunk_indices

        candidate_set = defaultdict(dict)
        for document_id, mention_id, score, idx in zip(example_ids, mention_ids, scores, indices):
            # Make identifier consisting of document_id and span
            candidates = []
            for i, (index, score_) in enumerate(zip(idx, score)):
                if self.entity_restrictions is not None and self.entity_indices[str(index)] not in self.entity_restrictions:
                    continue
                candidates.append((self.entity_indices[str(index)], score_))
                if i >= self.num_candidates - 1:
                    break
            candidate_set[document_id][mention_id] = candidates
        return candidate_set

    def prepare_bi_encoder_input(self, texts,
                                 all_boundaries) -> List[list]:
        bi_encoder_input = []

        for text, boundaries in zip(texts, all_boundaries):
            context_representations = []
            for start, end in boundaries:
                context_representations.append(get_mention_input(text, start, end))
            bi_encoder_input.append(context_representations)
        return bi_encoder_input

    def prepare_cross_encoder_input(self, candidates, all_boundaries, texts):
        all_text_representations = []
        all_example_indices = []
        all_mention_indices = []
        all_candidate_types = []
        batched_mention_indices = []
        batch_all_qids = []
        all_qids = []
        for example_idx, (text, boundaries) in enumerate(zip(texts, all_boundaries)):
            mention_indices = []
            candidate_types = []
            qids=  []
            for mention_idx, (start, end) in enumerate(boundaries):
                context_representation = get_mention_input(text, start, end)
                candidates_of_mention = candidates[example_idx][mention_idx]
                for qid, score in candidates_of_mention:
                    if qid in self.entity_descriptions:
                        entity_representation = self.entity_descriptions[qid]
                        concatenated_representation = context_representation + " {} ".format(
                            self.cross_encoder.tokenizer.sep_token) + entity_representation
                        all_example_indices.append(example_idx)
                        all_text_representations.append(concatenated_representation)
                        all_mention_indices.append(mention_idx)
                        all_qids.append(qid)
                        qids.append(qid)
                        mention_indices.append(mention_idx)
                        if qid in self.types_dictionary:
                            candidate_types.append(self.types_dictionary[qid])
                        else:
                            candidate_types.append([])
            all_candidate_types.append(candidate_types)
            batched_mention_indices.append(mention_indices)
            batch_all_qids.append(qids)
        all_text_representations = self.cross_encoder.tokenizer.batch_encode_plus(all_text_representations,
                                                                                      return_tensors="pt",
                                                                                      padding=True)
        maximum_num_mentions = max([len(item) for item in all_candidate_types])
        maximum_num_types = max([len(types) for item in all_candidate_types for types in item])
        mention_types = -torch.ones((len(all_candidate_types), maximum_num_mentions, maximum_num_types),
                                    dtype=torch.long, device=self.device)
        for i, item in enumerate(all_candidate_types):
            for j, types in enumerate(item):
                for k, type in enumerate(types):
                    mention_types[i, j, k] = type
        return all_text_representations, all_example_indices, all_mention_indices, batched_mention_indices, batch_all_qids, all_qids, mention_types

    def get_tokenized_text(self, texts, boundaries_list, qids_list):
        tokens = self.relation_extractor.tokenizer(texts, return_offsets_mapping=True, padding=True, truncation=True,
                                                   return_tensors="pt")
        all_mention_types = []
        all_mention_tokens = []
        for idx, (offsets, boundaries) in enumerate(zip(tokens["offset_mapping"], boundaries_list)):
            qids = qids_list[idx]
            mention_types = []
            mention_tokens = []
            for idx_, (entity_start, entity_end) in enumerate(boundaries):
                qids_ = qids[idx_]
                # Find the first token whose start offset matches or exceeds the entity start position
                start_token = next((i for i, (start, _) in enumerate(offsets) if start >= entity_start and i != 0), None)

                # Find the last token whose end offset matches or exceeds the entity end position
                end_token = next((i for i, (_, end) in enumerate(offsets) if end >= entity_end and i != 0), None)

                if self.types_dictionary is not None:
                    for qid, _ in qids_:
                        if qid in self.types_dictionary:
                            mention_types.append(self.types_dictionary[qid])
                        else:
                            mention_types.append([])
                # Ensure both start_token and end_token are not None
                if start_token is not None and end_token is not None:
                    mention_tokens.append((start_token, end_token))
                else:
                    raise Exception("Invalid boundaries")

            all_mention_types.append(mention_types)
            all_mention_tokens.append(mention_tokens)
        all_mention_tokens = pad_sequence([torch.tensor(item) for item in all_mention_tokens], batch_first=True, padding_value=-1)
        num_types = [len(types) for item in all_mention_types for types in item]
        if not num_types:
            maximum_num_types = 0
        else:
            maximum_num_types = max(num_types)
        maximum_num_mentions = max([len(item) for item in all_mention_types])
        mention_types = -torch.ones((len(all_mention_types), maximum_num_mentions, maximum_num_types), dtype=torch.long)
        for i, item in enumerate(all_mention_types):
            for j, types in enumerate(item):
                for k, type in enumerate(types):
                    mention_types[i, j, k] = type
        return all_mention_tokens, mention_types, tokens

    def get_cross_encoder_scores(self, candidates, all_boundaries, texts):
        sentence_features, example_indices, mention_indices, batched_mention_indices, batched_all_qids, all_qids, all_candidate_types = self.prepare_cross_encoder_input(candidates, all_boundaries, texts)
        if all_candidate_types is not None:
            if all_candidate_types.size(2) == 0:
                all_candidate_types = -torch.ones((all_candidate_types.size(0), all_candidate_types.size(1), 1),
                                                  dtype=torch.long, device=self.device)
        if self.relation_extractor is not None:
            scores, _ = self.cross_encoder(sentence_features, example_indices)
            if self.alternative_relation_extractor:
                mention_positions, all_candidate_types, text_features = self.get_tokenized_text(texts, all_boundaries, candidates)
                alignment = None
                if mention_positions.size(1) != all_candidate_types.size(1):
                    alignment = all_candidate_types.size(1) // mention_positions.size(1)
                mention_positions = mention_positions.to(self.device)
                all_candidate_types = all_candidate_types.to(self.device)
                property_scores, padding_matrix = self.relation_extractor(mention_positions,
                                                                          all_candidate_types,
                                                                          input_ids=text_features["input_ids"].to(self.device),
                                                                          attention_mask=text_features["attention_mask"].to(self.device),
                                                                          alignment=alignment)
            else:
                _, property_scores = self.relation_extractor(sentence_features, example_indices, all_candidate_types)
        else:
            scores, property_scores = self.cross_encoder(sentence_features, example_indices, all_candidate_types)
        all_pairs = {example_index: dict() for example_index in range(len(texts))}
        property_scores = torch.sigmoid(property_scores).cpu().detach().numpy()
        scores = torch.sigmoid(scores).cpu().detach().numpy()
        for idx, (all_qids_, mention_indices_, property_scores_) in enumerate(zip(batched_all_qids, batched_mention_indices, property_scores)):
            for i in range(len(property_scores_)):
                for j in range(len(property_scores_)):
                    if i < len(mention_indices_) and j < len(mention_indices_):
                        if (mention_indices_[i] != mention_indices_[j]) or (i == j):
                            all_pairs[idx][(mention_indices_[i], all_qids_[i], mention_indices_[j], all_qids_[j] )] = property_scores_[i][j]
        score_dict = {example_index: defaultdict(list) for example_index in range(len(texts))}
        for score, mention_idx, example_idx, qid in zip(scores, mention_indices, example_indices, all_qids):
            if self.spoof_linking:
                score_dict[example_idx][mention_idx].append((qid, 1.0))
            else:
                score_dict[example_idx][mention_idx].append((qid, score))
        return score_dict, all_pairs


    def create_graphs(self, ce_scores, all_pairs, mention_tokens):
        graphs = []
        for example_idx, scores in ce_scores.items():
            graph = networkx.MultiDiGraph()

            included_nodes = set()
            nodes = []
            for mention_idx, scores_ in scores.items():
                for qid, score in scores_:
                    if self.combine_mention_and_ce_score(score, mention_tokens[example_idx][mention_idx][2]) > self.combined_threshold:
                        nodes.append(((qid, mention_idx),
                                      {"type": "entity", "ce_score": score, "mention_score": mention_tokens[example_idx][mention_idx][2],
                                       "mention_idx": mention_idx}))
                        included_nodes.add((qid, mention_idx))
            graph.add_nodes_from(nodes)
            edges = []
            for (mention_idx, qid, mention_idx_, qid_), property_scores in all_pairs[example_idx].items():
                if (qid, mention_idx) in included_nodes and (qid_, mention_idx_) in included_nodes:
                    for idx, score in enumerate(property_scores):
                        if score > self.first_property_threshold:
                            property = self.property_indices_inverse[idx]
                            edges.append(((qid, mention_idx), (qid_, mention_idx_),
                                          {"key": property, "property_score": score}))
            graph.add_edges_from(edges)
            graphs.append(graph)
        return graphs


    def prune_graphs(self, graphs, property_threshold=None,  combined_threshold=None):
        if property_threshold is None:
            property_threshold = self.property_threshold
        if combined_threshold is None:
            combined_threshold = self.combined_threshold
        pruned_graphs = []
        for graph in graphs:
            copied_graph = graph.copy()
            for u, v, key, attr in graph.edges(data=True, keys=True):
                if attr["property_score"] < property_threshold:
                    copied_graph.remove_edge(u, v, key=key)
            for node in graph.nodes():
                if self.combine_mention_and_ce_score(copied_graph.nodes[node]["ce_score"],
                                                     copied_graph.nodes[node]["mention_score"]) < combined_threshold:
                    copied_graph.remove_node(node)

            pruned_graphs.append(copied_graph)
        return pruned_graphs

    def disambiguate(self, scores, all_pairs, mention_tokens, property_threshold=None, combined_threshold=None):
        if self.disambiguation_mode == DisambiguationMode.SIMPLE:
            return self.disambiguate_no_graph_simple(scores, all_pairs, mention_tokens,
                                                     property_threshold=property_threshold,
                                                     combined_threshold=combined_threshold)
        else:
            graphs = self.create_graphs(scores, all_pairs, mention_tokens)
            graphs = self.disambiguate_graphs(graphs)
            graphs = self.prune_graphs(graphs, property_threshold=property_threshold, combined_threshold=combined_threshold)
            return graphs


    def disambiguate_graphs_simple(self, graphs):
        disambiguated_graphs = []
        for graph in graphs:
            copied_graph = graph.copy()
            for u, v, key, attr in graph.edges(data=True, keys=True):
                if key == "P31" or attr["property_score"] < self.property_threshold:
                    copied_graph.remove_edge(u, v, key=key)
            # Prune nodes of the same mention based on candidate score
            for mention_idx in set([data["mention_idx"] for _, data in graph.nodes(data=True)]):
                nodes = [node for node, data in graph.nodes(data=True) if data["mention_idx"] == mention_idx]
                scores = [data["ce_score"] for _, data in graph.nodes(data=True) if data["mention_idx"] == mention_idx]
                max_score = max(scores)
                for node, score in zip(nodes, scores):
                    if score < max_score:
                        copied_graph.remove_node(node)
            disambiguated_graphs.append(copied_graph)
        return disambiguated_graphs

    def combine_mention_and_ce_score(self, mention_score, ce_score):
        return (mention_score + ce_score) / 2

    def combine_property_scores(self, property_scores):
        return sum(property_scores)

    def disambiguate_graphs(self, graphs, top_k=5):
        disambiguated_graphs = []
        for graph in graphs:
            copied_graph = graph.copy()
            # Get subgraph pairs by mention_idx
            nodes_per_subgraph = []
            for mention_idx in {data["mention_idx"] for _, data in graph.nodes(data=True)}:
                nodes = [node for node, data in graph.nodes(data=True) if data["mention_idx"] == mention_idx]
                nodes_per_subgraph.append(nodes)
            current_candidates = []
            for idx, subgraph_1 in enumerate(nodes_per_subgraph):
                if not current_candidates:
                    # Calculate the aggregated score of each node in subgraph_1 to all nodes in the other subgraphs
                    scores = []
                    node_scores = {}
                    for node in subgraph_1:
                        score = 0
                        for subgraph_2 in nodes_per_subgraph[idx+1:]:
                            property_scores = []
                            for node_2 in subgraph_2:
                                if graph.has_edge(node, node_2):
                                    property_scores.append(self.combine_property_scores([x["property_score"] for x in graph.get_edge_data(node, node_2).values()]))
                            score += max(property_scores, default=0)
                        scores.append((node, score))
                        ce_score = graph.nodes[node]["ce_score"]
                        mention_score = graph.nodes[node]["mention_score"]
                        node_scores[node] = ce_score + mention_score
                    top_k_nodes = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
                    current_candidates = [([x[0]], node_scores[x[0]]) for x in top_k_nodes]
                else:
                    new_candidates = []
                    for nodes, score in current_candidates:
                        for node in subgraph_1:
                            property_scores = []
                            for node_ in nodes:
                                if graph.has_edge(node, node_):
                                    property_scores.append(self.combine_property_scores([x["property_score"] for x in graph.get_edge_data(node, node_).values()]))
                            score_ = max(property_scores, default=0)
                            nodes_ = nodes + [node]
                            ce_score = graph.nodes[node]["ce_score"]
                            mention_score = graph.nodes[node]["mention_score"]
                            score_ = score + score_ + ce_score + mention_score
                            new_candidates.append((nodes_, score_))
                    current_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:top_k]
            if current_candidates:
                final_nodes = current_candidates[0][0]
            else:
                final_nodes = set()
            for node in list(copied_graph.nodes()):
                if node not in final_nodes:
                    copied_graph.remove_node(node)
            disambiguated_graphs.append(copied_graph)
        return disambiguated_graphs

    def disambiguate_no_graph_simple(self, ce_scores, all_pairs, mention_tokens, average: bool = True,
                                     property_threshold=None, combined_threshold=None,
                                     kwargs: Dict = None):
        if combined_threshold is None:
            combined_threshold = self.combined_threshold
        if property_threshold is None:
            property_threshold = self.property_threshold
        outputs = []
        for example_idx, scores in ce_scores.items():
            triples = set()
            all_occuring_entities = set()

            property_scores_to_include = {}
            if self.include_property_scores:
                for mention_idx, mention_idx_, qid, qid_, property_scores in all_pairs[example_idx]:
                    maximum_property_score = max(property_scores)
                    property_scores_to_include[(qid, mention_idx)] = max(maximum_property_score, property_scores_to_include.get((qid, mention_idx), 0))
                    property_scores_to_include[(qid_, mention_idx_)] = max(maximum_property_score, property_scores_to_include.get((qid_, mention_idx_), 0))
            for mention_idx, scores_ in scores.items():
                qids, scores__  = zip(*scores_)
                maximum_idx = np.argmax(scores__)
                maximum_score = scores__[maximum_idx]
                qid = qids[maximum_idx]
                scores_to_combine = [maximum_score]
                if self.include_mention_scores:
                    scores_to_combine.append(mention_tokens[example_idx][mention_idx][2])
                if property_scores_to_include:
                    if (qid, mention_idx) in property_scores_to_include:
                        scores_to_combine.append(property_scores_to_include[(qid, mention_idx)])
                if average:
                    combined_score = sum(scores_to_combine) / len(scores_to_combine)
                else:
                    combined_score = max(scores_to_combine)
                if combined_score > combined_threshold:
                    all_occuring_entities.add((qid, mention_idx))
            for qid, mention_idx in all_occuring_entities:
                for qid_, mention_idx_ in all_occuring_entities:
                    key = (mention_idx, qid, mention_idx_, qid_ )
                    if key in all_pairs[example_idx]:
                        property_scores = all_pairs[example_idx][key]
                        above_threshold_indices = np.where(property_scores > property_threshold)[0]

                        if above_threshold_indices.size > 0:
                            properties = [self.property_indices_inverse[x] for x in above_threshold_indices]
                            above_threshold_property_scores = property_scores[above_threshold_indices]
                            triples.update(zip([qid] * len(properties), properties, [qid_] * len(properties),
                                               above_threshold_property_scores))
            if self.only_one_relation_per_pair:
                pairwise_dict = defaultdict(list)
                for triple in triples:
                    key = tuple(sorted([triple[0], triple[2]]))
                    pairwise_dict[key].append(triple)
                triples = set()
                for key, triples_ in pairwise_dict.items():
                    max_triple = max(triples_, key=lambda x: x[3])
                    triples.add((max_triple[0], max_triple[1], max_triple[2]))
            all_occuring_entities = {x[0] for x in all_occuring_entities}
            outputs.append((triples, all_occuring_entities))
        return outputs

    def create_output(self, disambiguated):
        for graph in disambiguated:
            for node in graph.nodes(data=True):
                print(node)
                outgoing_edges = graph.out_edges(node[0], data=True, keys=True)
                print(outgoing_edges)
            networkx.draw(graph, with_labels=True,
                          pos=networkx.circular_layout(graph))
            plt.show()

    def get_boundaries(self, texts, mention_tokens, tokenized):
        offset_mappings = tokenized["offset_mapping"]
        all_boundaries = []
        for text, mention_tokens_, offset_mappings_ in zip(texts, mention_tokens, offset_mappings):
            boundaries = []
            for start, end, score in mention_tokens_:
                start = offset_mappings_[start][0]
                end = offset_mappings_[end][1]
                boundaries.append((start, end))
            all_boundaries.append(boundaries)
        return all_boundaries

    def run_and_visualize(self, texts):
        disambiguated, _ = self.run(texts)
        self.create_output(disambiguated)

    def get_triples_and_qids(self, graph):
        triples = []
        for u, v, key, attr in graph.edges(data=True, keys=True):
            triples.append((u[0], attr["key"], v[0]))
        qids = set()
        for node in graph.nodes():
            qids.add(node[0])
        return triples, qids
    def calculate_metrics(self, disambiguated_graphs, all_gt_triples, candidate_entities, identifiers, filter_set=None,
                          return_details: bool = False):
        if filter_set is None:
            filter_set = {"P31"}#, "P106", "P641", "P17"}
        tp_filtered = 0
        fp_filtered = 0
        fn_filtered = 0
        tp = 0
        fp = 0
        fn = 0
        tp_entity = 0
        fp_entity = 0
        fn_entity = 0
        tp_entity_triples = 0
        fp_entity_triples = 0
        fn_entity_triples = 0
        tp_property = 0
        fp_property = 0
        fn_property = 0
        gt_candidate_included = 0
        per_property_tp = defaultdict(int)
        per_property_fp = defaultdict(int)
        per_property_fn = defaultdict(int)
        per_bucket_tp = {x: 0 for x in range(len(self.bucket_ids))}
        per_bucket_fp = {x: 0 for x in range(len(self.bucket_ids))}
        per_bucket_fn = {x: 0 for x in range(len(self.bucket_ids))}
        per_bucket_counter = {x: 0 for x in range(len(self.bucket_ids))}

        detailed_results = []

        assert len(disambiguated_graphs) == len(all_gt_triples), f"{len(disambiguated_graphs)} != {len(all_gt_triples)}"
        for graph, gt_triples, candidates, identifier in zip(disambiguated_graphs, all_gt_triples, candidate_entities, identifiers):
            if isinstance(graph, networkx.MultiDiGraph):
                triples, all_occuring_entities = self.get_triples_and_qids(graph)
            else:
                triples, all_occuring_entities = graph
            all_occuring_entities_in_triples = set([x[0] for x in triples] + [x[2] for x in triples])
            all_occuring_properties = set([x[1] for x in triples])
            all_occuring_entities_gt = set([x[0] for x in gt_triples] + [x[2] for x in gt_triples])
            gt_candidate_included += len(set(candidates).intersection(all_occuring_entities_gt))
            all_occuring_properties_gt = set([x[1] for x in gt_triples])
            triples_filtered = set()
            for triple in triples:
                if triple[1] not in filter_set:
                    triples_filtered.add(triple)
            gt_triples_filtered = set()
            for triple in gt_triples:
                if triple[1] not in filter_set:
                    gt_triples_filtered.add(triple)
            tp_filtered += len(set(triples_filtered).intersection(set(gt_triples_filtered)))
            fp_filtered += len(set(triples_filtered).difference(set(gt_triples_filtered)))
            fn_filtered += len(set(gt_triples_filtered).difference(set(triples_filtered)))
            tp += len(set(triples).intersection(set(gt_triples)))
            fp += len(set(triples).difference(set(gt_triples)))
            fn += len(set(gt_triples).difference(set(triples)))
            tp_entity += len(all_occuring_entities.intersection(all_occuring_entities_gt))
            fp_entity += len(all_occuring_entities.difference(all_occuring_entities_gt))
            fn_entity += len(all_occuring_entities_gt.difference(all_occuring_entities))
            tp_entity_triples += len(all_occuring_entities_in_triples.intersection(all_occuring_entities_gt))
            fp_entity_triples += len(all_occuring_entities_in_triples.difference(all_occuring_entities_gt))
            fn_entity_triples += len(all_occuring_entities_gt.difference(all_occuring_entities_in_triples))
            tp_property += len(all_occuring_properties.intersection(all_occuring_properties_gt))
            fp_property += len(all_occuring_properties.difference(all_occuring_properties_gt))
            fn_property += len(all_occuring_properties_gt.difference(all_occuring_properties))

            fp_triples = set()
            fn_triples = set()
            for triple in set(triples).intersection(set(gt_triples)):
                per_property_tp[triple[1]] += 1
                bucket_id = self.get_bucket_id_for_rel_name(triple[1])
                per_bucket_tp[bucket_id] += 1
                per_bucket_counter[bucket_id] += 1
            for triple in set(triples).difference(set(gt_triples)):
                per_property_fp[triple[1]] += 1
                fp_triples.add(triple)
                bucket_id = self.get_bucket_id_for_rel_name(triple[1])
                per_bucket_fp[bucket_id] += 1
                per_bucket_counter[bucket_id] += 1
            for triple in set(gt_triples).difference(set(triples)):
                per_property_fn[triple[1]] += 1
                fn_triples.add(triple)
                bucket_id = self.get_bucket_id_for_rel_name(triple[1])
                per_bucket_fn[bucket_id] += 1
                per_bucket_counter[bucket_id] += 1
            fp_triples = [list(x) for x in fp_triples]
            fn_triples = [list(x) for x in fn_triples]
            if fp_triples or fn_triples:
                detailed_results.append({"identifier": identifier, "fp": fp_triples, "fn": fn_triples})
        precision = (tp / (tp + fp)) if tp + fp > 0 else 0
        recall = (tp / (tp + fn)) if tp + fn > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
        f2 = 5 * ((precision * recall) / (4 * precision + recall)) if 4 * precision + recall > 0 else 0
        macro_precision = sum([per_property_tp[x] / (per_property_tp[x] + per_property_fp[x]) if per_property_tp[
                                                                                                     x] +
                                                                                                 per_property_fp[
                                                                                                     x] > 0 else 0
                               for x in per_property_tp]) / len(per_property_tp)
        macro_recall = sum([per_property_tp[x] / (per_property_tp[x] + per_property_fn[x]) if per_property_tp[x] +
                                                                                              per_property_fn[
                                                                                                  x] > 0 else 0 for
                            x in per_property_tp]) / len(per_property_tp)
        macro_f1 = 2 * ((macro_precision * macro_recall) / (
                macro_precision + macro_recall)) if macro_precision + macro_recall > 0 else 0
        macro_f2 = 5 * ((macro_precision * macro_recall) / (
                4 * macro_precision + macro_recall)) if macro_precision + macro_recall > 0 else 0
        per_property_recall = {x: per_property_tp[x] / (per_property_tp[x] + per_property_fn[x]) if per_property_tp[x] + per_property_fn[x] > 0 else 0 for x in per_property_tp}
        per_property_precision = {x: per_property_tp[x] / (per_property_tp[x] + per_property_fp[x]) if per_property_tp[x] + per_property_fp[x] > 0 else 0 for x in per_property_tp}
        per_property_f1 = {x: 2 * ((per_property_precision[x] * per_property_recall[x]) / (per_property_precision[x] + per_property_recall[x])) if per_property_precision[x] + per_property_recall[x] > 0 else 0 for x in per_property_tp}
        per_property_f2 = {x: 5 * ((per_property_precision[x] * per_property_recall[x]) / (4 * per_property_precision[x] + per_property_recall[x])) if 4 * per_property_precision[x] + per_property_recall[x] > 0 else 0 for x in per_property_tp}
        precision_filtered = (tp_filtered / (tp_filtered + fp_filtered)) if tp_filtered + fp_filtered > 0 else 0
        recall_filtered = (tp_filtered / (tp_filtered + fn_filtered)) if tp_filtered + fn_filtered > 0 else 0
        f1_filtered = 2 * ((precision_filtered * recall_filtered) / (
                    precision_filtered + recall_filtered)) if precision_filtered + recall_filtered > 0 else 0
        precision_entity = (tp_entity / (tp_entity + fp_entity)) if tp_entity + fp_entity > 0 else 0
        recall_entity = (tp_entity / (tp_entity + fn_entity)) if tp_entity + fn_entity > 0 else 0
        f1_entity = 2 * ((precision_entity * recall_entity) / (
                    precision_entity + recall_entity)) if precision_entity + recall_entity > 0 else 0
        precision_entity_triples = (tp_entity_triples / (
                    tp_entity_triples + fp_entity_triples)) if tp_entity_triples + fp_entity_triples > 0 else 0
        recall_entity_triples = (tp_entity_triples / (
                    tp_entity_triples + fn_entity_triples)) if tp_entity_triples + fn_entity_triples > 0 else 0
        f1_entity_triples = 2 * ((precision_entity_triples * recall_entity_triples) / (
                    precision_entity_triples + recall_entity_triples)) if precision_entity_triples + recall_entity_triples > 0 else 0
        precision_property = (tp_property / (tp_property + fp_property)) if tp_property + fp_property > 0 else 0
        recall_property = (tp_property / (tp_property + fn_property)) if tp_property + fn_property > 0 else 0
        f1_property = 2 * ((precision_property * recall_property) / (
                    precision_property + recall_property)) if precision_property + recall_property > 0 else 0
        candidate_recall = (gt_candidate_included / (tp_entity + fn_entity)) if tp_entity + fn_entity > 0 else 0
        per_property_metrics = {x: {"precision": per_property_precision[x], "recall": per_property_recall[x], "f1": per_property_f1[x], "f2": per_property_f2[x],} for x in per_property_tp}
        per_bucket_precision = {x: per_bucket_tp[x] / (per_bucket_tp[x] + per_bucket_fp[x]) if per_bucket_tp[x] + per_bucket_fp[x] > 0 else 0 for x in per_bucket_tp}
        per_bucket_recall = {x: per_bucket_tp[x] / (per_bucket_tp[x] + per_bucket_fn[x]) if per_bucket_tp[x] + per_bucket_fn[x] > 0 else 0 for x in per_bucket_tp}
        per_bucket_f1 = {x: 2 * ((per_bucket_precision[x] * per_bucket_recall[x]) / (per_bucket_precision[x] + per_bucket_recall[x])) if per_bucket_precision[x] + per_bucket_recall[x] > 0 else 0 for x in per_bucket_tp}
        per_bucket_f2 = {x: 5 * ((per_bucket_precision[x] * per_bucket_recall[x]) / (4 * per_bucket_precision[x] + per_bucket_recall[x])) if 4 * per_bucket_precision[x] + per_bucket_recall[x] > 0 else 0 for x in per_bucket_tp}
        per_bucket_metrics = {x: {"precision": per_bucket_precision[x], "recall": per_bucket_recall[x], "f1": per_bucket_f1[x], "f2": per_bucket_f2[x], "freq": self.bin_edges[x],
                                  "num_relations": self.bucket_id2num_ref_rels[x], "counter": per_bucket_counter[x]} for x in per_bucket_tp.keys()}
        results = {"precision": precision, "recall": recall, "f1": f1, "f2": f2,
                "precision_entity": precision_entity, "recall_entity": recall_entity, "f1_entity": f1_entity,
                "precision_entity_triples": precision_entity_triples, "recall_entity_triples": recall_entity_triples, "f1_entity_triples": f1_entity_triples,
                "precision_property": precision_property, "recall_property": recall_property, "f1_property": f1_property,
                "precision_filtered": precision_filtered, "recall_filtered": recall_filtered, "f1_filtered": f1_filtered,
                "candidate_recall": candidate_recall, "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1, "macro_f2": macro_f2,}
        if return_details:
            return results, per_bucket_metrics, detailed_results
        return results, per_bucket_metrics
    def alt_benchmark(self, examples, property_thresholds, combined_thresholds):
        all_disambiguated_graphs, candidate_entities = self.run_with_thresholds(examples, property_thresholds, combined_thresholds)
        all_gt_triples = [x["triples"] for x in examples]
        identifiers = [x["identifier"] for x in examples]
        results = []
        for (property_threshold, combined_threshold), disambiguated_graphs in all_disambiguated_graphs.items():
            stats, per_property_metrics = self.calculate_metrics(disambiguated_graphs, all_gt_triples, candidate_entities, identifiers)
            stats["property_threshold"] = property_threshold
            stats["combined_threshold"] = combined_threshold
            results.append(stats)
        return results

    def bootstrap_calculate_metrics(self, disambiguated_graphs, all_gt_triples, candidate_entities, identifiers):
        num_datapoints = len(disambiguated_graphs)
        random.seed(123)
        all_metrics = defaultdict(list)
        all_per_property_metrics = defaultdict(lambda: defaultdict(list))
        for _ in tqdm(range(self.num_bootstrap_samples)):
            bootstrap_ids = random.choices(range(num_datapoints), k=num_datapoints)
            bootstrap_disambiguated_graphs = [disambiguated_graphs[x] for x in bootstrap_ids]
            bootstrap_gt_triples = [all_gt_triples[x] for x in bootstrap_ids]
            bootstrap_identifiers = [identifiers[x] for x in bootstrap_ids]
            bootstrap_candidate_entities = [candidate_entities[x] for x in bootstrap_ids]
            metrics, per_property_metrics = self.calculate_metrics(bootstrap_disambiguated_graphs, bootstrap_gt_triples, bootstrap_candidate_entities, bootstrap_identifiers)
            for key, value in metrics.items():
                all_metrics[key].append(value)
            for key, value in per_property_metrics.items():
                for subkey, subvalue in value.items():
                    all_per_property_metrics[key][subkey].append(subvalue)

        final_per_property_metrics = {}
        for key, value in all_per_property_metrics.items():
            final_per_property_metrics[key] = {}
            for subkey, subvalue in value.items():
                final_per_property_metrics[key][f"{subkey}/mean"] = np.mean(subvalue)
                final_per_property_metrics[key][f"{subkey}/std"] = np.std(subvalue)

        # Calculate mean and standard deviation
        final_metrics = {}
        for key, value in all_metrics.items():
            final_metrics[f"{key}/mean"] = np.mean(value)
            final_metrics[f"{key}/std"] = np.std(value)
        return final_metrics, final_per_property_metrics

    def benchmark(self, examples):
        if torch.cuda.is_available():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            disambiguated_graphs, candidate_entities = self.run(examples)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
        else:
            curr_time = time.time()
            disambiguated_graphs, candidate_entities = self.run(examples)
            curr_time = time.time() - curr_time

        all_gt_triples = [x["triples"] for x in examples]
        identifiers = [x["identifier"] for x in examples]
        metrics, per_property_metrics, detailed_results = self.calculate_metrics(disambiguated_graphs, all_gt_triples, candidate_entities, identifiers, return_details=True)
        bootstrap_calculate_metrics, bootstrap_per_property_metrics = self.bootstrap_calculate_metrics(disambiguated_graphs, all_gt_triples, candidate_entities, identifiers)
        metrics.update(bootstrap_calculate_metrics)
        for key, value in bootstrap_per_property_metrics.items():
            per_property_metrics[key].update(value)
        metrics["time"] = curr_time
        metrics["average_time"] = curr_time / len(examples)
        return metrics, per_property_metrics, detailed_results


    def run_with_thresholds(self, all_examples, property_thresholds, combined_thresholds, batch_size=8, graph_batch_size=64):
        input_for_disambiguation = []
        disambiguated_graphs: Dict[tuple, list] = defaultdict(list)
        candidate_entities: List[set] = []
        with torch.no_grad():
            for i in tqdm(range(0, len(all_examples), batch_size)):
                texts = [x["text"] for x in all_examples[i:i+batch_size]]
                if self.spoof_boundaries:
                    all_boundaries = [x["boundaries"] for x in all_examples[i:i+batch_size]]
                    mention_tokens = [[(None, None, 1.0) for y in x["boundaries"]] for x in all_examples[i:i+batch_size]]
                else:
                    tokenized = self.mention_recognizer.tokenizer(texts, return_tensors="pt",
                                                                  return_offsets_mapping=True, padding=True,
                                                                  truncation=True)
                    input_ids = tokenized["input_ids"].to(self.device)
                    attention_mask = tokenized["attention_mask"].to(self.device)
                    mention_matrices = self.mention_recognizer(input_ids, attention_mask)
                    # Calculate combination
                    attention_matrices = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)
                    mention_tokens = self.get_mention_tokens(mention_matrices, attention_matrices)
                    all_boundaries = self.get_boundaries(texts, mention_tokens, tokenized)

                bi_encoder_input = self.prepare_bi_encoder_input(texts, all_boundaries)
                candidates = self.get_candidates(bi_encoder_input)
                for example_idx in range(len(texts)):
                    sub_candidate_entities = set()
                    if example_idx in candidates:
                        for value in candidates[example_idx].values():
                            sub_candidate_entities.update({entity[0] for entity in value})
                    candidate_entities.append(sub_candidate_entities)

                scores, all_pairs = self.get_cross_encoder_scores(candidates, all_boundaries, texts)
                assert len(scores) == len(all_pairs) == len(mention_tokens) == len(texts), f"{len(scores)}, {len(all_pairs)}, {len(mention_tokens)}, {len(texts)}"
                input_for_disambiguation.append((scores, all_pairs, mention_tokens))

                if len(input_for_disambiguation) >= graph_batch_size:
                    for property_threshold in property_thresholds:
                        for combined_threshold in combined_thresholds:
                            for scores, all_pairs, mention_tokens in input_for_disambiguation:
                                disambiguated_graphs[(property_threshold, combined_threshold)].extend(self.disambiguate(scores, all_pairs, mention_tokens, property_threshold, combined_threshold))
                    input_for_disambiguation = []

        if input_for_disambiguation:
            for property_threshold in property_thresholds:
                for combined_threshold in combined_thresholds:
                    for scores, all_pairs, mention_tokens in input_for_disambiguation:
                        disambiguated_graphs[(property_threshold, combined_threshold)].extend(
                            self.disambiguate(scores, all_pairs, mention_tokens, property_threshold,
                                              combined_threshold))

        return disambiguated_graphs, candidate_entities


    def run(self, all_examples, batch_size=4, graph_batch_size=64):
        input_for_disambiguation = []
        disambiguated_graphs = []
        candidate_entities: List[set] = []
        with torch.no_grad():
            for i in tqdm(range(0, len(all_examples), batch_size)):
                texts = [x["text"] for x in all_examples[i:i+batch_size]]
                if self.spoof_boundaries:
                    all_boundaries = [x["boundaries"] for x in all_examples[i:i+batch_size]]
                    mention_tokens = [[(None, None, 1.0) for y in x["boundaries"]] for x in
                                      all_examples[i:i + batch_size]]
                else:
                    tokenized = self.mention_recognizer.tokenizer(texts, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
                    input_ids = tokenized["input_ids"].to(self.device)
                    attention_mask = tokenized["attention_mask"].to(self.device)
                    mention_matrices = self.mention_recognizer(input_ids, attention_mask)
                    # Calculate combination
                    attention_matrices = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)
                    mention_tokens = self.get_mention_tokens(mention_matrices, attention_matrices)
                    all_boundaries = self.get_boundaries(texts, mention_tokens, tokenized)
                bi_encoder_input = self.prepare_bi_encoder_input(texts, all_boundaries)
                candidates = self.get_candidates(bi_encoder_input)
                if self.spoof_candidates and self.spoof_boundaries:
                    all_candidates = [x["qids"] for x in all_examples[i:i+batch_size]]
                    for example_idx in range(len(texts)):
                        if example_idx in candidates:
                            for key in candidates[example_idx]:
                                if self.spoof_linking:
                                    candidates[example_idx][key] = [(all_candidates[example_idx][key], 1.0)]
                                else:
                                    if not any([x[0] == all_candidates[example_idx][key] for x in candidates[example_idx][key]]):
                                        candidates[example_idx][key].append((all_candidates[example_idx][key], 1.0))

                scores, all_pairs = self.get_cross_encoder_scores(candidates, all_boundaries, texts)
                for example_idx in range(len(texts)):
                    sub_candidate_entities = set()
                    if example_idx in candidates:
                        for value in candidates[example_idx].values():
                            sub_candidate_entities.update({entity[0] for entity in value})
                    candidate_entities.append(sub_candidate_entities)
                assert len(scores) == len(all_pairs) == len(mention_tokens) == len(texts), f"{len(scores)}, {len(all_pairs)}, {len(mention_tokens)}, {len(texts)}"
                input_for_disambiguation.append((scores, all_pairs, mention_tokens))

                if len(input_for_disambiguation) >= graph_batch_size:
                    for scores, all_pairs, mention_tokens in input_for_disambiguation:
                        disambiguated_graphs.extend(self.disambiguate(scores, all_pairs, mention_tokens))
                    input_for_disambiguation = []

        if input_for_disambiguation:
            for scores, all_pairs, mention_tokens in tqdm(input_for_disambiguation):
                disambiguated_graphs.extend(self.disambiguate(scores, all_pairs, mention_tokens))

        return disambiguated_graphs, candidate_entities

    @staticmethod
    def get_bin_edges(_max, base=2):
        bin_edges = [0]

        for power in range(math.ceil(np.log(_max) / np.log(base)) + 1):
            bin_edges.append(base ** power)

        if bin_edges[-1] == _max:
            power += 1
            bin_edges.append(2 ** power)

        return bin_edges

    def get_bucket_ids(self, rel_names):
        return [self.get_bucket_id_for_rel_name(rel_name) for rel_name in rel_names]

    def get_bucket_id_for_rel_name(self, rel_name):
        return self.get_bucket_id_for_occ_count(self.relation_counts[rel_name])

    def get_bucket_id_for_occ_count(self, value):
        assert value >= self.bin_edges[0]
        assert value < self.bin_edges[-1]

        for i in range(0, len(self.bin_edges) - 1):
            if value < self.bin_edges[i + 1]:
                return i

def load_dataset(path, debug=False, available_entities: set=None,
                 available_properties: set=None, add_true_boundaries: bool=False):
    examples = []
    for idx, item in enumerate(jsonlines.open(path)):
        identifier = item.get("id", idx)
        if  add_true_boundaries:
            try:
                boundaries, qids = get_boundaries(item)
            except ValueError:
                continue
        else:
            boundaries = None
            qids = None

        triples = []
        raw_triples = []
        if "non_formatted_wikidata_id_output" in item['meta_obj']:
            raw_triples = item['meta_obj']['non_formatted_wikidata_id_output']
        else:
            for x in item["output"]:
                raw_triples += x['non_formatted_wikidata_id_output']

        for s, p, o in raw_triples:
            if s.startswith("Q") and p.startswith("P") and o.startswith("Q"):
                if available_entities is None or (s in available_entities and o in available_entities):
                    if available_properties is None or p in available_properties:
                        triples.append((s, p, o))
        if triples:
            examples.append({
                "text": item["input"],
                "triples": triples,
                "identifier": identifier,
                "boundaries": boundaries,
                "qids": qids,
            })

    if debug:
        return random.sample(examples, 100)
    else:
        return examples


def evaluate_different_thresholds(discriminator: DiscriminativeCIE, examples):
    mention_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    property_thresholds = [0.2, 0.3 , 0.4, 0.5]
    combined_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.80, ]
    table = []
    for mention_threshold in mention_thresholds:
        discriminator.mention_threshold = mention_threshold

        results = discriminator.alt_benchmark(examples, property_thresholds, combined_thresholds)
        for result in results:
            print(f"mention_threshold: {mention_threshold}, property_threshold: {result['property_threshold']}",
                  f"combined_threshold: {result['combined_threshold']}")
            wandb.log({**result, "mention_threshold": mention_threshold})
            table.append((mention_threshold, result))
    all_headers = ["mention_threshold"]
    all_headers.extend(list(table[0][1].keys()))
    rearranged_table = []
    for row in table:
        all_results = [row[1][header] for header in all_headers[1:]]
        rearranged_table.append([row[0]] + all_results)
    wandb.log({"table": wandb.Table(data=rearranged_table, columns=all_headers)})


class Mode(enum.Enum):
    ET = "et"
    E = "e"

def main(dataset_path: str, include_property_scores: bool, include_mention_scores: bool, disambiguation_mode: DisambiguationMode,
         bi_encoder_path: str,
         mention_recognizer_path: str,
         crossencoder_path: str,
         relation_extractor_path: str,
         entity_restrictions: str,
         property_restrictions: str,
         debug: bool,
         mode: Mode,
         mention_threshold: float,
         property_threshold: float,
         combined_threshold: float,
         spoof_boundaries: bool,
         spoof_candidates: bool,
         spoof_linking: bool,
         alternative_relation_extractor: bool,
         alternative_relation_extractor_use_types: bool,
         alternative_relation_extractor_deactivate_text: bool,
         num_candidates: int):
    debug_seeds = [42, 43, 44, 45, 46]

    wandb.init(project="DisCIE")

    if relation_extractor_path is None:
        crossencoder_directory = Path(crossencoder_path).parent
    else:
        crossencoder_directory = Path(relation_extractor_path).parent
    types_index = None
    if Path(crossencoder_directory / "types_index.json").exists():
        types_index = json.load(open(f"{crossencoder_directory}/types_index.json"))

    alt_num_types = None
    if relation_extractor_path is not None:
        crossencoder_directory = Path(crossencoder_path).parent
        alt_num_types = len(json.load(open(f"{crossencoder_directory}/types_index.json")))

    discriminator = DiscriminativeCIE(bi_encoder_path,
                                      mention_recognizer_path,
                                      crossencoder_path,
                                      relation_extractor_path,
                                      types_index=types_index,
                                      include_property_scores=include_property_scores,
                                      include_mention_scores=include_mention_scores,
                                      alt_num_types=alt_num_types,
                                      disambiguation_mode=disambiguation_mode,
                                      entity_restrictions=entity_restrictions,
                                        property_restrictions=property_restrictions,
                                        mention_threshold=mention_threshold,
                                        property_threshold=property_threshold,
                                        combined_threshold=combined_threshold,
                                      spoof_boundaries=spoof_boundaries,
                                      spoof_candidates=spoof_candidates,
                                        spoof_linking=spoof_linking,
                                        alternative_relation_extractor=alternative_relation_extractor,
                                        alternative_relation_extractor_use_types=alternative_relation_extractor_use_types,
                                        alternative_relation_extractor_deactivate_text=alternative_relation_extractor_deactivate_text,
                                      num_candidates=num_candidates
                                      )
    if spoof_linking:
        assert spoof_candidates
    if  spoof_candidates:
        assert spoof_boundaries

    examples = load_dataset(dataset_path, debug=debug, available_entities=discriminator.entity_descriptions,
                            available_properties=discriminator.property_indices, add_true_boundaries=spoof_boundaries)
    if mode == Mode.ET:
        evaluate_different_thresholds(discriminator, examples)
    elif mode == Mode.E:
        results, per_property_metrics, detailed_results = discriminator.benchmark(examples)
        wandb.log({**results, "mention_threshold": mention_threshold, "property_threshold": property_threshold})
        property_metrics = sorted(list(per_property_metrics.values())[0].keys())
        sorted_property_names = sorted(per_property_metrics.keys())
        data = []
        for property_name in sorted_property_names:
            data.append([property_name] + [per_property_metrics[property_name][metric] for metric in property_metrics])
        per_property_table = wandb.Table(columns=["Bucket"] + list(property_metrics),
                    data=data)
        wandb.log({"table": per_property_table})
        # Create unique file name including also the filename
        file_name = f"results_{Path(dataset_path).stem}_{mention_threshold}_{property_threshold}_{combined_threshold}.json"
        per_property_stats_filename = f"per_property_stats_{Path(dataset_path).stem}_{mention_threshold}_{property_threshold}_{combined_threshold}.json"
        json.dump(per_property_metrics, open(per_property_stats_filename, "w"))
        json.dump(detailed_results, open(file_name, "w"))



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true", default=False)
    argparser.add_argument("--include_property_scores", action="store_true", default=False)
    argparser.add_argument("--spoof_boundaries", action="store_true", default=False)
    argparser.add_argument("--spoof_candidates", action="store_true", default=False)
    argparser.add_argument("--spoof_linking", action="store_true", default=False)
    argparser.add_argument("--include_mention_scores", action="store_true", default=False)
    argparser.add_argument("--alternative_relation_extractor", action="store_true", default=False)
    argparser.add_argument("--alternative_relation_extractor_use_types", action="store_true", default=False)
    argparser.add_argument("--alternative_relation_extractor_deactivate_text", action="store_true", default=False)
    argparser.add_argument("--disambiguation_mode", type=lambda x: DisambiguationMode[x], default=DisambiguationMode.SIMPLE,
                           choices=[elem for elem in DisambiguationMode])
    argparser.add_argument("--dataset_path", type=str, default="datasets/rebel/en_test.jsonl")
    argparser.add_argument("--bi_encoder_path", type=str, default="models/run_training_bi_encoder_new")
    argparser.add_argument("--mention_recognizer_path", type=str, default="models/mention_recognizer/model-epoch=06-val_f1=0.85_val_f1.ckpt")
    argparser.add_argument("--crossencoder_path", type=str, default="models/cross_encoder/model-epoch=13-val_triple_f1=0.85_triple_f1.ckpt")
    argparser.add_argument("--relation_extractor_path", type=str, default="models/relation_extractor/model-epoch=25-val_triple_f1=0.90_triple_f1.ckpt")
    argparser.add_argument("--entity_restrictions", type=str, default=None)
    argparser.add_argument("--property_restrictions", type=str, default=None)
    argparser.add_argument("--mention_threshold", type=float, default=0.5)
    argparser.add_argument("--property_threshold", type=float, default=0.5)
    argparser.add_argument("--combined_threshold", type=float, default=0.5)
    argparser.add_argument("--num_candidates", type=int, default=10)
    argparser.add_argument("--mode", type=lambda x: Mode[x], default=Mode.ET)
    args = argparser.parse_args()

    main(args.dataset_path, args.include_property_scores, args.include_mention_scores, args.disambiguation_mode,
         args.bi_encoder_path, args.mention_recognizer_path, args.crossencoder_path, args.relation_extractor_path,
            args.entity_restrictions, args.property_restrictions, args.debug, args.mode,
         args.mention_threshold, args.property_threshold, args.combined_threshold, args.spoof_boundaries,
         args.spoof_candidates, args.spoof_linking,
         args.alternative_relation_extractor, args.alternative_relation_extractor_use_types,
         args.alternative_relation_extractor_deactivate_text,
         args.num_candidates)



