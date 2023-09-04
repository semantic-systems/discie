import argparse
import json
import math
import os
import pickle
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Iterable, Tuple, Type, Dict, Callable, Optional, Union

import jsonlines
import numpy as np
import pytorch_lightning
import torch
import wandb
import faiss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sentence_transformers import SentenceTransformer, losses, InputExample, util, models
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.util import batch_to_device, fullname
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from src.candidate_generation.pl_candidate_generator import PLModule, create_dataloader, PLCrossEncoder, \
    create_dataloader_ce
from src.candidate_generation.special_tokens import CXS_TOKEN, CXE_TOKEN, TXS_TOKEN
from src.cie_utils.utils import map_page_ids_to_wikidata_ids



def load_entity_descriptions(entity_descriptions_file_path: str = "data/entity_wikidata_mapped.jsonl"):
    entity_descriptions_dict = {}
    for elem in tqdm(jsonlines.open(entity_descriptions_file_path)):
        qid = elem["qid"]
        text = elem["text"]
        entity_descriptions_dict[qid] = text
    return entity_descriptions_dict


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_examples: List[InputExample], raw_data: List = None):
        self.input_examples = input_examples
        self.raw_data = raw_data

    def __getitem__(self, item):
        return self.input_examples[item]

    def __len__(self):
        return len(self.input_examples)


def load_dataset_alt(dataset_file_path: str, tokenizer, length=32, stride=16):
    examples = []
    counter = 0
    for elem in jsonlines.open(dataset_file_path):
        try:
            boundaries, qids = get_boundaries(elem)
        except:
            continue
        text = elem["input"]
        tokenized_text = tokenizer(text, return_offsets_mapping=True,return_tensors="pt")
        if len(tokenized_text["input_ids"][0]) > length - 2:
            offset_mapping = tokenized_text["offset_mapping"][0]
            tokenized_text = tokenized_text["input_ids"][0]
            tokenized_text = tokenized_text[1:-1]
            offset_mapping = offset_mapping[1:-1]
            num_qids = 0
            for ins_num in range(math.ceil(len(tokenized_text) / stride)):
                begin = ins_num * stride
                end = min(ins_num * stride + length - 2, len(tokenized_text) - 1)
                char_begin = offset_mapping[begin][0]
                char_end = offset_mapping[end][1]
                instance_ids = tokenized_text[begin:end]
                new_ground_truth_qids = []
                for (start_pos, end_pos), qid in zip(boundaries, qids):
                    if start_pos >= char_begin and end_pos <= char_end:
                        new_ground_truth_qids.append(qid)
                num_qids += len(new_ground_truth_qids)
                if new_ground_truth_qids:
                    new_text = tokenizer.decode(instance_ids)
                    examples.append((new_text, new_ground_truth_qids))
            if  num_qids < len(qids):
                counter += 1
        else:

            ground_truth_qids_ = set()
            triples = elem["meta_obj"]["non_formatted_wikidata_id_output"]
            for triple in triples:
                ground_truth_qids_.add(triple[0])
                ground_truth_qids_.add(triple[2])
            examples.append((text, list(ground_truth_qids_)))
    print(counter)
    return examples


def load_dataset(dataset_file_path: str, full_dataset: bool = False):
    examples = []
    for elem in jsonlines.open(dataset_file_path):
        if full_dataset:
            examples.append(elem)
        else:
            qid = elem["qid"]
            text = elem["context"]
            examples.append((text, qid))
    return examples


def load_examples(dataset_file_path: str,
                  entity_descriptions_file_path: str = "data/entity_wikidata_mapped.jsonl") -> Tuple[List[InputExample], dict]:
    input_examples = []
    entity_descriptions_dict = load_entity_descriptions(entity_descriptions_file_path)

    for elem in jsonlines.open(dataset_file_path):
        qid = elem["qid"]
        text = elem["context"]
        label = 1
        if qid in entity_descriptions_dict:
            input_examples.append(InputExample(texts=[text, entity_descriptions_dict[qid]], label=label))
    return input_examples, entity_descriptions_dict


class CustomExample:
    def __init__(self, text_representations, text, ground_truth_qids=None, triples=None,
                 mention_types=None):
        self.text = text
        self.ground_truth_qids = ground_truth_qids
        self.text_representations = text_representations
        self.triples = triples
        self.mention_types = mention_types


def create_example(unique_boundaries, unique_ground_truth_qids, text, num_candidates, candidate_dict, identifier,
                entity_descriptions_dict, tokenizer,
                final_triples, types_index):
    text_representations: List[list] = []
    mention_types = []
    for (start, end), qid in zip(unique_boundaries, unique_ground_truth_qids):

            if candidate_dict is not None and identifier in candidate_dict and str((start, end)) in candidate_dict[identifier]:
                other_candidates = candidate_dict[identifier][str((start, end))]["candidates"][:num_candidates]
            else:
                other_candidates = []
            context_representation = get_mention_input(text, start, end)
            if qid in entity_descriptions_dict:
                entity_representation = entity_descriptions_dict[qid]
            else:
                entity_representation = ""
            concatenated_representation = context_representation + " {} ".format(
                tokenizer.sep_token) + entity_representation
            all_representations = []
            all_representations.append(concatenated_representation)
            if types_index is not None:
                if qid in types_index:
                    mention_types.append(types_index[qid])
                else:
                    mention_types.append([])
            for other_qid in other_candidates:
                if other_qid != qid and other_qid in entity_descriptions_dict:
                    other_entity_representation = entity_descriptions_dict[other_qid]
                    all_representations.append(
                        context_representation + " {} ".format(tokenizer.sep_token) + other_entity_representation)
                    mention_types.append([])
            text_representations.append(all_representations)
    if len(text_representations) > 1:
        return CustomExample(text_representations, text, unique_ground_truth_qids, final_triples, mention_types)
    else:
        return None

def load_examples_cross_encoder(dataset_file_path: str,
                  entity_descriptions_dict, tokenizer, property_indices, candidate_dict: dict,
                                num_candidates=10, max_num_entities: int = 8, stride=4,
                                types_dictionary = None) -> List[CustomExample]:
    examples = []
    num_gqids = 0
    counter = 0
    max_num_qids = 0
    multiple_relations_for_pair = 0
    for idx, elem in enumerate(jsonlines.open(dataset_file_path)):
        try:
            unique_boundaries, unique_ground_truth_qids, final_triples = get_boundaries_crossencoder(elem, property_indices)
        except:
            continue
        if len(final_triples) == 0:
            continue
        num_triples_per_pair = defaultdict(int)
        for triple in final_triples:
            num_triples_per_pair[(triple[0], triple[2])] += 1
        if any([x > 1 for x in num_triples_per_pair.values()]):
            multiple_relations_for_pair += 1
        text = elem["input"]
        identifier = str(elem["id"])
        num_gqids += len(unique_ground_truth_qids)
        counter += 1
        max_num_qids = max(max_num_qids, len(unique_ground_truth_qids))
        if len(unique_ground_truth_qids) > max_num_entities:
            # Split into multiple examples each containing maximum max_num_entities entities and shifted by stride
            for ins_num in range(math.ceil(len(unique_ground_truth_qids) / stride)):
                begin = ins_num * stride
                end = min(ins_num * stride + max_num_entities, len(unique_ground_truth_qids))
                new_ground_truth_qids = unique_ground_truth_qids[begin:end]
                new_unique_boundaries = unique_boundaries[begin:end]
                new_triples = []
                for triple in final_triples:
                    if begin <= triple[0] < end and begin <= triple[1] < end:
                        new_triple = (triple[0] - begin, triple[1] - begin, triple[2])
                        new_triples.append(new_triple)

                new_example = create_example(new_unique_boundaries, new_ground_truth_qids, text, num_candidates, candidate_dict, identifier,
                    entity_descriptions_dict, tokenizer,
                    new_triples, types_index=types_dictionary)
                if new_example is not None:
                    examples.append(new_example)

        else:
            new_example = create_example(unique_boundaries, unique_ground_truth_qids, text, num_candidates, candidate_dict, identifier,
                entity_descriptions_dict, tokenizer,
                final_triples, types_index=types_dictionary)
            if new_example is not None:
                examples.append(new_example)
    print(num_gqids / counter)
    print(max_num_qids)
    print(multiple_relations_for_pair/counter)
    return examples

def load_examples_alt(dataset_file_path: str,
                  entity_descriptions_dict, tokenizer) -> List[InputExample]:
    texts = []
    ground_truth_qids = []
    raw_examples = []
    for idx, elem in enumerate(load_dataset_alt(dataset_file_path, tokenizer)):
        texts.append(elem[0])
        ground_truth_qids.append(elem[1])
        raw_examples.append((elem[0], list(elem[1])))

    max_len = max([len(x) for x in ground_truth_qids])
    input_examples = []
    for text, ground_truth_qids_ in zip(texts, ground_truth_qids):
        qid_descriptions = [entity_descriptions_dict[qid] for qid in ground_truth_qids_ if qid in entity_descriptions_dict]
        padded_qid_descriptions = qid_descriptions + [""] * (max_len - len(qid_descriptions))
        labels = [1] * len(qid_descriptions) + [-1] * (max_len - len(qid_descriptions))
        input_examples.append(InputExample(texts=[text] + padded_qid_descriptions, label=labels))

    return CustomDataset(input_examples, raw_examples)


def create_mapped_entity_descriptions_file(entity_descriptions_file_path: str ="/graphbasejointelre/data/entity.jsonl"):
    all_page_ids = set()
    examples = []
    for elem in tqdm(jsonlines.open(entity_descriptions_file_path)):
        page_id = elem["idx"][elem["idx"].rfind("=") + 1:]
        all_page_ids.add(page_id)
        examples.append({
            "id": page_id,
            "title": elem["title"],
            "text": elem["text"]
        })
    wikipedia_id2wikidata_id = json.load(open( "/graphbasejointelre/data/wikipedia_wikidata_mapping.json"))
    not_found = all_page_ids.difference(set(wikipedia_id2wikidata_id.keys()))

    all_available_wikidata_ids = set()
    not_found_wikipedia_ids = set()
    for wikipedia_id in not_found:
        if str(wikipedia_id) in wikipedia_id2wikidata_id:
            all_available_wikidata_ids.add(wikipedia_id2wikidata_id[str(wikipedia_id)])
        else:
            not_found_wikipedia_ids.add(str(wikipedia_id))

    mapping = map_page_ids_to_wikidata_ids(not_found_wikipedia_ids)
    wikipedia_id2wikidata_id.update(mapping)

    filtered_examples = []
    for example in examples:
        if example["id"] in wikipedia_id2wikidata_id:
            example["qid"] = wikipedia_id2wikidata_id[example["id"]]
            example["text"] = f"{example['title']} {CXS_TOKEN} {example['text'][:200]}"
            filtered_examples.append(example)
    output_file_path = "entity_mapped.jsonl"
    with jsonlines.open(output_file_path, "w") as writer:
        writer.write_all(filtered_examples)


def get_mention_input(text: str, start: int, end: int, limit:int = 200):
    mention = text[start:end]
    right_context = text[end:]
    right_context = right_context[:limit]
    left_context = text[:start]
    left_context = left_context[-limit:]
    context_representation = f"{mention} {CXE_TOKEN} {left_context} {TXS_TOKEN} {right_context}"
    return context_representation


def get_entity_input():
    pass


def get_boundaries_crossencoder(example, property_indices):
    boundaries = []
    ground_truth_qids = []
    mentions = []
    triples = []
    counter = 0
    if "boundaries" in example:
        example_boundaries = example["boundaries"]
        qids = example.get("qids", [])
        for subj, pred, obj in example['meta_obj']['non_formatted_wikidata_id_output']:
            if subj.startswith("Q") and obj.startswith("Q") and pred in property_indices:
                subj_idx = qids.index(subj)
                if subj_idx < 0:
                    continue
                obj_idx = qids.index(obj)
                if obj_idx < 0:
                    continue
                boundaries.append(example_boundaries[subj_idx])
                boundaries.append(example_boundaries[obj_idx])
                ground_truth_qids.append(subj)
                ground_truth_qids.append(obj)
                mentions.append(example["input"][example_boundaries[subj_idx][0]:example_boundaries[subj_idx][1]])
                mentions.append(example["input"][example_boundaries[obj_idx][0]:example_boundaries[obj_idx][1]])
                triples.append((2 * counter, 2 * counter + 1, property_indices[pred]))
                counter += 1


    elif "triple_entity_boundaries" in example["meta_obj"]:
        for triple, ((subj_start, subj_end), (obj_start, obj_end)), (sub_mention, _, obj_mention) in zip(
                example['meta_obj']['non_formatted_wikidata_id_output'],
                example['meta_obj']['triple_entity_boundaries'],
                example['meta_obj']['substring_triples']):
            if triple[0].startswith("Q") and triple[2].startswith("Q") and triple[1] in property_indices:
                boundaries.append((subj_start, subj_end))
                ground_truth_qids.append(triple[0])
                mentions.append(sub_mention)
                boundaries.append((obj_start, obj_end))
                ground_truth_qids.append(triple[2])
                mentions.append(obj_mention)
                triples.append((2 * counter, 2 * counter + 1, property_indices[triple[1]]))
                counter += 1
    else:
        raise ValueError("No boundaries found")

    if not all([example["input"][boundary[0]:boundary[1]] == mention for boundary, mention in zip(boundaries, mentions)]):
        # Normalize boundaries to be within the text, currently the refer to the full text
        # To do, find the position of the input in the full text and subtract it from the boundaries
        full_text = example["meta_obj"]["full_text"]
        input_start = full_text.find(example["input"])
        if input_start == -1:
            input_start = full_text.find(example["input"][:-2])
        orig_boundaries = boundaries
        boundaries = [(start - input_start, end - input_start) for start, end in boundaries]
        if not all([example["input"][boundary[0]:boundary[1]] == mention for boundary, mention in zip(boundaries, mentions)]):
            raise ValueError(f"Boundaries do not match mentions: {orig_boundaries} {mentions} {example['input']} {full_text}")

    triples_replaced_with_boundaries = []
    for triple in triples:
        triples_replaced_with_boundaries.append((boundaries[triple[0]], boundaries[triple[1]], triple[2]))



    # Sort boundaries and ground_truth_qids by boundaries, smallest first
    boundaries, ground_truth_qids = zip(*sorted(zip(boundaries, ground_truth_qids), key=lambda x: x[0][0]))
    # Only allow unique boundaries, if there are multiple entities with the same boundary, only the first one is kept
    # This means that the ground_truth_qids are also shortened
    unique_boundaries = []
    unique_ground_truth_qids = []
    final_triples = []
    for boundary, qid in zip(boundaries, ground_truth_qids):
        if boundary not in unique_boundaries:
            unique_boundaries.append(boundary)
            unique_ground_truth_qids.append(qid)
    for triple in triples_replaced_with_boundaries:
        if triple[0] in unique_boundaries and triple[1] in unique_boundaries:
            boundary_idx1 = unique_boundaries.index(triple[0])
            boundary_idx2 = unique_boundaries.index(triple[1])
            final_triples.append((boundary_idx1, boundary_idx2, triple[2]))

    return unique_boundaries, unique_ground_truth_qids, final_triples


def get_candidates(bi_encoder, faiss_index, entity_indices, bi_encoder_input: List[List[str]], num_candidates: int = 100):
    elongated_bi_encoder_input = []
    example_ids = []
    mention_ids = []
    for idx, context_representations in enumerate(bi_encoder_input):
        for ment_idx, context_representation in enumerate(context_representations):
            elongated_bi_encoder_input.append(context_representation)
            example_ids.append(idx)
            mention_ids.append(ment_idx)

    encoded_sentences = bi_encoder.encode(elongated_bi_encoder_input, batch_size=256)
    # if self.bi_encoder.normalize:
    encoded_sentences = encoded_sentences / np.linalg.norm(encoded_sentences, axis=1).reshape(-1, 1)

    # Run the following batch wise
    n = len(encoded_sentences)
    k = num_candidates
    scores = np.empty((n, k), dtype=np.float32)
    indices = np.empty((n, k), dtype=np.int64)

    chunk_size = 10000  # Adjust this value based on the size of your dataset and available memory

    for i in range(0, n, chunk_size):
        start = i
        end = min(i + chunk_size, n)
        chunk_scores, chunk_indices = faiss_index.search(encoded_sentences[start:end], k)

        scores[start:end] = chunk_scores
        indices[start:end] = chunk_indices

    candidate_set = defaultdict(dict)
    for document_id, mention_id, score, idx in zip(example_ids, mention_ids, scores, indices):
        # Make identifier consisting of document_id and span
        candidates = []
        for i, (index, score_) in enumerate(zip(idx, score)):
            candidates.append((entity_indices[str(index)], score_))
        candidate_set[document_id][mention_id] = candidates
    return candidate_set

def prepare_bi_encoder_input(texts,
                             all_boundaries) -> List[list]:
    bi_encoder_input = []

    for text, boundaries in zip(texts, all_boundaries):
        context_representations = []
        for start, end in boundaries:
            context_representations.append(get_mention_input(text, start, end))
        bi_encoder_input.append(context_representations)
    return bi_encoder_input


def get_boundaries(example, ignore_boundaries=False, candidate_generation_tuple: tuple=None):
    if not ignore_boundaries and "boundaries" in example:
        return example["boundaries"], example.get("qids", None)
    boundaries = []
    ground_truth_qids = []
    mentions = []
    if "triple_entity_boundaries" in example["meta_obj"]:
        for triple, ((subj_start, subj_end), (obj_start, obj_end)), (sub_mention, _, obj_mention) in zip(
                example['meta_obj']['non_formatted_wikidata_id_output'],
                example['meta_obj']['triple_entity_boundaries'],
                example['meta_obj']['substring_triples']):
            if triple[0].startswith("Q") and triple[2].startswith("Q"):
                boundaries.append((subj_start, subj_end))
                ground_truth_qids.append(triple[0])
                mentions.append(sub_mention)
            if triple[2].startswith("Q"):
                boundaries.append((obj_start, obj_end))
                ground_truth_qids.append(triple[2])
                mentions.append(obj_mention)
    elif "entity_tokens_mask" in example["meta_obj"]:
        entity_mask = example["meta_obj"]["entity_tokens_mask"]
        # split example["input"] into token spans by whitespace
        token_spans = []
        start = 0
        for i, c in enumerate(example["input"]):
            if c == " ":
                token_spans.append((start, i))
                start = i + 1
        token_spans.append((start, len(example["input"])))

        assert len(entity_mask) == len(token_spans)
        # Get entity boundaries given by entity_mask, which is a list of 0s and 1s, where 1 indicate an entity
        # token, 1s without a 0 in between are considered as one entity, the spans of all such entities are
        # combined
        entity_boundaries_by_length = []
        entity_active = False
        for i, (token_span, mask) in enumerate(zip(token_spans, entity_mask)):
            if mask == 1:
                if not entity_active:
                    entity_active = True
                    entity_boundaries_by_length.append([token_span])
                else:
                    entity_boundaries_by_length[-1].append(token_span)
            else:
                entity_active = False

        for triple in example['meta_obj']['non_formatted_wikidata_id_output']:
            if triple[0].startswith("Q"):
                ground_truth_qids.append(triple[0])
            if triple[2].startswith("Q"):
                ground_truth_qids.append(triple[2])
        # Get the longest entity boundary for each entity
        for entity_boundaries in entity_boundaries_by_length:
            boundaries.append((entity_boundaries[0][0], entity_boundaries[-1][1]))

        if candidate_generation_tuple is not None:
            ground_truth_qids_set = set(ground_truth_qids)
            bi_encoder, faiss_index, entity_indices = candidate_generation_tuple
            boundaries_for_candidate_generation = []
            for entity_boundaries in entity_boundaries_by_length:
                for idx, boundary in enumerate(entity_boundaries):
                    for boundary_ in entity_boundaries[idx:]:
                        boundaries_for_candidate_generation.append((boundary[0], boundary_[1]))
            prepared = prepare_bi_encoder_input([example["input"]], [boundaries_for_candidate_generation])
            candidates = get_candidates(bi_encoder,faiss_index, entity_indices, prepared)
            best_per_ground_truth = defaultdict(lambda: (None, -1.0))
            for boundary, candidates_of_boundary in zip(boundaries_for_candidate_generation, list(candidates.values())[0].values()):
                all_candidates_with_scores = {candidate: score for candidate, score in candidates_of_boundary}
                if len(candidates_of_boundary) > 0:
                    for ground_truth_qid in ground_truth_qids_set:
                        if ground_truth_qid in all_candidates_with_scores:
                            if all_candidates_with_scores[ground_truth_qid] > best_per_ground_truth[ground_truth_qid][1]:
                                best_per_ground_truth[ground_truth_qid] = (boundary, all_candidates_with_scores[ground_truth_qid])
            ground_truth_qids = []
            boundaries = []
            mentions = []
            for ground_truth_qid, (boundary, score) in best_per_ground_truth.items():
                if boundary is not None:
                    boundaries.append(boundary)
                    ground_truth_qids.append(ground_truth_qid)
                    mentions.append(example["input"][boundary[0]:boundary[1]])
    else:
        return [], []

    if len(boundaries) < len(ground_truth_qids):
        raise ValueError(f"Boundaries do not match ground_truth_qids: {ground_truth_qids}")

    if not all([example["input"][boundary[0]:boundary[1]] == mention for boundary, mention in zip(boundaries, mentions)]):
        # Normalize boundaries to be within the text, currently the refer to the full text
        # To do, find the position of the input in the full text and subtract it from the boundaries
        full_text = example["meta_obj"]["full_text"]
        input_start = full_text.find(example["input"])
        if input_start == -1:
            input_start = full_text.find(example["input"][:-2])
        orig_boundaries = boundaries
        boundaries = [(start - input_start, end - input_start) for start, end in boundaries]
        if not all([example["input"][boundary[0]:boundary[1]] == mention for boundary, mention in zip(boundaries, mentions)]):
            raise ValueError(f"Boundaries do not match mentions: {orig_boundaries} {mentions} {example['input']} {full_text}")



    # Sort boundaries and ground_truth_qids by boundaries, smallest first
    boundaries, ground_truth_qids = zip(*sorted(zip(boundaries, ground_truth_qids), key=lambda x: x[0][0]))
    # Only allow unique boundaries, if there are multiple entities with the same boundary, only the first one is kept
    # This means that the ground_truth_qids are also shortened
    unique_boundaries = []
    unique_ground_truth_qids = []
    for boundary, qid in zip(boundaries, ground_truth_qids):
        if boundary not in unique_boundaries:
            unique_boundaries.append(boundary)
            unique_ground_truth_qids.append(qid)

    return unique_boundaries, unique_ground_truth_qids


def map_dataset(dataset_file_path: str):
    dataset = [item for item in jsonlines.open(dataset_file_path)]
    mapped_dataset = []
    for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
        if "boundaries" in example:
            boundaries = example["boundaries"]
            ground_truth_qids = example["qids"]
        else:
            try:
                boundaries, ground_truth_qids = get_boundaries(example)
            except ValueError:
                continue

        for (start, end), qid in zip(boundaries, ground_truth_qids):
            context_representation = get_mention_input(example["input"], start, end)
            mapped_dataset.append({
                "id": example["id"],
                "qid": qid,
                "context": context_representation,
                "span": (start, end),
            })
    output_file_path = dataset_file_path.replace(".jsonl", "_mapped.jsonl")
    with jsonlines.open(output_file_path, "w") as writer:
        writer.write_all(mapped_dataset)


class PLDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return self.test_dataloader

class SentenceTransformerWithWandbInFit(SentenceTransformer):
    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0,
            evaluate_every_n_epochs: int = 1,
            epochs_of_training_before_regeneration: int = -1,
            entity_descriptions = None,
            normalize: bool = False,
            ):
        ##Add info to model card
        # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps(
            {"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch,
             "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),
             "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps,
             "max_grad_norm": max_grad_norm}, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}",
                                                                                                     info_loss_functions).replace(
            "{FIT_PARAMETERS}", info_fit_parameters)

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self._target_device)
                    features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()
                    wandb.log({"loss": loss_value.item()})

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
            if evaluate_every_n_epochs > 0 and (epoch + 1) % evaluate_every_n_epochs == 0 or epoch == epochs - 1:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)
            if epochs_of_training_before_regeneration > 0 and (epoch + 1) % epochs_of_training_before_regeneration == 0:
                old_dataloaders = dataloaders
                dataloaders = []
                for dataloader in old_dataloaders:
                    if epoch == evaluator.epoch_of_index:
                        new_dataset = create_dataset_with_hard_negatives(dataloader.dataset, entity_descriptions, self,
                                                                         normalize, *evaluator.current_index)
                    else:
                        new_dataset = create_dataset_with_hard_negatives(dataloader.dataset, entity_descriptions, self,
                                                                         normalize)
                    dataloader = DataLoader(new_dataset, shuffle=True, batch_size=dataloader.batch_size,
                                            collate_fn=dataloader.collate_fn)
                    dataloaders.append(dataloader)

                    

        if evaluator is None and output_path is not None:  # No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


class SentenceTransformerDual(SentenceTransformerWithWandbInFit):
    def __init__(self, model_name_or_path: Optional[str] = None,
                 modules: Optional[Iterable[nn.Module]] = None,
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 use_auth_token: Union[bool, str, None] = None):
        super().__init__(model_name_or_path, modules, device, cache_folder, use_auth_token)
        self.document_model = SentenceTransformer(model_name_or_path, modules, device, cache_folder, use_auth_token)


def create_index(model, entity_descriptions, normalize: bool):
    keys = list(entity_descriptions.keys())
    entity_indices = {i: key for i, key in enumerate(keys)}
    sentences = [entity_descriptions[key] for key in keys]
    embeddings = model.encode(sentences, batch_size=512, show_progress_bar=True)
    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dimension)
    nlist = int(math.sqrt(len(sentences)))
    nprobe = int(math.sqrt(nlist))  # Taking square root of nlist
    nprobe = min(nprobe, nlist)
    # nprobe = 100
    # nlist = 10000

    faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss_index.nprobe = nprobe
    # Add tqdm for index creation  with a batch size of 10000
    # for i in tqdm(range(0, embeddings.shape[0], 10000), desc="Index creation"):
    #    faiss_index.add(embeddings[i:i + 10000])
    faiss_index.train(embeddings)
    print("Index trained")
    faiss_index.add(embeddings)
    return faiss_index, entity_indices

def encode_sentences(sentences, model, normalize: bool, batch_size: int = 32):
    if hasattr(model, "document_model"):
        encoded_sentences = model.document_model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    else:
        encoded_sentences = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    if normalize:
        encoded_sentences = encoded_sentences / np.linalg.norm(encoded_sentences, axis=1).reshape(-1, 1)
    return encoded_sentences

class CustomEvaluator(SentenceEvaluator):
    def __init__(self, dataset, entity_descriptions, normalize: bool, batch_size=32,
                 ):
        self.dataset = dataset
        self.entity_descriptions = entity_descriptions
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_candidates = 100
        self.current_index = None
        self.epoch_of_index = None
        super().__init__()

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """
        faiss_index, entity_indices = create_index(model, self.entity_descriptions, self.normalize)
        self.current_index = (faiss_index, entity_indices)
        self.epoch_of_index = epoch

        sentences = [example[0] for example in self.dataset]
        qids = [example[1] for example in self.dataset]
        encoded_sentences = encode_sentences(sentences, model, self.normalize)
        scores, indices = faiss_index.search(encoded_sentences, self.num_candidates)

        accuracy = 0
        counter = 0
        for qid, idx in zip(qids, indices):
            if isinstance(qid, list):
                valid_qids = {entity_indices[i] for i in idx[:min(len(qid),len(idx))]}
                for idx_, q in enumerate(qid):
                    if q in valid_qids:
                        accuracy += 1
                    counter += 1
            else:
                if qid == entity_indices[idx[0]]:
                    accuracy += 1
                counter += 1
        # Calculate hit@k for k in [1:20]
        hit_at_10 = 0
        for k in range(1, self.num_candidates + 1):
            hits = 0
            counter = 0
            for qid, idx in zip(qids, indices):
                if isinstance(qid, list):
                    valid_qids = {entity_indices[i] for i in idx[:k] if i >= 0 }
                    for idx_, q in enumerate(qid):
                        if q in valid_qids:
                            hits += 1
                        counter += 1
                else:
                    if qid in {entity_indices[i] for i in idx[:k]}:
                        hits += 1
                    counter += 1
            if k == 10:
                hit_at_10 = hits / counter
            wandb.log({"hit@{}".format(k): hits / counter})
        wandb.log({"accuracy": accuracy / counter})
        return hit_at_10

def init_model(model_name: str = "sentence-transformers/all-MiniLM-L12-v2", separated: bool = False, manually: bool = False, model_type: str = "single"):
    if manually:
        # Define the underlying transformer model, e.g., 'roberta-base'
        transformer_model = models.Transformer(model_name)

        # Define a mean pooling layer
        pooling_layer = models.Pooling(transformer_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=False,
                                       pooling_mode_cls_token=True,
                                       pooling_mode_max_tokens=False)

        # Combine the transformer model with the pooling layer
        if model_type == "single":
            model = SentenceTransformerWithWandbInFit(modules=[transformer_model, pooling_layer])
        elif model_type == "dual":
            model = SentenceTransformerDual(modules=[transformer_model, pooling_layer])
        else:
            raise ValueError("model_type must be either single or dual")
    else:
        if model_type == "single":
            model = SentenceTransformerWithWandbInFit(model_name)
        elif model_type == "dual":
            model = SentenceTransformerDual(model_name)
        else:
            raise ValueError("model_type must be either single or dual")

    model.tokenizer.add_tokens([CXS_TOKEN, CXE_TOKEN, TXS_TOKEN])
    model_bert = model._first_module()
    model_bert.auto_model.resize_token_embeddings(len(model.tokenizer))

    if model_type == "dual":
        model_doc = model.document_model._first_module()
        model_doc.auto_model.resize_token_embeddings(len(model.tokenizer))

    if separated:
        return model_bert, model.tokenizer
    else:
        return model


def sum_log_nce_loss(logits, mask, reduction='sum'):
    """
        :param logits: reranking logits(B x C) or span loss(B x C x L)
        :param mask: reranking mask(B x C) or span mask(B x C x L)
        :return: sum log p_positive i  over (positive i, negatives)
    """
    gold_scores = logits.masked_fill(~(mask.bool()), 0)
    gold_scores_sum = gold_scores.sum(-1)  # B x C
    neg_logits = logits.masked_fill(mask.bool(), float('-inf'))  # B x C x L
    neg_log_sum_exp = torch.logsumexp(neg_logits, -1, keepdim=True)  # B x C x 1
    norm_term = torch.logaddexp(logits, neg_log_sum_exp).masked_fill(~(
        mask.bool()), 0).sum(-1)
    gold_log_probs = gold_scores_sum - norm_term
    loss = -gold_log_probs.sum()
    if reduction == 'mean':
        print('mean reduction')
        loss /= logits.size(0)
    return loss


class AltLoss(torch.nn.Module):
    def __init__(self, model, normalize, scale=20.0):
        super().__init__()
        self.normalize = normalize
        self.scale = scale
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        if hasattr(self.model, "document_model"):
            main = self.model.document_model(sentence_features[0])['sentence_embedding']
        else:
            main = self.model(sentence_features[0])['sentence_embedding']

        new_labels = torch.zeros((labels.size(0), torch.sum(labels >= 0)), dtype=torch.long,
                                 device=labels.device)
        offset = 0
        candidates = []
        concatenated_sentence_features = torch.stack(
            [self.model(x)['sentence_embedding'] for x in sentence_features[1:]], dim=1)
        for idx, labels_ in enumerate(labels):
            new_labels_ = [l for l in labels_ if l >= 0]
            new_labels[idx, offset:offset + len(new_labels_)] = torch.tensor(new_labels_)
            offset += len(new_labels_)
            candidates.append(concatenated_sentence_features[idx, labels_ >= 0, :])
        candidates = torch.cat(candidates, dim=0)

        if self.normalize:
            candidates = candidates / torch.norm(candidates, dim=1).unsqueeze(-1)
            main = main / torch.norm(main, dim=1).unsqueeze(-1)

        scores = torch.mm(main, candidates.transpose(0, 1))
        epsilon = 1e-7
        if self.normalize:
            scores *= self.scale

        loss = sum_log_nce_loss(scores, new_labels)
        return loss




def create_dataset_with_hard_negatives(training_examples: List[InputExample], entity_descriptions: dict, model: SentenceTransformer,
                                       normalize: bool, faiss_index = None, entity_indices = None):
    if faiss_index is None:
        faiss_index, entity_indices = create_index(model, entity_descriptions, normalize)
    else:
        assert entity_indices is not None

    texts = []
    ground_truths = []
    for example in training_examples:
        text = example.texts[0]
        labels = example.label
        ground_truth = [t for t, l in zip(example.texts[1:], labels) if l == 1]
        texts.append(text)
        ground_truths.append(ground_truth)
    encoded_sentences = encode_sentences(texts, model, normalize)

    batch_size = 1000  # Set the size of your batches

    n, d = encoded_sentences.shape

    scores = []
    indices = []

    # Split the encoded_sentences into batches
    for i in tqdm(range(0, n, batch_size)):
        batch = encoded_sentences[i: i + batch_size]
        score_batch, indices_batch = faiss_index.search(batch, 10)
        scores.append(score_batch)
        indices.append(indices_batch)

    # Concatenate all results
    scores = np.concatenate(scores, axis=0)
    indices = np.concatenate(indices, axis=0)

    new_texts = []
    new_labels = []
    for example, ground_truth, scores_, indices_ in tqdm(zip(training_examples, ground_truths, scores, indices)):
        new_labels_ = [1] * len(ground_truth)
        hard_negatives = []
        for score, index in zip(scores_, indices_):
            if entity_descriptions[entity_indices[index]] not in ground_truth:
                new_labels_.append(0)
                hard_negatives.append(entity_descriptions[entity_indices[index]])
        new_labels.append(new_labels_)
        new_texts.append([example.texts[0]] + ground_truth + hard_negatives)
    new_training_examples = []
    maximum_texts = max([len(x) for x in new_texts])
    maximum_labels = max([len(x) for x in new_labels])
    for text, labels in zip(new_texts, new_labels):
        new_training_examples.append(InputExample(texts=text + [""] * (maximum_texts - len(text)),
                                                  label=labels + [-1] * (maximum_labels - len(labels))))
    return new_training_examples


def load_property_indices(filename: str ="data/rel_id2surface_form.jsonl"):
    properties = []
    for elem in jsonlines.open(filename):
        properties.append(elem["wikidata_id"])

    sorted_properties = sorted(properties)
    property_indices = {p: i for i, p in enumerate(sorted_properties)}
    return property_indices


def load_candidates(candidate_set_path: str):
    return json.load(open(candidate_set_path))


def get_type_dictionary(filter_set=None, types_index=None,
                        type_dictionary_file: str = "data/item_types_relation_extraction_alt.jsonl"):
    types_dictionary = {}
    types_to_include = set()
    counter = 0
    #if Path("data/types_dictionary.pkl").exists():
    #    types_dictionary = pickle.load(open("data/types_dictionary.pkl", "rb"))
    #else:
    for item in tqdm(jsonlines.open(type_dictionary_file)):
        if filter_set is not None and item["item"] not in filter_set:
            continue
        if types_index is not None:
            types_dictionary[item["item"]] = [types_index[t] for t in item["types"] if t in types_index]
        else:
            types_dictionary[item["item"]] = set(item["types"])
        types_to_include.update(item["types"])
        counter += 1
        #if counter > 100000:
        #    break
        # Dump to pickle
        #pickle.dump(types_dictionary, open("data/types_dictionary.pkl", "wb"))
    types_to_include = sorted(list(types_to_include))
    if types_index is None:
        types_index = {t: i for i, t in enumerate(types_to_include)}
        types_dictionary = {k: [types_index[t] for t in v if t in types_index] for k, v in types_dictionary.items()}

    return types_dictionary, types_index


def get_all_eligible_qids(dataset_path: str, property_indices):
    all_qids = set()
    with jsonlines.open(dataset_path) as reader:
        for example in tqdm(reader):
            try:
                boundaries, qids, triples = get_boundaries_crossencoder(example, property_indices)
                all_qids.update(qids)
            except ValueError:
                continue

    return all_qids


def train_crossencoder(training_dataset: str,
          eval_dataset: str,
          training_canddiate_set_path: str,
            eval_candidate_set_path: str,
          output_path: str = "cross_encoder",
          epochs_of_training_before_regeneration: int = None,
          model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
          batch_size: int = 128,
          epochs: int = 30,
          checkpoint_path: str = None,
                       include_types=True,
                       candidate_weight=1.0,
                       num_candidates=10,
                       types_index_path: str = None,
                       type_dictionary_file: str = "data/item_types_relation_extraction_alt.jsonl"):
    if Path(output_path).exists():
        output_path = output_path + "_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_path, exist_ok=True)
    property_indices = load_property_indices()
    entity_descriptions = load_entity_descriptions()
    all_qids = set()
    all_qids.update(get_all_eligible_qids(training_dataset, property_indices))
    all_qids.update(get_all_eligible_qids(eval_dataset, property_indices))
    types_dictionary = None
    types_index = None
    if include_types:
        if types_index_path is not None:
            types_index = json.load(open(types_index_path))
        types_dictionary, types_index = get_type_dictionary(all_qids, types_index, type_dictionary_file)
        json.dump(types_index, open(os.path.join(output_path, "types_index.json"), "w"))
    if checkpoint_path is not None:
        module = PLCrossEncoder.load_from_checkpoint(checkpoint_path, model_name=model_name, num_properties=len(property_indices), entity_descriptions=entity_descriptions,
                        epochs_of_training_before_regeneration=epochs_of_training_before_regeneration,
                        batch_size=batch_size, number_of_types=len(types_index) if types_index else None,
                                candidate_weight=candidate_weight)
    else:
        module = PLCrossEncoder(model_name, num_properties=len(property_indices), entity_descriptions=entity_descriptions,
                        epochs_of_training_before_regeneration=epochs_of_training_before_regeneration,
                        batch_size=batch_size, number_of_types=len(types_index) if types_index else None,
                                candidate_weight=candidate_weight)

    if num_candidates > 0:
        dev_candidate_dict = load_candidates(eval_candidate_set_path)
        train_candidate_dict = load_candidates(training_canddiate_set_path)
    else:
        dev_candidate_dict = None
        train_candidate_dict = None
    dev_sentences = load_examples_cross_encoder(eval_dataset, entity_descriptions, module.tokenizer, property_indices,
                                                dev_candidate_dict, types_dictionary=types_dictionary, num_candidates=num_candidates)

    training_examples = load_examples_cross_encoder(training_dataset, entity_descriptions, module.tokenizer,
                                                    property_indices, train_candidate_dict, types_dictionary=types_dictionary,
                                                    num_candidates=num_candidates)

    module.total_steps = (len(training_examples) // batch_size) * epochs
    train_dataloader = create_dataloader_ce(training_examples, shuffle=True, batch_size=batch_size,
                                         tokenizer=module.tokenizer, num_workers=4)
    dev_dataloader = create_dataloader_ce(dev_sentences, shuffle=True, batch_size=batch_size,
                                       tokenizer=module.tokenizer, num_workers=4)
    wandb_logger = WandbLogger(project="crossencoder", name=output_path)
    checkpoint_callback_1 = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_path,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        save_last=True,
        mode='min',
    )

    checkpoint_callback_2 = ModelCheckpoint(
        monitor='val_triple_f1',
        dirpath=output_path,
        filename='model-{epoch:02d}-{val_triple_f1:.2f}_triple_f1',
        save_top_k=1,
        save_last=True,
        mode='max',
    )
    checkpoint_callback_3 = ModelCheckpoint(
        monitor='val_candidate_accuracy',
        dirpath=output_path,
        filename='model-{epoch:02d}-{val_candidate_accuracy:.2f}_val_candidate_accuracy',
        save_top_k=1,
        save_last=True,
        mode='max',
    )

    num_gpus = 0
    accelerator = None
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        accelerator = "ddp" if num_gpus > 1 else None
    trainer_params = {"gpus": num_gpus, "max_epochs": epochs, "logger": wandb_logger,
                      "callbacks": [checkpoint_callback_1, checkpoint_callback_2,
                                    checkpoint_callback_3], "accelerator": accelerator}
    trainer = Trainer(**trainer_params)
    #trainer.validate(model=module, val_dataloaders=dev_dataloader)

    trainer.fit(module, train_dataloader, dev_dataloader)

def train(training_dataset: str =  "data/rebel/en_train_mapped.jsonl",
          eval_dataset: str = "data/rebel/en_val_mapped.jsonl",
          normalize: bool = True,
          output_path: str = "runs_training_bi_encoder",
          epochs_of_training_before_regeneration: int = None,
          model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
          batch_size: int = 256,
          manually: bool = False,
          model_type: str = "single",
          evaluate_every_n_epochs: int = 2,
          epochs: int = 30,
          checkpoint_path: str = None,):

    epochs = 10
    if epochs_of_training_before_regeneration is not None:
        # if torch.cuda.device_count() > 1:
        #     model._modules["0"].auto_model.encoder = DataParallel(model._modules["0"].auto_model.encoder)

        entity_descriptions = load_entity_descriptions()
        module = PLModule(model_name, entity_descriptions=entity_descriptions, normalize=normalize,
                          epochs_of_training_before_regeneration=epochs_of_training_before_regeneration,
                          batch_size=batch_size)
        dev_sentences = load_examples_alt(eval_dataset, entity_descriptions, module.tokenizer)
        training_examples = load_examples_alt(training_dataset, entity_descriptions, module.tokenizer)

        module.total_steps = (len(training_examples) // batch_size ) * epochs
        train_dataloader = create_dataloader(training_examples, shuffle=True, batch_size=batch_size,
                                             tokenizer=module.tokenizer, num_workers=4)
        dev_dataloader = create_dataloader(dev_sentences, shuffle=False, batch_size=batch_size,
                                           tokenizer=module.tokenizer, num_workers=4)
        wandb_logger = WandbLogger(project="sentence-transformers", name=output_path)
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='candidate_generator_checkpoints',
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=2,
            save_last=True,
            mode='min',
        )

        num_gpus = 0
        accelerator = None
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            accelerator = "ddp" if num_gpus > 1 else None
        trainer_params = {"gpus": num_gpus, "max_epochs": 1, "logger": wandb_logger,
                          "callbacks": [checkpoint_callback], "accelerator": accelerator}
        trainer = Trainer(**trainer_params)

        for epoch in range(epochs):
            if epoch > 0:
                # Resume from the last model checkpoint if not the first epoch
                checkpoint_path = checkpoint_callback.last_model_path
                trainer = Trainer(**trainer_params, resume_from_checkpoint=checkpoint_path)


            trainer.fit(module, train_dataloader)

            faiss_index = None
            entity_indices = None
            if not torch.cuda.is_available():
                module = module.to("cpu")
            else:
                module = module.to("cuda")
            if (epoch + 1) % evaluate_every_n_epochs == 0:
                module.eval()
                module.freeze()
                if faiss_index is None:
                    faiss_index, entity_indices = module.create_index()
                trainer.validate(module, dev_dataloader)
                module.train()
                module.unfreeze()

            if (epoch + 1) % epochs_of_training_before_regeneration == 0:
                if faiss_index is None:
                    with torch.no_grad():
                        module.eval()
                        module.freeze()
                        faiss_index, entity_indices = module.create_index()
                        module.train()
                        module.unfreeze()
                train_dataloader = create_dataloader(module.update_dataloader_with_hard_negatives(training_examples, faiss_index,
                                                                                                entity_indices),
                                                     shuffle=True, batch_size=batch_size,
                                                     tokenizer=module.tokenizer, num_workers=4
                                                     )
                dev_dataloader = create_dataloader(module.update_dataloader_with_hard_negatives(dev_sentences, faiss_index,
                                                                                                entity_indices),
                                                    shuffle=False, batch_size=batch_size,
                                                    tokenizer=module.tokenizer, num_workers=4
                                                    )
            module.to("cpu")

    else:
        wandb.init(project="sentence-transformers", name=output_path)
        model = init_model(model_name, manually=manually, model_type=model_type)
        if normalize:
            loss_function = losses.MultipleNegativesSymmetricRankingLoss(model=model)
        else:
            loss_function = losses.MultipleNegativesSymmetricRankingLoss(model=model,
                                                                         scale=1, similarity_fct=util.dot_score)
        dev_sentences = load_dataset(eval_dataset)
        training_examples, entity_descriptions = load_examples(training_dataset)
        evaluator = CustomEvaluator(dev_sentences, entity_descriptions, normalize=normalize)
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
        model.fit(train_objectives=[(train_dataloader,loss_function)], epochs=epochs, evaluator=evaluator,
                  save_best_model=True,
                  checkpoint_save_total_limit=2, checkpoint_path=output_path,
                  checkpoint_save_steps=len(train_dataloader),
                  output_path=f"output/{output_path}",
                  evaluate_every_n_epochs=2)

        model.save(output_path)

def create_faiss_index_for_entity_descriptions(model: str, normalize: bool = True,
                                               faiss_index_name: str = "faiss.index",
                                               faiss_indices_name: str = "faiss.indices",
                                               filter_set_path: str = None):
    filter_set = None
    if filter_set_path is not None:
        filter_set = set(json.load(open(filter_set_path, "r")))
    if isinstance(model, str):
        model = init_model(model)
    entity_descriptions = load_entity_descriptions()

    keys = [key for key in entity_descriptions.keys() if (filter_set is None or key in filter_set)]
    entity_indices = {i: key for i, key in enumerate(keys)}
    sentences = [entity_descriptions[key] for key in keys]
    embeddings = model.encode(sentences, batch_size=256, show_progress_bar=True)

    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dimension)
    nlist = int(math.sqrt(len(sentences)))
    nprobe = int(math.sqrt(nlist))  # Taking square root of nlist
    nprobe = min(nprobe, nlist)
    faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss_index.nprobe = nprobe

    faiss_index.train(embeddings)
    print("Index trained")
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, faiss_index_name)
    json.dump(entity_indices, open(faiss_indices_name, "w"))


def generate_candidates(dataset_path: str, model_path: str,
                        faiss_index_path: str = "faiss_index",
                        entity_indices_path: str = "entity_indices_faiss_index.json",
                        normalize: bool = True):
    model = init_model(model_path)
    faiss_index = faiss.read_index(faiss_index_path)
    entity_indices = json.load(open(entity_indices_path))
    dataset = load_dataset(dataset_path, full_dataset=True)
    sentences = [example["context"] for example in dataset]
    document_ids = [example["id"] for example in dataset]
    spans = [example["span"] for example in dataset]
    qids = [example["qid"] for example in dataset]
    encoded_sentences = model.encode(sentences, batch_size=256, show_progress_bar=True)
    if normalize:
        encoded_sentences = encoded_sentences / np.linalg.norm(encoded_sentences, axis=1).reshape(-1, 1)

    # Run the following batch wise
    n = len(encoded_sentences)
    k = 20
    scores = np.empty((n, k), dtype=np.float32)
    indices = np.empty((n, k), dtype=np.int64)

    chunk_size = 10000  # Adjust this value based on the size of your dataset and available memory

    for i in tqdm(range(0, n, chunk_size), desc="Searching", unit="chunks"):
        start = i
        end = min(i + chunk_size, n)
        chunk_scores, chunk_indices = faiss_index.search(encoded_sentences[start:end], k)

        scores[start:end] = chunk_scores
        indices[start:end] = chunk_indices


    candidate_set = defaultdict(dict)
    for document_id, span, qid, idx in zip(document_ids, spans, qids, indices):
        # Make identifier consisting of document_id and span
        candidate_set[document_id][str(tuple(span))] = {
            "candidates": [entity_indices[str(i)] for i in idx],
            "qid": qid,
            "span": span,
            "document_id": document_id
        }
    output_path = dataset_path.replace(".jsonl", "_candidate_set.json")
    json.dump(candidate_set, open(output_path, "w"))




class MODE(Enum):
    TRAIN_ALT = 0
    TRAIN = 1
    TRAIN_HARD = 2
    TRAIN_CE = 3
    CREATE_INDEX = 4
    GENERATE_CANDIDATES = 5


if __name__ == "__main__":

    # Add argumentparser which allows to speciify the dataset names, if the dataset was not yet mapped please map it
    #  furthermore there should be a train mode and a model for generating an index for the entities
    # train should only accept a mapped dataset, a mapped dataset is the original path of the dataset with _mapped in front of the file ending
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_directory", type=str, default="models/small")
    parser.add_argument("--train_dataset", type=str, default="data/rebel_small/en_train_small_filtered.jsonl",
                        help="Path to the training dataset")
    parser.add_argument("--output_path", type=str, default="run_training_bi_encoder",
                        help="Path to the output directory")
    parser.add_argument("--eval_dataset", type=str, default="data/rebel_small/en_val_small_v2_filtered.jsonl",
                        help="Path to the evaluation dataset")
    parser.add_argument("--normalize", type=bool, default=True,
                        help="Whether to normalize the embeddings")
    parser.add_argument("--exclude_types", action="store_true", default=False)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L12-v2",
                        help="Model name")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--candidate_weight", type=float, default=1.0)
    parser.add_argument("--training_candidate_set_path", type=str, default= "data/rebel/en_train_mapped_candidate_set.json",
                        help="Path to the training candidate set")
    parser.add_argument("--eval_candidate_set_path", type=str, default="data/rebel/en_val_mapped_candidate_set.json",
                        help="Path to the evaluation candidate set")
    parser.add_argument("--mode", type=lambda x: MODE[x], default=MODE.TRAIN,
                        help="Whether to train the model, create an index or generate candidates")
    parser.add_argument("--candidate_generation_dataset", type=str, default= "data/rebel/en_train_small_filtered.jsonl",
                        help="Path to the dataset for which to generate candidates")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--types_index_path", type=str, default=None)
    parser.add_argument("--filter_set_path", type=str, default=None)
    parser.add_argument("--type_dictionary_file", type=str, default="data/item_types_relation_extraction_alt.jsonl")
    args = parser.parse_args()
    model_directory =  args.model_directory
    faiss_index_name = f"faiss_index_{model_directory.split('/')[-1]}"
    faiss_indices_name = f"entity_indices_{faiss_index_name}.json"
    if not os.path.exists(args.eval_dataset.replace(".jsonl", "_mapped.jsonl")):
        map_dataset(args.eval_dataset)
    if not os.path.exists(args.train_dataset.replace(".jsonl", "_mapped.jsonl")):
        map_dataset(args.train_dataset)
    if args.mode == MODE.TRAIN:
        train(args.train_dataset.replace(".jsonl", "_mapped.jsonl"), args.eval_dataset.replace(".jsonl", "_mapped.jsonl"),
              args.normalize, args.output_path, model_name=args.model_name,
              batch_size=args.batch_size, epochs_of_training_before_regeneration=None)
    elif args.mode == MODE.TRAIN_ALT:
        train(args.train_dataset, args.eval_dataset,
                args.normalize, args.output_path, model_name=args.model_name,
                                  batch_size=args.batch_size, epochs_of_training_before_regeneration=0)
    elif args.mode == MODE.TRAIN_HARD:
        train(args.train_dataset, args.eval_dataset,
              args.normalize, args.output_path, model_name=args.model_name,
              batch_size=args.batch_size, epochs_of_training_before_regeneration=2, evaluate_every_n_epochs=2,
              manually=True)
    elif args.mode == MODE.TRAIN_CE:
        train_crossencoder(args.train_dataset, args.eval_dataset, args.training_candidate_set_path,
                            args.eval_candidate_set_path, model_name=args.model_name,
                           batch_size=args.batch_size,
                           checkpoint_path=args.checkpoint_path,
                           candidate_weight=args.candidate_weight,
                           num_candidates=args.num_candidates,
                           types_index_path=args.types_index_path, include_types=not args.exclude_types,
                           type_dictionary_file=args.type_dictionary_file)
    elif args.mode == MODE.CREATE_INDEX:
        create_faiss_index_for_entity_descriptions(model_directory,
                                                   normalize=args.normalize,faiss_index_name=faiss_index_name,
                                                   faiss_indices_name=faiss_indices_name,
                                                   filter_set_path=args.filter_set_path)
    elif args.mode == MODE.GENERATE_CANDIDATES:
        if not os.path.exists(args.candidate_generation_dataset.replace(".jsonl", "_mapped.jsonl")):
            map_dataset(args.candidate_generation_dataset)
        generate_candidates(args.candidate_generation_dataset.replace(".jsonl", "_mapped.jsonl"),
                            model_directory, faiss_index_path=faiss_index_name,
                            entity_indices_path=faiss_indices_name)







