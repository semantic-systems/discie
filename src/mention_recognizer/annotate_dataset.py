import argparse
import hashlib
import json
from collections import defaultdict
from typing import List, Tuple

import faiss
import jsonlines
import numpy
import torch
from tqdm import tqdm

from src.candidate_generation.candidate_generator import init_model, prepare_bi_encoder_input, get_candidates, \
    load_entity_descriptions
from src.mention_recognizer.pairwise_mention_recognizer import PairwiseMentionRecognizer


def get_boundaries(texts, mention_tokens, tokenized):
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




def get_mention_tokens(mention_matrices: torch.Tensor, attention_matrices: torch.Tensor) -> List[
    List[Tuple[int, int, float]]]:
    mention_tokens = []
    for mention_matrix, attention_matrix in zip(mention_matrices, attention_matrices):
        scores = torch.sigmoid(mention_matrix)
        attention_matrix = attention_matrix.triu(diagonal=0)
        scores = scores * attention_matrix
        scores = scores.cpu().detach().numpy()
        mention_tokens_for_text = []
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                if scores[i][j] > 0.1:
                    mention_tokens_for_text.append((i if i != 0 else i + 1, j, scores[i][j]))
        mention_tokens.append(mention_tokens_for_text)
    return mention_tokens

def get_mention_boundaries(texts, mention_recognizer):
    tokenized = mention_recognizer.tokenizer(texts, return_tensors="pt", return_offsets_mapping=True, padding=True,
                                                  truncation=True)
    input_ids = tokenized["input_ids"].to(mention_recognizer.device)
    attention_mask = tokenized["attention_mask"].to(mention_recognizer.device)
    mention_matrices = mention_recognizer(input_ids, attention_mask)
    # Calculate combination
    attention_matrices = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)
    mention_tokens = get_mention_tokens(mention_matrices, attention_matrices)
    all_boundaries = get_boundaries(texts, mention_tokens, tokenized)
    return all_boundaries


def compare_to_ground_truth(prepared_entity_descriptions, prepared, bi_encoder, batch_size=1024):
    all_most_similar_mentions = []
    for i in tqdm(range(0, len(prepared), batch_size)):
        example_indices = []
        concatenated_prepared = []
        concatenated_entity_descriptions = []
        for j, (entity_description, prepared_) in enumerate(zip(prepared_entity_descriptions[i:i + batch_size], prepared[i:i + batch_size])):
            example_indices += [j] * len(entity_description)
            concatenated_prepared.extend(prepared_)
            concatenated_entity_descriptions.extend(entity_description)
        encoded = bi_encoder.encode(concatenated_entity_descriptions + concatenated_prepared)
        encoded_prepared = encoded[len(concatenated_entity_descriptions):]
        encoded_entity_description = encoded[:len(concatenated_entity_descriptions)]
        normalized_entity_description = encoded_entity_description / numpy.linalg.norm(encoded_entity_description, axis=1, keepdims=True)
        normalized_prepared = encoded_prepared / numpy.linalg.norm(encoded_prepared, axis=1, keepdims=True)
        split_normalized_entity_description = numpy.split(normalized_entity_description, numpy.cumsum([len(x) for x in prepared_entity_descriptions[i:i + batch_size]]))
        split_normalized_prepared = numpy.split(normalized_prepared, numpy.cumsum([len(x) for x in prepared[i:i + batch_size]]))
        for j, (entity_description, prepared_) in enumerate(zip(split_normalized_entity_description, split_normalized_prepared)):
            similarity = numpy.dot(entity_description, prepared_.T)
            if similarity.shape[0] > 0:
                most_similar_mentions = list(numpy.argmax(similarity, axis=1))
                all_most_similar_mentions.append(most_similar_mentions)



    # for i, (entity_description, prepared_) in enumerate(zip(prepared_entity_descriptions, prepared)):
    #     encoded_entity_description = bi_encoder.encode(entity_description)
    #     encoded_prepared = bi_encoder.encode(prepared_)
    #     normalized_entity_description = encoded_entity_description / numpy.linalg.norm(encoded_entity_description, axis=1, keepdims=True)
    #     normalized_prepared = encoded_prepared / numpy.linalg.norm(encoded_prepared, axis=1, keepdims=True)
    #     similarity = numpy.dot(normalized_entity_description, normalized_prepared.T)
    #     most_similar_mentions = list(numpy.argmax(similarity, axis=1))
    #     all_most_similar_mentions.append(most_similar_mentions)
    return all_most_similar_mentions

def get_boundaries_by_candidate_generator(dataset_path, candidate_generation_tuple: tuple, mention_recognizer, entity_descriptions,
                                          batch_size: int = 256):
    all_boundaries_for_candidate_generation = []
    all_texts = []
    all_ground_truth_qids = []
    with jsonlines.open(dataset_path) as reader:
        for idx, example in enumerate(tqdm(reader)):
            boundaries = []
            ground_truth_qids = []
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

            bi_encoder, faiss_index, entity_indices = candidate_generation_tuple
            boundaries_for_candidate_generation = []
            for entity_boundaries in entity_boundaries_by_length:
                for idx, boundary in enumerate(entity_boundaries):
                    for boundary_ in entity_boundaries[idx:]:
                        boundaries_for_candidate_generation.append((boundary[0], boundary_[1]))
            all_boundaries_for_candidate_generation.append(boundaries_for_candidate_generation)
            all_texts.append(example["input"])
            all_ground_truth_qids.append(ground_truth_qids)
    all_additional_mention_boundaries = []
    for i in tqdm(range(0, len(all_texts), batch_size)):
        texts = all_texts[i:i + batch_size]
        additional_mention_boundaries = get_mention_boundaries(texts,mention_recognizer)
        additional_mention_boundaries = [[(int(start), int(end)) for start, end in boundaries] for boundaries in
                                            additional_mention_boundaries]
        all_additional_mention_boundaries.extend(additional_mention_boundaries)
    all_boundaries_for_candidate_generation = [boundaries + additional_mention_boundaries for boundaries,
                                                                                           additional_mention_boundaries in
                                                                                           zip(
                                                                                               all_boundaries_for_candidate_generation,
                                                                                               all_additional_mention_boundaries)]
    prepared = prepare_bi_encoder_input(all_texts,
                                            all_boundaries_for_candidate_generation)



    prepared_entity_descriptions = []
    for ground_truth_qids in all_ground_truth_qids:
        descriptions = []
        for ground_truth_qid in ground_truth_qids:
            if ground_truth_qid not in entity_descriptions:
                descriptions.append("")
            else:
                description = entity_descriptions[ground_truth_qid]
                descriptions.append(description)
        prepared_entity_descriptions.append(descriptions)

    all_most_similar_mentions = compare_to_ground_truth(prepared_entity_descriptions, prepared, bi_encoder)

    # all_candidates = {}
    # for i in tqdm(range(0, len(prepared), batch_size)):
    #     sub_prepared = prepared[i:i + batch_size]
    #     new_candidates = get_candidates(bi_encoder, faiss_index, entity_indices, sub_prepared)
    #     new_candidates = {key + len(all_candidates): value for key, value in new_candidates.items()}
    #     all_candidates.update(new_candidates)
    all_unique_boundaries = []
    all_unique_ground_truth_qids = []

    for idx, (boundaries_for_candidate_generation, orig_ground_truth_qids) in enumerate(zip(all_boundaries_for_candidate_generation,
                                                                        all_ground_truth_qids)):


        ground_truth_qids_set = set(orig_ground_truth_qids)
        # best_per_ground_truth = defaultdict(lambda: (None, -1.0))
        # for idx_, boundary in enumerate(boundaries_for_candidate_generation):
        #     candidates = all_candidates[idx][idx_]
        #     candidates_with_scores = {candidate: score for candidate, score in candidates}
        #     if len(candidates) > 0:
        #         for ground_truth_qid in ground_truth_qids_set:
        #             if ground_truth_qid in candidates_with_scores:
        #                 if candidates_with_scores[ground_truth_qid] > best_per_ground_truth[ground_truth_qid][
        #                     1]:
        #                     best_per_ground_truth[ground_truth_qid] = (
        #                     boundary, candidates_with_scores[ground_truth_qid])

        ground_truth_qids = []
        boundaries = []
        mentions = []
        most_similar_mentions = all_most_similar_mentions[idx]
        for ground_truth_idx in range(len(orig_ground_truth_qids)):
            boundaries.append(boundaries_for_candidate_generation[most_similar_mentions[ground_truth_idx]])
            ground_truth_qids.append(orig_ground_truth_qids[ground_truth_idx])
            mentions.append(all_texts[idx][boundaries[-1][0]:boundaries[-1][1]])

        # ground_truth_qids = []
        # boundaries = []
        # mentions = []
        # for ground_truth_qid, (boundary, score) in best_per_ground_truth.items():
        #     if boundary is not None:
        #         boundaries.append(boundary)
        #         ground_truth_qids.append(ground_truth_qid)
        #         mentions.append(all_texts[idx][boundary[0]:boundary[1]])

        # Sort boundaries and ground_truth_qids by boundaries, smallest first
        if len(boundaries) > 0:
            boundaries, ground_truth_qids = zip(*sorted(zip(boundaries, ground_truth_qids), key=lambda x: x[0][0]))
        # Only allow unique boundaries, if there are multiple entities with the same boundary, only the first one is kept
        # This means that the ground_truth_qids are also shortened
        unique_boundaries = []
        unique_ground_truth_qids = []
        for boundary, qid in zip(boundaries, ground_truth_qids):
            if boundary not in unique_boundaries:
                unique_boundaries.append(boundary)
                unique_ground_truth_qids.append(qid)

        if len(boundaries) < len(ground_truth_qids_set):
            all_unique_boundaries.append(None)
            all_unique_ground_truth_qids.append(None)
        else:
            all_unique_boundaries.append(unique_boundaries)
            all_unique_ground_truth_qids.append(unique_ground_truth_qids)
    return all_unique_boundaries, all_unique_ground_truth_qids


def get_candidate_generator_tuple(model_name: str):
    bi_encoder = init_model(model_name)
    if torch.cuda.is_available():
        bi_encoder.cuda()
    bi_encoder.eval()
    index_name = "/data1/moeller/GenIE/indices/" + hashlib.md5(model_name.encode('utf-8')).hexdigest()[
                                                        0:10]

    index_path = index_name + "/faiss.index"
    indices_path = index_name + "/faiss.indices"
    faiss_index = faiss.read_index(index_path)
    entity_indices = json.load(open(indices_path))
    return bi_encoder, faiss_index, entity_indices


def annotate_dataset(dataset_path, candidate_generation_tuple, mention_recognizer):
    entity_descriptions = load_entity_descriptions()
    counter_errors = 0
    counter_all = 0
    overlap_counter = 0
    all_unique_boundaries, all_unique_ground_truth_qids = get_boundaries_by_candidate_generator(dataset_path,
                                                                                                candidate_generation_tuple,
                                                                                                mention_recognizer,
                                                                                                entity_descriptions)

    new_dataset_path = dataset_path.replace(".jsonl", "_annotated.jsonl")
    with jsonlines.open(new_dataset_path, mode='w') as writer:
        with jsonlines.open(dataset_path) as reader:
            for example, boundaries, qids in zip(tqdm(reader), all_unique_boundaries, all_unique_ground_truth_qids):
                counter_all += 1
                if boundaries is None:
                    counter_errors += 1
                    continue
                overlapping_boundaries = defaultdict(set)
                for idx, boundary in enumerate(boundaries):
                    overlapping_boundaries[boundary].add(qids[idx])
                if len(overlapping_boundaries) < len(boundaries):
                    overlap_counter += 1
                example["boundaries"] = [list(x) for x in boundaries]
                example["qids"] = qids
                writer.write(example)
    print(f"Overlap: {overlap_counter}/{counter_all}")
    print(f"Errors: {counter_errors}/{counter_all}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="/data1/moeller/GenIE/run_training_bi_encoder_new")
    parser.add_argument("--mention_recognizer_path", type=str,
                           default="/data1/moeller/GenIE/mention_recognizer_2023-07-22_18-10-13/model-epoch=06-val_f1=0.85_val_f1.ckpt")
    args = parser.parse_args()
    mention_recognizer = PairwiseMentionRecognizer.load_from_checkpoint(args.mention_recognizer_path,
                                                                        model_name="distilbert-base-cased")
    if torch.cuda.is_available():
        mention_recognizer.cuda()
    mention_recognizer.eval()
    candidate_generation_tuple = get_candidate_generator_tuple(args.model_name)
    annotate_dataset(args.dataset_path, candidate_generation_tuple, mention_recognizer)