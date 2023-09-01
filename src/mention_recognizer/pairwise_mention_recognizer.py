import argparse
import hashlib
import json
import math
import random
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import faiss
import jsonlines
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.candidate_generation.candidate_generator import get_boundaries, get_mention_input, init_model, \
    prepare_bi_encoder_input, get_candidates


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean', split_positive_negative=False):
        super().__init__()
        self.reduction = reduction
        self.split_positive_negative = split_positive_negative
    def forward(self, outputs, labels, attention_matrix=None):
        logits = -labels * outputs
        logits[attention_matrix == 0] = - math.inf
        if self.split_positive_negative:
            positive_logits = logits
            positive_logits[labels == -1] = - math.inf
            positive_logits = torch.cat([positive_logits, torch.zeros((positive_logits.size(0), positive_logits.size(1), 1), device=positive_logits.device)], dim=-1)
            positive_logits[:, 1:, -1] = - math.inf
            positive_scores = torch.logsumexp(positive_logits, dim=(-1, -2))
            negative_logits = logits
            negative_logits[labels == 1] = - math.inf
            negative_logits = torch.cat([negative_logits, torch.zeros((negative_logits.size(0), negative_logits.size(1), 1), device=negative_logits.device)], dim=-1)
            negative_logits[:, 1:, -1] = - math.inf
            negative_scores = torch.logsumexp(negative_logits, dim=(-1, -2))
            scores = positive_scores + negative_scores
        else:
            logits = torch.cat([logits, torch.zeros((logits.size(0), logits.size(1), 1), device=logits.device)], dim=-1)
            logits[:, 1:, -1] = - math.inf
            scores = torch.logsumexp(logits, dim=(-1, -2))
        return torch.mean(scores)


class FocalLossMultiLabel(nn.Module):
    def __init__(self, alpha=1, gamma=0):
        super(FocalLossMultiLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sub_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets, attention_matrix=None):
        inputs_to_use = inputs[attention_matrix == 1]
        targets_to_use = targets[attention_matrix == 1]
        targets_to_use[targets_to_use == -1] = 0
        BCE_loss = self.sub_loss(inputs_to_use, targets_to_use)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)


class PairwiseMentionRecognizer(LightningModule):
    def __init__(self, model_name: str, mode='att_based', loss_type='focal_loss'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Dim x Dim
        self.combination_matrix = Parameter(torch.randn(self.model.config.hidden_size, self.model.config.hidden_size))
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(2 * self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.model.config.hidden_size, 1))
        self.key_linear = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.query_linear = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.combine_matrix = Parameter(torch.randn(self.model.config.hidden_size, 1))

        self.mode = mode

        self.loss_type = loss_type

        if self.loss_type == "cross_entropy":
            self.loss = CrossEntropyLoss()
        elif self.loss_type == "focal_loss":
            self.loss = FocalLossMultiLabel()
        else:
            self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')




    def cross_encode(self, embedded):
        # Expand dimensions
        x1 = embedded.unsqueeze(2)  # Shape becomes (B, 1, D)
        x2 = embedded.unsqueeze(1)  # Shape becomes (1, B, D)

        x1 = x1.repeat(1, 1, x1.size(1), 1)  # Shape becomes (B, B, D)
        x2 = x2.repeat(1, x2.size(2), 1, 1)  # Shape becomes (B, B, D)

        # Due to broadcasting rules, when we concatenate `x1` and `x2` along the last dimension,
        # `x1` will be repeated along the second dimension and `x2` will be repeated along the first dimension
        result = torch.cat([x1, x2], dim=-1)  # Shape becomes (B, B, 2D)

        scores = self.sequential(result).squeeze(-1)
        return scores


    def product(self, embedded):
        return torch.matmul(embedded, torch.matmul(self.combination_matrix, embedded.transpose(-1, -2)))

    def att_based(self, embedded):
        keys = self.key_linear(embedded)
        queries = self.query_linear(embedded)
        partial_scores = torch.matmul(keys, queries.transpose(-1, -2))
        # Calculate pairwise sums of embedded
        pairwise_sums = embedded.unsqueeze(2) + embedded.unsqueeze(1)
        pairwise_sums = torch.matmul(pairwise_sums, self.combine_matrix).squeeze(-1)
        # Calculate scores
        scores = partial_scores + pairwise_sums
        return scores

    def forward(self, *args, **kwargs):
        embedded = self.model(*args, **kwargs).last_hidden_state

        if self.mode == 'product':
            scores = self.product(embedded)
        elif self.mode == 'cross_encode':
            scores = self.cross_encode(embedded)
        elif self.mode == 'att_based':
            scores = self.att_based(embedded)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        return scores

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        new_labels = torch.zeros(outputs.shape, device=self.device)
        for i, labels_ in enumerate(labels):
            for label in labels_:
                if label[0] != -1:
                    new_labels[i, label[0], label[1]] = 1
        attention_matrix = inputs['attention_mask'].unsqueeze(-1) * inputs['attention_mask'].unsqueeze(-2)
        # Set everything below the diagonal to 0
        attention_matrix = attention_matrix.triu(diagonal=0)

        final_labels = new_labels[attention_matrix == 1]
        final_outputs = outputs[attention_matrix == 1]
        if self.loss_type in {'focal_loss', 'cross_entropy'}:
            new_labels[new_labels == 0] = -1
            new_labels[attention_matrix == 0] = 0
            loss = self.loss(outputs, new_labels, attention_matrix)
        else:
            # Create sequence of new_labels
            loss = self.loss(final_outputs, final_labels)
            negative_loss_mean = loss[final_labels == 0].mean()
            positive_loss_mean = loss[final_labels == 1].mean()
            loss = negative_loss_mean + positive_loss_mean
            self.log('train_negative_loss', negative_loss_mean)
            self.log('train_positive_loss', positive_loss_mean)
        self.log('train_loss', loss)
        if torch.any(torch.isnan(outputs)):
            raise ValueError('Outputs is NaN')
        if torch.any(torch.isnan(loss)):
            raise ValueError('Loss is NaN')
        return loss


    def calculate_tp_fp_fn(self, outputs, labels):
        # Calculate recall, precision and f1
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        outputs = torch.sigmoid(outputs)
        for output, label in zip(outputs, labels):
            if label == 1:
                if output >= 0.5:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if output >= 0.5:
                    false_positives += 1
        return true_positives, false_positives, false_negatives

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        new_labels = torch.zeros(outputs.shape, device=self.device)
        for i, labels_ in enumerate(labels):
            for label in labels_:
                if label[0] != -1:
                    new_labels[i, label[0], label[1]] = 1
        attention_matrix = inputs['attention_mask'].unsqueeze(-1) * inputs['attention_mask'].unsqueeze(-2)
        #Set everything below the diagonal to 0
        attention_matrix = attention_matrix.triu(diagonal=0)

        final_labels = new_labels[attention_matrix == 1]
        final_outputs = outputs[attention_matrix == 1]
        if self.loss_type in {'focal_loss', 'cross_entropy'}:
            new_labels[new_labels == 0] = -1
            new_labels[attention_matrix == 0] = 0
            loss = self.loss(outputs, new_labels, attention_matrix)
            negative_loss_mean = 0.0
            positive_loss_mean = 0.0
        else:
            # Create sequence of new_labels
            loss = self.loss(final_outputs, final_labels)
            negative_loss_mean = loss[final_labels == 0].mean()
            positive_loss_mean = loss[final_labels == 1].mean()
            loss = negative_loss_mean + positive_loss_mean
        tp, fp, np = self.calculate_tp_fp_fn(final_outputs, final_labels)
        return loss, negative_loss_mean, positive_loss_mean, tp, fp, np


    def validation_epoch_end(self, outputs):
        results = torch.stack([torch.tensor(x, device=self.device) for x in outputs])
        loss, negative_loss_mean, positive_loss_mean = results[:, :3].mean(dim=0)
        tp, fp, np = results[:, 3:].sum(dim=0)
        recall = tp / (tp + np) if tp + np > 0 else 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f2 = 5 * (precision * recall) / (4 * precision + recall) if 4 * precision + recall > 0 else 0
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_negative_loss_mean', negative_loss_mean, on_epoch=True)
        self.log('val_positive_loss_mean', positive_loss_mean, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_f1', f1, on_epoch=True)
        self.log('val_f2', f2, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    entity_token_spans =  []
    current_span = []
    for i, label in enumerate(new_labels):
        if label == 1:
            current_span.append(i)
        elif label != 2 and len(current_span) > 0:
            current_span.append(i - 1)
            entity_token_spans.append(current_span)
            current_span = []
    return entity_token_spans


def get_boundaries_by_candidate_generator(dataset_path, candidate_generation_tuple: tuple, batch_size: int = 256):
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
    all_candidates = {}
    prepared = prepare_bi_encoder_input(all_texts,
                                            all_boundaries_for_candidate_generation)
    for i in tqdm(range(0, len(prepared), batch_size)):
        sub_prepared = prepared[i:i + batch_size]
        new_candidates = get_candidates(bi_encoder, faiss_index, entity_indices, sub_prepared)
        new_candidates = {key + len(all_candidates): value for key, value in new_candidates.items()}
        all_candidates.update(new_candidates)
    all_unique_boundaries = []
    all_unique_ground_truth_qids = []

    for idx, (boundaries_for_candidate_generation, orig_ground_truth_qids) in enumerate(zip(all_boundaries_for_candidate_generation,
                                                                        all_ground_truth_qids)):
        ground_truth_qids_set = set(orig_ground_truth_qids)
        best_per_ground_truth = defaultdict(lambda: (None, -1.0))
        for boundary, candidates in zip(boundaries_for_candidate_generation,
                                                    all_candidates[idx].values()):
            candidates_with_scores = {candidate: score for candidate, score in candidates}
            if len(candidates) > 0:
                for ground_truth_qid in ground_truth_qids_set:
                    if ground_truth_qid in candidates_with_scores:
                        if candidates_with_scores[ground_truth_qid] > best_per_ground_truth[ground_truth_qid][
                            1]:
                            best_per_ground_truth[ground_truth_qid] = (
                            boundary, candidates_with_scores[ground_truth_qid])
        ground_truth_qids = []
        boundaries = []
        mentions = []
        for ground_truth_qid, (boundary, score) in best_per_ground_truth.items():
            if boundary is not None:
                boundaries.append(boundary)
                ground_truth_qids.append(ground_truth_qid)
                mentions.append(all_texts[idx][boundary[0]:boundary[1]])

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


def get_tokenized_dataset(dataset_path: str, tokenizer):
    all_examples = []
    with jsonlines.open(dataset_path) as reader:
        for example in tqdm(reader):
            try:
                boundaries, _ = get_boundaries(example)
                tokens = tokenizer(example["input"], return_offsets_mapping=True, padding=True, truncation=True)
                offsets = tokens["offset_mapping"]
                mention_tokens = []
                for entity_start, entity_end in boundaries:
                    if entity_start == entity_end:
                        continue
                    # Find the first token whose start offset matches or exceeds the entity start position
                    start_token = next((i for i, (start, _) in enumerate(offsets) if start >= entity_start and i != 0), None)

                    # Find the last token whose end offset matches or exceeds the entity end position
                    end_token = next((i for i, (_, end) in enumerate(offsets) if end >= entity_end and i != 0), None)

                    # Ensure both start_token and end_token are not None
                    if start_token is not None and end_token is not None:
                        mention_tokens.append((start_token, end_token))
                    else:
                        warnings.warn("Invalid boundaries")
                        continue
                tokens["labels"] = mention_tokens
            except ValueError:
                continue
            if mention_tokens:
                all_examples.append(tokens)

    return all_examples


def get_tokenized_dataset_with_candidate_generation(dataset_path: str, tokenizer, candidate_generation_tuple):
    counter_errors = 0
    counter_all = 0
    all_examples = []
    all_unique_boundaries, all_unique_ground_truth_qids  = get_boundaries_by_candidate_generator(dataset_path, candidate_generation_tuple)
    with jsonlines.open(dataset_path) as reader:
        for example, boundaries, _ in zip(tqdm(reader), all_unique_boundaries, all_unique_ground_truth_qids):
            if boundaries is None:
                counter_errors += 1
                continue
            counter_all += 1
            try:
                tokens = tokenizer(example["input"], return_offsets_mapping=True, padding=True, truncation=True)
                offsets = tokens["offset_mapping"]
                mention_tokens = []
                for entity_start, entity_end in boundaries:
                    if entity_start == entity_end:
                        continue
                    # Find the first token whose start offset matches or exceeds the entity start position
                    start_token = next((i for i, (start, _) in enumerate(offsets) if start >= entity_start and i != 0), None)

                    # Find the last token whose end offset matches or exceeds the entity end position
                    end_token = next((i for i, (_, end) in enumerate(offsets) if end >= entity_end and i != 0), None)

                    # Ensure both start_token and end_token are not None
                    if start_token is not None and end_token is not None:
                        mention_tokens.append((start_token, end_token))
                    else:
                        warnings.warn("Invalid boundaries")
                        continue
                tokens["labels"] = mention_tokens
            except ValueError:
                counter_errors += 1
                continue
            if mention_tokens:
                all_examples.append(tokens)
    print(f"Errors: {counter_errors}/{counter_all}")

    return all_examples


def custom_collate_fn(batch):
    # Pad
    input_ids = pad_sequence([torch.tensor(item["input_ids"])for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([torch.tensor(item["attention_mask"])for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.tensor(item["labels"]) for item in batch], batch_first=True, padding_value=-1)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_candidate_generator_tuple(model_name: str="/data1/moeller/GenIE/run_training_bi_encoder_new"):
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


def main(model_name: str = "distilbert-base-cased", output_path: str = "mention_recognizer",
         train_dataset_path: str = "/data1/moeller/GenIE/data/rebel_small/en_train_small_filtered.jsonl",
         val_dataset_path: str = "/data1/moeller/GenIE/data/rebel_small/en_val_small_v2_filtered.jsonl",
         model_path: str = None,
         batch_size=16):
    set_seed()
    if model_path is not None:
        model = PairwiseMentionRecognizer.load_from_checkpoint(model_path)
    else:
        model = PairwiseMentionRecognizer(model_name)

    dataset = get_tokenized_dataset(train_dataset_path, model.tokenizer)
    val_dataset = get_tokenized_dataset(val_dataset_path, model.tokenizer)

    wandb_logger = WandbLogger(project="mention_recognizer", name=output_path)

    if Path(output_path).exists():
        output_path = output_path + "_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_path,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        save_last=True,
        mode='min',
    )

    checkpoint_callback_2 = ModelCheckpoint(
        monitor='val_f1',
        dirpath=output_path,
        filename='model-{epoch:02d}-{val_f1:.2f}_val_f1',
        save_top_k=1,
        save_last=True,
        mode='max',
    )

    checkpoint_callback_4 = ModelCheckpoint(
        monitor='val_f2',
        dirpath=output_path,
        filename='model-{epoch:02d}-{val_f2:.2f}_val_f2',
        save_top_k=1,
        save_last=True,
        mode='max',
    )

    checkpoint_callback_3 = ModelCheckpoint(
        monitor='val_recall',
        dirpath=output_path,
        filename='model-{epoch:02d}-{val_f1:.2f}_val_recall',
        save_top_k=1,
        save_last=True,
        mode='max',
    )

    num_gpus = torch.cuda.device_count()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    trainer = Trainer(max_epochs=20, logger=wandb_logger, callbacks=[checkpoint_callback, checkpoint_callback_4,
                                                                     checkpoint_callback_2, checkpoint_callback_3],
                      gpus=num_gpus, accelerator=("ddp" if num_gpus > 1 else None) )
    trainer.fit(model, dataloader, val_dataloader)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, default=None)
    argparser.add_argument("--train_dataset_path", type=str, default="/data1/moeller/GenIE/data/rebel/en_train.jsonl")
    argparser.add_argument("--val_dataset_path", type=str, default="/data1/moeller/GenIE/data/rebel/en_val.jsonl")
    args = argparser.parse_args()

    main(train_dataset_path=args.train_dataset_path, val_dataset_path=args.val_dataset_path, model_path=args.model_path)


