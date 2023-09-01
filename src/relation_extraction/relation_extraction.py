import argparse
import os
from datetime import datetime

import ojson as json
import math
import random

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

from src.candidate_generation.candidate_generator import get_boundaries, load_property_indices, \
    get_boundaries_crossencoder


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
            logits = torch.cat(
                [logits, torch.zeros((logits.size(0), logits.size(1), logits.size(2), 1), device=logits.device)],
                dim=-1)
            logits[:, :, 1:, -1] = - math.inf
            logits[:, 1:, :, -1] = - math.inf

            scores = torch.logsumexp(logits, dim=(-1, -2, -3))
        return torch.mean(scores)


class FocalLossMultiLabel(nn.Module):
    def __init__(self, alpha=1.0, gamma=0.0):
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


class PairwiseRelationRecognizer(LightningModule):
    def __init__(self, model_name: str, num_relations: int, mode='att_based' , number_of_types: int = None,
                 deactivate_text: bool = False):
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
        self.combine_matrix = Parameter(torch.randn(self.model.config.hidden_size, num_relations))
        self.use_types = False
        self.deactivate_text = deactivate_text
        if number_of_types is not None:
            type_dim = 100
            self.use_types = True
            # +1 for no type
            self.type_embeddings = torch.nn.Embedding(number_of_types + 1, type_dim)
            self.type_key_linear = torch.nn.Linear(type_dim, type_dim)
            self.type_query_linear = torch.nn.Linear(type_dim, type_dim)
            self.type_combine_matrix = Parameter(torch.randn(type_dim, num_relations))
            self.type_combiner = torch.nn.Linear(2 * type_dim, num_relations)
        self.num_relations = num_relations


        self.mode = mode

        if self.mode == "att_based":
            self.loss = CrossEntropyLoss()
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
        scores = pairwise_sums + partial_scores.unsqueeze(-1).expand_as(pairwise_sums)
        return scores

    def forward(self, mention_positions, mention_types, alignment=None, *args, **kwargs):
        padding_matrix = []
        for i, mention_positions_ in enumerate(mention_positions):
            counter = 0
            for start, end in mention_positions_:
                if start != -1:
                    counter += 1
            padding_matrix.append(torch.ones(counter, device=self.device))
        padding_matrix = pad_sequence(padding_matrix, batch_first=True, padding_value=0)
        if not self.deactivate_text:
            padding_matrix = []
            embedded = self.model(*args, **kwargs).last_hidden_state
            batch_wise_mentions = []

            for i, mention_positions_ in enumerate(mention_positions):
                mentions = []
                for start, end in mention_positions_:
                    if start != -1:
                        mention = embedded[i, start, :] + embedded[i, end, :]
                        mention = mention / 2
                        mentions.append(mention)
                mentions = torch.stack(mentions)
                batch_wise_mentions.append(mentions)
                padding_matrix.append(torch.ones(mentions.size(0), device=self.device))
            batch_wise_mentions = pad_sequence(batch_wise_mentions, batch_first=True, padding_value=0)
            padding_matrix = pad_sequence(padding_matrix, batch_first=True, padding_value=0)

            if self.mode == 'product':
                scores = self.product(batch_wise_mentions)
            elif self.mode == 'cross_encode':
                scores = self.cross_encode(batch_wise_mentions)
            elif self.mode == 'att_based':
                scores = self.att_based(batch_wise_mentions)
            else:
                raise ValueError(f'Unknown mode: {self.mode}')
        else:
            scores = torch.zeros(mention_positions.size(0), mention_positions.size(1), mention_positions.size(1),
                                 self.num_relations, device=self.device)
        if alignment:
            scores = torch.repeat_interleave(scores, alignment, dim=1).repeat_interleave(alignment, dim=2)
            padding_matrix = torch.repeat_interleave(padding_matrix, alignment, dim=1)
        if self.use_types:
            indices = mention_types
            indices[indices == -1] = self.type_embeddings.num_embeddings - 1
            padding = indices != self.type_embeddings.num_embeddings - 1
            num_types = torch.sum(padding, dim=-1)
            padding[num_types == 0, :] = True
            num_types[num_types == 0] = padding.size(-1)
            type_embeddings = self.type_embeddings(indices)
            type_embeddings[~padding, :] = 0.0
            summed_type_embeddings = type_embeddings.sum(dim=2)
            type_embeddings = summed_type_embeddings / num_types.unsqueeze(-1).expand_as(summed_type_embeddings)


            #type_keys = self.type_key_linear(type_embeddings)
            #type_queries = self.type_query_linear(type_embeddings)
            #type_scores = torch.matmul(type_keys, type_queries.transpose(-1, -2))

            #pairwise_type_scores= type_embeddings.unsqueeze(2) + type_embeddings.unsqueeze(1)
            #pairwise_type_scores = torch.matmul(pairwise_type_scores, self.type_combine_matrix).squeeze(-1)

            #type_scores = type_scores.unsqueeze(-1).expand_as(pairwise_type_scores)

            #scores += type_scores + pairwise_type_scores


            # Expand dimensions
            x1 = type_embeddings.unsqueeze(2)  # Shape becomes (B, 1, D)
            x2 = type_embeddings.unsqueeze(1)  # Shape becomes (1, B, D)

            x1 = x1.repeat(1, 1, x1.size(1), 1)  # Shape becomes (B, B, D)
            x2 = x2.repeat(1, x2.size(2), 1, 1)  # Shape becomes (B, B, D)

            # Due to broadcasting rules, when we concatenate `x1` and `x2` along the last dimension,
            # `x1` will be repeated along the second dimension and `x2` will be repeated along the first dimension
            result = torch.cat([x1, x2], dim=-1)  # Shape becomes (B, B, 2D)
            types_based_relation_scores = self.type_combiner(result).squeeze(-1)
            scores += types_based_relation_scores

        return scores, padding_matrix

    def training_step(self, batch, batch_idx):
        inputs, mention_types, labels, triple_labels = batch
        outputs, padding_matrix = self(labels, mention_types, **inputs)
        new_labels = torch.zeros(outputs.shape, device=self.device)
        for i, labels_ in enumerate(triple_labels):
            for label in labels_:
                if label[0] != -1:
                    new_labels[i, label[0], label[1], label[2]] = 1.0
        padding_matrix = padding_matrix.unsqueeze(-1) * padding_matrix.unsqueeze(-2)

        final_labels = new_labels[padding_matrix == 1]
        final_outputs = outputs[padding_matrix == 1]
        if self.mode == 'att_based':
            new_labels[new_labels == 0] = -1
            new_labels[padding_matrix == 0] = 0
            loss = self.loss(outputs, new_labels, padding_matrix)
            negative_loss_mean = 0.0
            positive_loss_mean = 0.0
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
        outputs = torch.sigmoid(outputs)
        true_positives = torch.sum((outputs >= 0.5) * (labels == 1))
        false_positives = torch.sum((outputs >= 0.5) * (labels == 0))
        false_negatives = torch.sum((outputs < 0.5) * (labels == 1))
        return true_positives, false_positives, false_negatives

    def validation_step(self, batch, batch_idx):
        inputs, mention_types, labels, triple_labels = batch
        outputs, padding_matrix = self(labels, mention_types, **inputs)
        new_labels = torch.zeros(outputs.shape, device=self.device)
        for i, labels_ in enumerate(triple_labels):
            for label in labels_:
                if label[0] != -1:
                    new_labels[i, label[0], label[1], label[2]] = 1.0
        padding_matrix = padding_matrix.unsqueeze(-1) * padding_matrix.unsqueeze(-2)

        final_labels = new_labels[padding_matrix == 1].view(-1)
        final_outputs = outputs[padding_matrix == 1].view(-1)
        if self.mode == 'att_based':
            new_labels[new_labels == 0] = -1
            new_labels[padding_matrix == 0] = 0
            loss = self.loss(outputs, new_labels, padding_matrix)
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


def get_tokenized_dataset(dataset_path: str, tokenizer, property_indices, types_index=None):
    all_examples = []
    with jsonlines.open(dataset_path) as reader:
        for example in tqdm(reader):
            try:
                boundaries, qids, triples = get_boundaries_crossencoder(example, property_indices)
                tokens = tokenizer(example["input"], return_offsets_mapping=True, padding=True, truncation=True)
                offsets = tokens["offset_mapping"]
                mention_types = []
                mention_tokens = []
                for (entity_start, entity_end), qid in zip(boundaries, qids):
                    # Find the first token whose start offset matches or exceeds the entity start position
                    start_token = next((i for i, (start, _) in enumerate(offsets) if start >= entity_start and i != 0), None)

                    # Find the last token whose end offset matches or exceeds the entity end position
                    end_token = next((i for i, (_, end) in enumerate(offsets) if end >= entity_end and i != 0), None)

                    if types_index is not None:
                        if qid in types_index:
                            mention_types.append(types_index[qid])
                        else:
                            mention_types.append([])

                    # Ensure both start_token and end_token are not None
                    if start_token is not None and end_token is not None:
                        mention_tokens.append((start_token, end_token))
                    else:
                        raise Exception("Invalid boundaries")
                tokens["mention_types"] = mention_types
                tokens["labels"] = mention_tokens
                tokens["triple_labels"] = triples
            except Exception as e:
                continue

            all_examples.append(tokens)

    return all_examples


def custom_collate_fn(batch):
    # Pad
    input_ids = pad_sequence([torch.tensor(item["input_ids"])for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([torch.tensor(item["attention_mask"])for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.tensor(item["labels"]) for item in batch], batch_first=True, padding_value=-1)
    triple_labels = pad_sequence([torch.tensor(item["triple_labels"]) for item in batch], batch_first=True, padding_value=-1)
    num_types = [len(types) for item in batch for types in item["mention_types"]]
    if not num_types:
        maximum_num_types = 0
    else:
        maximum_num_types = max(num_types)
    maximum_num_mentions = max([len(item["mention_types"]) for item in batch])
    mention_types = -torch.ones((len(batch), maximum_num_mentions, maximum_num_types), dtype=torch.long)
    for i, item in enumerate(batch):
        for j, types in enumerate(item["mention_types"]):
            for k, type in enumerate(types):
                mention_types[i, j, k] = type

    return {"input_ids": input_ids, "attention_mask": attention_mask}, mention_types, labels, triple_labels


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def remap():
    types_dictionary = json.load(open("data/item_types_relation_extraction.json"))
    new_types_dictionary = jsonlines.open("data/item_types_relation_extraction.jsonl", "w")
    for k, v in tqdm(types_dictionary.items()):
        new_types_dictionary.write({"item": k, "types": v})

def get_type_dictionary(filter_set=None):
    types_dictionary = {}
    types_to_include = set()
    counter = 0
    for item in jsonlines.open("data/item_types_relation_extraction_alt.jsonl"):
        if filter_set is not None and item["item"] not in filter_set:
            continue
        types_dictionary[item["item"]] = set(item["types"])
        types_to_include.update(item["types"])
        counter += 1
        #if counter > 100000:
        #    break
    types_to_include = sorted(list(types_to_include))
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
def main(model_name: str = "sentence-transformers/all-MiniLM-L12-v2", output_path: str = "relation_extractor",
         train_dataset_path: str = "data/rebel/en_train.jsonl",
         val_dataset_path: str = "data/rebel/en_val.jsonl",
         batch_size=16,
         include_types=True,
         deactivate_text=False):
    set_seed()

    property_indices = load_property_indices()
    all_qids = set()
    all_qids.update(get_all_eligible_qids(train_dataset_path, property_indices))
    all_qids.update(get_all_eligible_qids(val_dataset_path, property_indices))
    types_dictionary = None
    types_index = None
    if include_types:
        types_dictionary, types_index = get_type_dictionary(all_qids)
    if os.path.exists(output_path):
        output_path = os.path.join(output_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_path, exist_ok=True)
    json.dump(types_index, open(os.path.join(output_path, "types_index.json"), "w"))

    model = PairwiseRelationRecognizer(model_name, len(property_indices), number_of_types=len(types_index) if types_index else None,
                                       deactivate_text=deactivate_text)
    dataset = get_tokenized_dataset(train_dataset_path, model.tokenizer, property_indices, types_dictionary)
    val_dataset = get_tokenized_dataset(val_dataset_path, model.tokenizer, property_indices, types_dictionary)

    del types_dictionary
    del types_index

    wandb_logger = WandbLogger(project="relation_extractor", name=output_path)
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
    trainer = Trainer(max_epochs=30, logger=wandb_logger, callbacks=[checkpoint_callback, checkpoint_callback_4,
                                                                     checkpoint_callback_2, checkpoint_callback_3],
                      gpus=num_gpus, accelerator=("ddp" if num_gpus > 1 else None) )
    trainer.fit(model, dataloader, val_dataloader)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--include_types", action="store_true", default=False)
    argparser.add_argument("--deactivate_text", action="store_true", default=False)
    args = argparser.parse_args()
    main(include_types=args.include_types, deactivate_text=args.deactivate_text)


