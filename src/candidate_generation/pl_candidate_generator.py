import math
from functools import partial
from typing import List, Iterable, Dict, Any

import faiss
import numpy as np
import pytorch_lightning
import torch
import transformers
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from sentence_transformers import InputExample
from sentence_transformers.util import batch_to_device
from torch import nn, Tensor
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from src.candidate_generation.special_tokens import CXS_TOKEN, CXE_TOKEN, TXS_TOKEN


def encode_sentences(sentences, model, tokenizer, normalize: bool, batch_size: int = 32):
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    input_ids, attention = tokenized_sentences["input_ids"], tokenized_sentences["attention_mask"]
    encoded_sentences = []
    for i in range(0, len(sentences), batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention = attention[i:i + batch_size]
        batch_input_ids = batch_input_ids.to(model.device)
        batch_attention = batch_attention.to(model.device)
        with torch.no_grad():
            batch_embeddings = model(batch_input_ids, batch_attention).last_hidden_state[:, 0, :]
        encoded_sentences.extend(batch_embeddings.cpu().numpy())
    encoded_sentences = np.array(encoded_sentences)
    if normalize:
        encoded_sentences = encoded_sentences / np.linalg.norm(encoded_sentences, axis=1).reshape(-1, 1)
    return encoded_sentences


def create_dataset_with_hard_negatives(training_examples: List[InputExample], entity_descriptions: dict,
                                       model: nn.Module, tokenizer: transformers.PreTrainedTokenizer,
                                       normalize: bool, faiss_index, entity_indices):
    texts = []
    ground_truths = []
    for example in training_examples:
        text = example.texts[0]
        labels = example.label
        ground_truth = [t for t, l in zip(example.texts[1:], labels) if l == 1]
        texts.append(text)
        ground_truths.append(ground_truth)
    encoded_sentences = encode_sentences(texts, model, tokenizer, normalize)

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
    def __init__(self, normalize, scale=20.0):
        super().__init__()
        self.normalize = normalize
        self.scale = scale

    def forward(self, scores, labels: Tensor):
        if self.normalize:
            scores *= self.scale

        loss = sum_log_nce_loss(scores, labels)
        return loss



def smart_batching_collate_ce(tokenizer, batch):
    """
    Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
    Here, batch is a list of tuples: [(tokens, label), ...]

    :param batch:
        a batch from a SmartBatchingDataset
    :return:
        a batch of tensors for the model
    """
    all_candidate_labels = []
    all_text_representations = []
    all_triple_labels = []
    mention_indices = []
    example_indices = []
    mention_idx = 0
    for idx, example in enumerate(batch):
        offsets = {}
        counter = 0
        for idx_, text_representations_ in enumerate(example.text_representations):
            all_text_representations += text_representations_
            candidate_labels = [1] + [0] * (len(text_representations_) - 1)
            all_candidate_labels += candidate_labels
            mention_indices += [mention_idx] * len(text_representations_)
            example_indices += [idx] * len(text_representations_)
            offsets[idx_] = counter
            counter += len(text_representations_)
            mention_idx += 1
        triple_labels = []
        if example.triples is not None:
            for sub_idx, obj_idx, p_id in example.triples:
                triple_labels.append([offsets[sub_idx], offsets[obj_idx], p_id])
        if not triple_labels:
            all_triple_labels.append(torch.tensor([[-1, -1, -1]]))
        else:
            all_triple_labels.append(torch.tensor(triple_labels))
    all_triple_labels = pad_sequence(all_triple_labels, batch_first=True, padding_value=-1)
    candidate_labels = torch.tensor(all_candidate_labels, dtype=torch.float32)
    tokenized = tokenizer(all_text_representations, return_tensors='pt', padding=True, truncation=True)

    maximum_num_mentions = max([len(item.mention_types) for item in batch])
    num_types = [len(types) for item in batch for types in item.mention_types]
    if not num_types:
        maximum_num_types = 0
    else:
        maximum_num_types = max(num_types)
    mention_types = -torch.ones((len(batch), maximum_num_mentions, maximum_num_types), dtype=torch.long)
    for i, item in enumerate(batch):
        for j, types in enumerate(item.mention_types):
            for k, type in enumerate(types):
                mention_types[i, j, k] = type

    return tokenized, candidate_labels, all_triple_labels, mention_indices, example_indices, mention_types

def smart_batching_collate(tokenizer, batch):
    """
    Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
    Here, batch is a list of tuples: [(tokens, label), ...]

    :param batch:
        a batch from a SmartBatchingDataset
    :return:
        a batch of tensors for the model
    """
    num_texts = len(batch[0].texts)
    texts = [[] for _ in range(num_texts)]
    labels = []

    for example in batch:
        for idx, text in enumerate(example.texts):
            texts[idx].append(text)

        labels.append(example.label)

    labels = torch.tensor(labels)

    sentence_features = []
    for idx in range(num_texts):
        tokenized = tokenizer(texts[idx], return_tensors='pt', padding=True, truncation=True)
        sentence_features.append(tokenized)

    return sentence_features, labels


def create_dataloader_ce(training_examples, tokenizer, batch_size, shuffle=False, **kwargs):
    return DataLoader(training_examples, shuffle=shuffle, batch_size=batch_size, **kwargs,
               collate_fn=partial(smart_batching_collate_ce, tokenizer))


def create_dataloader(training_examples, tokenizer, batch_size, shuffle=False, **kwargs):
    return DataLoader(training_examples, shuffle=shuffle, batch_size=batch_size, **kwargs,
               collate_fn=partial(smart_batching_collate, tokenizer))

class PLModule(pytorch_lightning.LightningModule):
    def __init__(self, model_name: str,
                 single: bool = True, total_steps: int= 100000,
                 scheduler: str = 'WarmupLinear', warmup_steps: int = 10000,
                 optimizer_params: Dict[str, object] = {'lr': 2e-5},
                 epochs_of_training_before_regeneration: int = -1,
                 entity_descriptions=None,
                 normalize: bool = False,
                 batch_size: int = 128):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer.add_tokens([CXS_TOKEN, CXE_TOKEN, TXS_TOKEN])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.document_model = None
        if not single:
            self.document_model = AutoModel.from_pretrained(model_name)
            self.document_model.resize_token_embeddings(len(self.tokenizer))

        self.loss = AltLoss(normalize)
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.optimizer_params = optimizer_params
        self.epoch_of_training_before_regeneration = epochs_of_training_before_regeneration
        self.entity_descriptions = entity_descriptions
        self.normalize = normalize
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.faiss_index = None
        self.entity_indices = None
        self.epoch_of_index = None
        self.num_candidates = 64

    @staticmethod
    def _get_scheduler(optimizer, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                num_training_steps=t_total)

    def configure_optimizers(self):
        # Here we can configure optimizers and learning rate schedulers
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        scheduler = self._get_scheduler(optimizer, warmup_steps=self.warmup_steps,
                                        t_total=self.total_steps)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    def forward(self,sentence_features, labels) -> Any:
        if self.document_model is not None:
            sentences = self.document_model(**sentence_features[0]).last_hidden_state[:, 0, :]
        else:
            sentences = self.model(**sentence_features[0]).last_hidden_state[:, 0, :]

        candidates = []
        concatenated_sentence_features = torch.stack(
            [self.model(**x).last_hidden_state[:, 0, :] for x in sentence_features[1:]], dim=1)
        for idx, labels_ in enumerate(labels):
            candidates.append(concatenated_sentence_features[idx, labels_ >= 0, :])
        candidates = torch.cat(candidates, dim=0)

        if self.normalize:
            candidates = candidates / torch.norm(candidates, dim=1).unsqueeze(-1)
            sentences = sentences / torch.norm(sentences, dim=1).unsqueeze(-1)

        scores = torch.mm(sentences, candidates.transpose(0, 1))
        return scores

    def training_step(self, batch, batch_idx):
        features, labels = batch
        labels = labels.to(self.device)
        new_labels = torch.zeros((labels.size(0), torch.sum(labels >= 0)), dtype=torch.long,
                                 device=labels.device)
        offset = 0
        for idx, labels_ in enumerate(labels):
            new_labels_ = [l for l in labels_ if l >= 0]
            new_labels[idx, offset:offset + len(new_labels_)] = torch.tensor(new_labels_)
            offset += len(new_labels_)

        features = list(map(lambda batch: batch_to_device(batch, self.device), features))
        scores = self.forward(features, labels)
        loss = self.loss(scores, new_labels)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        labels = labels.to(self.device)
        new_labels = torch.zeros((labels.size(0), torch.sum(labels >= 0)), dtype=torch.long,
                                 device=labels.device)
        offset = 0
        for idx, labels_ in enumerate(labels):
            new_labels_ = [l for l in labels_ if l >= 0]
            new_labels[idx, offset:offset + len(new_labels_)] = torch.tensor(new_labels_)
            offset += len(new_labels_)

        features = list(map(lambda batch: batch_to_device(batch, self.device), features))
        scores = self.forward(features, labels)
        loss = self.loss(scores, new_labels)
        return loss

    def evaluate_if_index_exists(self, faiss_index, entity_indices, dataset):
        sentences = [example[0] for example in dataset]
        qids = [example[1] for example in dataset]
        if self.document_model is not None:
            encoded_sentences = encode_sentences(sentences, self.document_model, self.tokenizer, self.normalize)
        else:
            encoded_sentences = encode_sentences(sentences, self.model, self.tokenizer, self.normalize)
        scores, indices = faiss_index.search(encoded_sentences, self.num_candidates)

        accuracy = 0
        counter = 0
        for qid, idx in zip(qids, indices):
            if isinstance(qid, list):
                valid_qids = {entity_indices[i] for i in idx[:min(len(qid), len(idx))]}
                for idx_, q in enumerate(qid):
                    if q in valid_qids:
                        accuracy += 1
                    counter += 1
            else:
                if qid == entity_indices[idx[0]]:
                    accuracy += 1
                counter += 1
        # Calculate hit@k for k in [1:20]
        for k in range(1, self.num_candidates + 1):
            hits = 0
            counter = 0
            for qid, idx in zip(qids, indices):
                if isinstance(qid, list):
                    valid_qids = {entity_indices[i] for i in idx[:k] if i >= 0}
                    for idx_, q in enumerate(qid):
                        if q in valid_qids:
                            hits += 1
                        counter += 1
                else:
                    if qid in {entity_indices[i] for i in idx[:k]}:
                        hits += 1
                    counter += 1
            self.log("hit@{}".format(k), hits / counter, on_step=False, on_epoch=True, prog_bar=False)
        self.log("accuracy", accuracy / counter, on_step=False, on_epoch=True, prog_bar=False)
        return faiss_index, entity_indices

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        val_loss = torch.stack(outputs).mean()
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def create_index(self, batch_size: int = 32):
        keys = list(self.entity_descriptions.keys())
        entity_indices = {i: key for i, key in enumerate(keys)}
        sentences = [self.entity_descriptions[key] for key in keys]
        tokenized_sentences = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        input_ids, attention = tokenized_sentences["input_ids"], tokenized_sentences["attention_mask"]
        encoded_sentences = []
        for i in range(0, len(sentences), batch_size):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention = attention[i:i + batch_size]
            batch_input_ids = batch_input_ids.to(self.device)
            batch_attention = batch_attention.to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model(batch_input_ids, batch_attention).last_hidden_state[:, 0, :]
            encoded_sentences.extend(batch_embeddings.cpu().numpy())
        embeddings = np.array(encoded_sentences)

        if self.normalize:
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
    def update_dataloader_with_hard_negatives(self, dataset, faiss_index, entity_indices):
        # Construct new dataset with hard negatives
        new_dataset = create_dataset_with_hard_negatives(dataset,
                                                         self.entity_descriptions, self.model, self.tokenizer,
                                                         self.normalize, faiss_index, entity_indices)

        # Update dataloader
        return new_dataset


class BalancedLoss(torch.nn.Module):
    def forward(self, outputs, labels, attention_matrix=None):
        logits = -labels * outputs
        logits[attention_matrix == 0] = - math.inf
        logits = torch.cat([logits, torch.zeros((logits.size(0), logits.size(1), logits.size(2), 1), device=logits.device)], dim=-1)
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


class PLCrossEncoder(pytorch_lightning.LightningModule):
    def __init__(self, model_name: str,
                 num_properties: int,
                 total_steps: int= 100000,
                 scheduler: str = 'WarmupLinear', warmup_steps: int = 10000,
                 optimizer_params: Dict[str, object] = {'lr': 2e-5},
                 epochs_of_training_before_regeneration: int = -1,
                 entity_descriptions=None,
                 alt_mode: bool = True,
                 alt_loss: bool = True,
                 focal_loss: bool = True,
                 batch_size: int = 128,
                 number_of_types: int = None,
                 candidate_weight: float = 1.0,
                 relation_weight: float = 100,
                 relations_to_mask: list = None,):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer.add_tokens([CXS_TOKEN, CXE_TOKEN, TXS_TOKEN])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.property_classification_head = torch.nn.Linear(2 * self.model.config.hidden_size, num_properties)
        self.scoring_head = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.num_properties = num_properties
        self.relations_to_mask = torch.tensor(relations_to_mask) if relations_to_mask is not None else None

        self.candidate_loss = torch.nn.BCEWithLogitsLoss()
        if alt_loss:
            if focal_loss:
                self.relation_loss = FocalLossMultiLabel()
            else:
                self.relation_loss = BalancedLoss()
        else:
            self.relation_loss = torch.nn.BCEWithLogitsLoss()

        if alt_mode:
            self.key_linear = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
            self.query_linear = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
            self.combine_matrix = Parameter(torch.randn(self.model.config.hidden_size, num_properties))
        self.use_types = False
        if number_of_types is not None:
            type_dim = 100
            self.use_types = True
            # +1 for no type
            self.type_embeddings = torch.nn.Embedding(number_of_types + 1, type_dim)
            self.type_key_linear = torch.nn.Linear(type_dim, type_dim)
            self.type_query_linear = torch.nn.Linear(type_dim, type_dim)
            self.type_combine_matrix = Parameter(torch.randn(type_dim, num_properties))
            self.type_combiner = torch.nn.Linear(2 * type_dim, num_properties)
        self.alt_mode = alt_mode
        self.alt_loss = alt_loss

        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.optimizer_params = optimizer_params
        self.epoch_of_training_before_regeneration = epochs_of_training_before_regeneration
        self.entity_descriptions = entity_descriptions
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.num_candidates = 10
        self.candidate_weight = candidate_weight
        self.relation_weight = relation_weight

    @staticmethod
    def _get_scheduler(optimizer, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                num_training_steps=t_total)

    def configure_optimizers(self):
        # Here we can configure optimizers and learning rate schedulers
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        scheduler = self._get_scheduler(optimizer, warmup_steps=self.warmup_steps,
                                        t_total=self.total_steps)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

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


    def concat_based(self, embedded):
        # Expand dimensions
        x1 = embedded.unsqueeze(2)  # Shape becomes (B, 1, D)
        x2 = embedded.unsqueeze(1)  # Shape becomes (1, B, D)

        x1 = x1.repeat(1, 1, x1.size(1), 1)  # Shape becomes (B, B, D)
        x2 = x2.repeat(1, x2.size(2), 1, 1)  # Shape becomes (B, B, D)

        # Due to broadcasting rules, when we concatenate `x1` and `x2` along the last dimension,
        # `x1` will be repeated along the second dimension and `x2` will be repeated along the first dimension
        result = torch.cat([x1, x2], dim=-1)  # Shape becomes (B, B, 2D)

        # Compute property scores
        property_scores = self.property_classification_head(result)  # Shape becomes (B, B, num_properties)

        return property_scores

    def get_batched_candidate_labels(self, example_indices, candidate_labels):
        # Create batched embeddings according to example_indices, it assigns to each element of the embeddings the batch it belongs to
        batched_candidate_labels = []
        current_example = 0
        current_candidate_labels = []
        for i in range(len(example_indices)):
            if example_indices[i] == current_example:
                current_candidate_labels.append(candidate_labels[i])
            else:
                current_example = example_indices[i]
                batched_candidate_labels.append(torch.stack(current_candidate_labels))
                current_candidate_labels = [candidate_labels[i]]
        if len(current_candidate_labels) > 0:
            batched_candidate_labels.append(torch.stack(current_candidate_labels))

        batched_candidate_labels = pad_sequence(batched_candidate_labels, batch_first=True)

        return batched_candidate_labels

    def forward(self,sentence_features, example_indices, mention_types=None) -> Any:
        input_ids = sentence_features['input_ids']
        attention_mask = sentence_features['attention_mask']
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        embeddings = self.model(input_ids, attention_mask).last_hidden_state[:, 0, :]

        scores = self.scoring_head(embeddings)

        if mention_types is not None:
            current_example = 0
            current_example_elements = []
            batched_embeddings = [torch.zeros((0, self.model.config.hidden_size), device=self.device) for _ in range(mention_types.size(0))]
            for i in range(len(example_indices)):
                if example_indices[i] == current_example:
                    current_example_elements.append(embeddings[i])
                else:
                    if current_example_elements:
                        batched_embeddings[current_example] = torch.stack(current_example_elements)
                    current_example = example_indices[i]
                    current_example_elements = [embeddings[i]]
            if len(current_example_elements) > 0:
                batched_embeddings[current_example] = torch.stack(current_example_elements)
        else:
            current_example = 0
            current_example_elements = []
            batched_embeddings = []
            for i in range(len(example_indices)):
                if example_indices[i] == current_example:
                    current_example_elements.append(embeddings[i])
                else:
                    if current_example_elements:
                        batched_embeddings.append(torch.stack(current_example_elements))
                    else:
                        batched_embeddings.append(torch.zeros((0, self.model.config.hidden_size), device=self.device))
                    current_example = example_indices[i]
                    current_example_elements = [embeddings[i]]
            if len(current_example_elements) > 0:
                batched_embeddings.append(torch.stack(current_example_elements))

        batched_embeddings = pad_sequence(batched_embeddings, batch_first=True)


        if self.alt_mode:
            property_scores = self.att_based(batched_embeddings)
        else:
            property_scores = self.concat_based(batched_embeddings)

        if self.use_types and mention_types is not None:
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

            property_scores += self.type_combiner(result).squeeze(-1)
        if self.relations_to_mask is not None:
            self.relations_to_mask = self.relations_to_mask.to(self.device)
            property_scores[:, :, :, self.relations_to_mask] = -1e9
        return scores, property_scores



    def training_step(self, batch, batch_idx):
        tokenized, candidate_labels, triple_labels, mention_indices, example_indices, mention_types = batch
        candidate_labels = candidate_labels.to(self.device)
        triple_labels = triple_labels.to(self.device)

        scores, property_scores = self.forward(tokenized, example_indices, mention_types)
        batched_candidate_labels = self.get_batched_candidate_labels(example_indices, candidate_labels)
        cand_loss = self.candidate_loss(scores.squeeze(), candidate_labels)

        filtering_matrix = []
        labels_for_triples = torch.zeros(
            (property_scores.size(0), property_scores.size(1), property_scores.size(1), self.num_properties),
            device=self.device)
        for i in range(triple_labels.size(0)):
            triple_labels_to_use = triple_labels[i, triple_labels[i, :, 0] != -1]
            labels_for_triples[
                i, triple_labels_to_use[:, 0], triple_labels_to_use[:, 1], triple_labels_to_use[:, 2]] += 1
            candidate_labels_to_use = batched_candidate_labels[i, :]
            expanded_tensor = candidate_labels_to_use.unsqueeze(1)  # Shape: (A, 1)
            # Create a boolean matrix based on element-wise comparison
            filtering_matrix.append((expanded_tensor * candidate_labels_to_use) > 0)
        filtering_matrix = torch.stack(filtering_matrix, dim=0)

        if self.alt_loss:
            positive_loss = 0.0
            negative_loss = 0.0
            labels_for_triples[labels_for_triples == 0] = -1
            labels_for_triples[~filtering_matrix] = 0
            rel_loss = self.relation_loss(property_scores, labels_for_triples, filtering_matrix)
        else:
            collapsed_property_scores = property_scores.view(-1, property_scores.size(-1))
            collapsed_final_filtering_matrix = filtering_matrix.view(-1)
            collapsed_labels_for_triples = labels_for_triples.view(-1, labels_for_triples.size(-1))
            collapsed_property_scores = collapsed_property_scores[collapsed_final_filtering_matrix, :]
            collapsed_labels_for_triples = collapsed_labels_for_triples[collapsed_final_filtering_matrix, :]

            positive_loss = self.relation_loss(collapsed_property_scores[collapsed_labels_for_triples == 1],
                                               collapsed_labels_for_triples[collapsed_labels_for_triples == 1])
            negative_loss = self.relation_loss(collapsed_property_scores[collapsed_labels_for_triples == 0],
                                               collapsed_labels_for_triples[collapsed_labels_for_triples == 0])
            rel_loss = (positive_loss + negative_loss) / 2
        loss = self.relation_weight * rel_loss + self.candidate_weight * cand_loss
        self.log('train_candidate_loss', cand_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_relation_loss', rel_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_negative_relation_loss', negative_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_positive_relation_loss', positive_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss


    def calculate_tp_fp_fn(self, scores, property_scores, candidate_labels, triple_labels, mention_indices, matrix):
        mention_dict = {}
        for i, mention in enumerate(mention_indices):
            if mention not in mention_dict:
                mention_dict[mention] = [(scores[i], candidate_labels[i])]
            else:
                mention_dict[mention].append((scores[i], candidate_labels[i]))

        correct_mentions = 0
        total_mentions = len(mention_dict)

        for mention, candidates in mention_dict.items():
            sorted_candidates = sorted(candidates, key=lambda x: x[0],
                                       reverse=True)  # sort by score in descending order
            if sorted_candidates[0][1] == 1:  # check if the candidate with the highest score is correct
                correct_mentions += 1

        tp = 0
        fp = 0
        fn = 0
        correct = 0
        counter = 0
        property_scores = torch.sigmoid(property_scores)
        for i in range(triple_labels.size(0)):
            for j in range(triple_labels.size(1)):
                if triple_labels[i, j, 0] == -1:
                    continue
                maximum_relation = torch.argmax(property_scores[i, triple_labels[i, j, 0], triple_labels[i, j, 1]])
                if maximum_relation == triple_labels[i, j, 2]:
                    correct += 1
                identified = property_scores[i, triple_labels[i, j, 0], triple_labels[i, j, 1]] > 0.5
                correct_relation = identified[triple_labels[i, j, 2]]
                if correct_relation:
                    tp += 1
                else:
                    fn += 1
                counter += 1
        other_triples = torch.sum(property_scores[matrix] > 0.5) - tp
        fp += other_triples

        return tp, fp, fn, correct_mentions, total_mentions, correct, counter



    def validation_step(self, batch, batch_idx):
        tokenized, candidate_labels, triple_labels, mention_indices, example_indices, mention_types = batch
        candidate_labels = candidate_labels.to(self.device)
        triple_labels = triple_labels.to(self.device)

        scores, property_scores = self.forward(tokenized, example_indices, mention_types)
        batched_candidate_labels = self.get_batched_candidate_labels(example_indices, candidate_labels)
        cand_loss = self.candidate_loss(scores.squeeze(), candidate_labels)

        filtering_matrix = []
        labels_for_triples = torch.zeros((property_scores.size(0), property_scores.size(1), property_scores.size(1), self.num_properties), device=self.device)
        for i in range(triple_labels.size(0)):
            triple_labels_to_use = triple_labels[i, triple_labels[i, :, 0] != -1]
            labels_for_triples[i, triple_labels_to_use[:, 0], triple_labels_to_use[:, 1], triple_labels_to_use[:, 2]] += 1
            candidate_labels_to_use = batched_candidate_labels[i, :]
            expanded_tensor = candidate_labels_to_use.unsqueeze(1)  # Shape: (A, 1)
            # Create a boolean matrix based on element-wise comparison
            filtering_matrix.append((expanded_tensor * candidate_labels_to_use) > 0)
        filtering_matrix = torch.stack(filtering_matrix, dim=0)

        if self.alt_loss:
            positive_loss = 0.0
            negative_loss = 0.0
            labels_for_triples[labels_for_triples == 0] = -1
            labels_for_triples[ ~filtering_matrix] = 0
            rel_loss = self.relation_loss(property_scores, labels_for_triples, filtering_matrix)
        else:
            collapsed_property_scores = property_scores.view(-1, property_scores.size(-1))
            collapsed_final_filtering_matrix = filtering_matrix.view(-1)
            collapsed_labels_for_triples = labels_for_triples.view(-1, labels_for_triples.size(-1))
            collapsed_property_scores = collapsed_property_scores[collapsed_final_filtering_matrix, :]
            collapsed_labels_for_triples = collapsed_labels_for_triples[collapsed_final_filtering_matrix, :]

            positive_loss = self.relation_loss(collapsed_property_scores[collapsed_labels_for_triples == 1],
                                               collapsed_labels_for_triples[collapsed_labels_for_triples == 1])
            negative_loss = self.relation_loss(collapsed_property_scores[collapsed_labels_for_triples == 0],
                                               collapsed_labels_for_triples[collapsed_labels_for_triples == 0])
            rel_loss = (positive_loss + negative_loss) / 2
        loss = self.relation_weight * rel_loss + self.candidate_weight * cand_loss
        tp, fp, fn, correct_mentions, total_mentions, correct_relations, total_relations = self.calculate_tp_fp_fn(scores, property_scores, candidate_labels, triple_labels, mention_indices, filtering_matrix)
        return loss, rel_loss, cand_loss, tp, fp, fn, correct_mentions, total_mentions, correct_relations, total_relations

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results = torch.tensor(outputs)
        (loss, rel_loss, cand_loss) = results[:, :3].mean(dim=0)
        (tp, fp, fn, correct_mentions,
         total_mentions, correct_relations, total_relations) = results[:, 3:].sum(dim=0)
        max_ranked_accuracy = correct_relations / total_relations if total_relations > 0 else 0
        triple_accuracy = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0
        triple_precision = tp / (tp + fp) if tp + fp > 0 else 0
        triple_recall = tp / (tp + fn) if tp + fn > 0 else 0
        triple_f1 = 2 * triple_precision * triple_recall / (
                    triple_precision + triple_recall) if triple_precision + triple_recall > 0 else 0
        cand_accuracy = correct_mentions / total_mentions if total_mentions > 0 else 0

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_candidate_loss', cand_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_relation_loss', rel_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_candidate_accuracy', cand_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_triple_accuracy', triple_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_triple_precision', triple_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_triple_recall', triple_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_triple_f1', triple_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_max_ranked_accuracy', max_ranked_accuracy, on_step=False, on_epoch=True, prog_bar=True)
