import argparse

import jsonlines
import numpy as np
import wandb
from datasets import load_metric
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer


from src.candidate_generation.candidate_generator import get_boundaries
from src.cie_utils.utils import separate_texts

metric = load_metric("seqeval")
label_names = ["O", "B", "I"]
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


class TokenClassificationDataset(Dataset):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


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

    return new_labels

def get_tokenized_dataset(dataset_path: str, tokenizer):
    all_examples = []
    with jsonlines.open(dataset_path) as reader:
        for example in tqdm(reader):
            try:
                boundaries, _ = get_boundaries(example)
                separated_texts, is_mention = separate_texts([example["input"]], [boundaries])
                separated_texts = separated_texts[0]
                is_mention = is_mention[0]
                tokens = tokenizer(separated_texts, is_split_into_words=True, padding=True, truncation=True)
                labels = [0 if not item else 1 for item in is_mention]
                new_labels = align_labels_with_tokens(labels, tokens.word_ids())
                tokens["labels"] = new_labels
            except ValueError:
                continue

            all_examples.append(tokens)

    return TokenClassificationDataset(all_examples)

def train(model_name: str = "distilbert-base-cased", output_path: str = "bert-finetuned-ner", dataset_path: str = "/data1/moeller/GenIE/data/rebel_small/en_train_small_filtered.jsonl"):
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = get_tokenized_dataset(dataset_path, tokenizer)
    dev_dataset = get_tokenized_dataset("/data1/moeller/GenIE/data/rebel_small/en_val_small_v2_filtered.jsonl", tokenizer)

    args = TrainingArguments(
        output_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=100,
        weight_decay=0.01,
        per_device_train_batch_size=64,
        push_to_hub=False,
        report_to=["wandb"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()

def eval(model_checkpoint: str, dataset_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    dataset = get_tokenized_dataset(dataset_path, tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments("dummy"),
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.evaluate()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="distilbert-base-cased")
    argparser.add_argument("--mode", type=str, default="train")
    argparser.add_argument("--dataset_path", type=str, default="/data1/moeller/GenIE/data/rebel_small/en_train_small_filtered.jsonl")
    argparser.add_argument("--output_path", type=str, default="bert-finetuned-ner")
    args = argparser.parse_args()

    if args.mode == "train":
        wandb.init(project="mention_recognizer_genie", name=args.output_path)
        train(args.model_name, args.output_path, args.dataset_path)
    elif args.mode == "eval":
        eval(args.model_name, args.dataset_path)





