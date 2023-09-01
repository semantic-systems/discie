import argparse

from flair.data import Sentence
from flair.nn import Classifier
import spacy
import torch
from tqdm import tqdm

from src.candidate_generation.candidate_generator import get_boundaries

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
import jsonlines
from transformers import AutoTokenizer, AutoModelForTokenClassification



def get_boundaries_bert(dataset, tokenizer, model, device):
    new_examples = []
    for example in tqdm(dataset):
        text = example["input"]
        # Tokenize the text
        tokens = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)

        # Predict labels
        with torch.no_grad():
            outputs = model(tokens.input_ids.to(device))
            predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

            # Get the boundaries of all entities signalized by B and I using the offsets and the predictions, boundaries are tuples of (start_idx_character, end_idx_character), the predictions are of type O, B, I, corresponding to 0, 1, 2
            # Also handle a missing B token at the beginning of an entity
            # Furthermore, the first token is the CLS token, which is not part of the input, so we have to subtract 1 from the start index
            boundaries = []
            current_start = None
            current_end = None
            for i, (offset_start, offset_end) in enumerate(tokens.offset_mapping[0]):
                if predictions[i] == 1 and current_start is None:
                    current_start = offset_start
                    current_end = offset_end
                elif predictions[i] == 1 and current_start is not None:
                    boundaries.append([int(current_start), int(current_end)])
                    current_start = offset_start
                    current_end = offset_end
                elif predictions[i] == 2 and current_start is not None:
                    current_end = offset_end
                elif current_start is not None:
                    boundaries.append([int(current_start), int(current_end)])
                    current_start = None
            if current_start is not None:
                boundaries.append([int(current_start), int(current_end)])
            example["boundaries"] = boundaries
            new_examples.append(example)
    return new_examples


def get_boundaries_spacy(dataset, fine_grained=False):
    new_examples = []
    for example in tqdm(dataset):
        ners = nlp(example["input"]).ents
        boundaries = []
        for ner in ners:
            boundaries.append((ner.start_char, ner.end_char))
        if fine_grained:
            for entity in ners:
                split_tokens = entity.text.split(" ")
                all_sequential_combinations = []
                for i in range(len(split_tokens)):
                    for j in range(i + 1, len(split_tokens) + 1):
                        all_sequential_combinations.append(" ".join(split_tokens[i:j]))
                for comb in all_sequential_combinations:
                    boundaries.append((entity.start_char + entity.text.find(comb),
                                       entity.start_char + entity.text.find(comb) + len(comb)))
        example["boundaries"] = boundaries
        new_examples.append(example)
    return new_examples

def get_boundaries_flair(dataset):
    tagger = Classifier.load('ner')
    new_examples = []
    for example in tqdm(dataset):
        sentence = Sentence(example["input"])
        tagger.predict(sentence)
        boundaries = []
        for ner in sentence.labels:
            boundaries.append((ner.data_point.start_position, ner.data_point.end_position))
        example["boundaries"] = boundaries
        new_examples.append(example)
    return new_examples


def evaluate_boundaries(dataset):
    tp = 0
    fp = 0
    fn = 0
    not_found = []
    for example in dataset:
        try:
            true_boundaries, _ = get_boundaries(example, ignore_boundaries=True)
            found_boundaries = [tuple(x) for x in example["boundaries"]]
            for boundary in found_boundaries:
                if boundary in true_boundaries:
                    tp += 1
                else:
                    fp += 1
            for boundary in true_boundaries:
                if boundary not in found_boundaries:
                    fn += 1
                    not_found.append(example["input"][boundary[0]:boundary[1]])
        except:
            pass
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(not_found)


def main(args, skip_dumping=False):
    dataset_path = args.dataset_path
    dataset_path_without_ending = dataset_path[:dataset_path.rfind(".")]
    with jsonlines.open(dataset_path) as reader:
        dataset = list(reader)[:1000]
    if args.model_checkpoint == "spacy":
        new_dataset = get_boundaries_spacy(dataset)
    elif args.model_checkpoint == "flair":
        new_dataset = get_boundaries_flair(dataset)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
        model = AutoModelForTokenClassification.from_pretrained(args.model_checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        new_dataset = get_boundaries_bert(dataset, tokenizer, model, device)
    evaluate_boundaries(new_dataset)
    if not skip_dumping:
        with jsonlines.open(dataset_path_without_ending + f"_with_boundaries{args.suffix}.jsonl", "w") as writer:
            writer.write_all(new_dataset)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_checkpoint", type=str, default="flair")
    argparser.add_argument("--dataset_path", type=str, default="data/rebel_small/en_test_small_filtered.jsonl")
    argparser.add_argument("--suffix", type=str, default="spacy")
    args = argparser.parse_args()
    main(args)

