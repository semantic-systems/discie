# README

## Training of entity linker / relation extractor

The main command is:
    
```python src/candidate/generation/candidate_generator.py```


### Arguments

- `--mode`: Choose the mode (choices: TRAIN, INDEX, CANDIDATES, TRAIN_CE) (default: TRAIN).
- `--train_dataset`: Path to the training dataset (default: "data/rebel/en_train.jsonl").
- `--eval_dataset`: Path to the evaluation dataset (default: "data/rebel/en_val.jsonl").
- `--output_path`: Path to the output directory (default: "run_training_bi_encoder").
- `--model_directory`: Specify the directory for the model (default: "models/small").
- `--checkpoint_path`: Specify the checkpoint path (default: None).
- `--candidate_generation_dataset`: Path to the dataset for candidate generation; important for mode CANDIDATES (default: "data/rebel/en_train.jsonl").
- `--training_candidate_set_path`: Path to the training candidate set; important for mode TRAIN_CE (default: "data/rebel/en_train_mapped_candidate_set.json").
- `--eval_candidate_set_path`: Path to the evaluation candidate set; important for mode TRAIN_CE (default: "data/rebel/en_val_mapped_candidate_set.json").
- `--model_name`: Specify the model name (default: "sentence-transformers/all-MiniLM-L12-v2").
- `--batch_size`: Set the batch size for training (default: 128).
- `--num_candidates`: Number of candidates to consider during training of cross-encoder (default: 10).
- `--candidate_weight`: Set the weight of the candidate loss for the cross-encoder (default: 1.0).
- `--normalize`: Enable/disable embeddings normalization (default: True).
- `--exclude_types`: Exclude types in relation extraction (default: False).
- `--types_index_path`: Specify the types index path (default: None).
- `--filter_set_path`: Specify the filter set path (default: None).
- `--type_dictionary_file`: Specify the type dictionary file (default: "data/item_types_relation_extraction_alt.jsonl").


## Training the mention recognizer

Execute the following arguments to train the mention recognizer:

```python src/mention_recognizer/mention_recognizer```


### Arguments

- `--model_name`: Specify the pre-trained model name or path (default: "distilbert-base-cased").
- `--mode`: Set the operational mode (choices: "train," "evaluate," "predict") (default: "train").
- `--dataset_path`: Path to the training dataset (default: "data/rebel/en_train.jsonl").
- `--output_path`: Specify the output directory or path (default: "bert-finetuned-ner").


## Training the bi-encoder

Execute the following arguments to train the bi-encoder:

```python src/candidate/generation/candidate_generator.py --mode TRAIN --train_dataset {train_dataset} --eval_dataset {eval_dataset} ```

Then we create an index for the bi-encoder:

```python src/candidate/generation/candidate_generator.py --mode INDEX --model_directory {model_directory}```
## Training the cross-encoder with relation extraction
To train the cross-encoder, we need initial candidate sets. We can generate them with the following command:

```python src/candidate/generation/candidate_generator.py --mode CANDIDATES --model_directory {model_directory} --candidate_generation_dataset {candidate_generation_dataset}```

This has to be done for the validation and training dataset.

Then we can train the cross-encoder with the following command:

```python src/candidate/generation/candidate_generator.py --mode TRAIN_CE --train_dataset {train_dataset} --eval_dataset {eval_dataset} --training_candidate_set_path {training_candidate_set_path} --eval_candidate_set_path {eval_candidate_set_path} ```

## Training the only the relation extractor
For that, we simply reduce the number of candidates to 0 and eliminate the candidate loss:

```python src/candidate/generation/candidate_generator.py --mode TRAIN_CE --num_candidates 0 --candidate_weight 0.0 --train_dataset {train_dataset} --eval_dataset {eval_dataset} --training_candidate_set_path {training_candidate_set_path} --eval_candidate_set_path {eval_candidate_set_path} ```

## Running DISCIE

### Arguments

The script accepts several command-line arguments for configuring its behavior. Here is a list of available arguments and their descriptions:

- `--debug`: Enable debugging mode (default: False).
- `--use_boundaries`: Use provided boundaries instead of doing mention recognition (default: False).
- `--include_mention_scores`: Include mention scores into the combined scores (default: False).
- `--include_property_scores`: Include property scores into the combined scores (default: False).
- `--alternative_relation_extractor`: Use an alternative relation extractor (default: False).
- `--alternative_relation_extractor_use_types`: Use types with the alternative relation extractor (default: False).
- `--alternative_relation_extractor_deactivate_text`: Deactivate text with the alternative relation extractor (default: False).
- `--disambiguation_mode`: Set the disambiguation mode (choices: SIMPLE, ...) (default: SIMPLE).
- `--dataset_path`: Specify the dataset path (default: "data/rebel_small/en_val_small_v2_filtered.jsonl").
- `--bi_encoder_path`: Specify the path to the bi-encoder model (default: "models/run_training_bi_encoder_new").
- `--mention_recognizer_path`: Specify the path to the mention recognizer model (default: "models/mention_recognizer_2023-07-22_18-10-13/model-epoch=06-val_f1=0.85_val_f1.ckpt").
- `--crossencoder_path`: Specify the path to the crossencoder model (default: "models/crossencoder_checkpoints/model-epoch=13-val_triple_f1=0.85_triple_f1.ckpt").
- `--relation_extractor_path`: Specify the path to a separate relation extractor model (default: "models/cross_encoder_2023-07-26_16-30-38/model-epoch=25-val_triple_f1=0.90_triple_f1.ckpt").
- `--entity_restrictions`: Specify entity restrictions (default: None). Necessary when evaluating on restricted datasets.
- `--property_restrictions`: Specify property restrictions (default: None). Necessary when evaluating on restricted datasets.
- `--mention_threshold`: Set the mention threshold (default: 0.5).
- `--property_threshold`: Set the property threshold (default: 0.5).
- `--combined_threshold`: Set the combined threshold (default: 0.5).
- `--num_candidates`: Specify the number of candidates (default: 10).
- `--mode`: Set the evaluation mode (choices: ET, E) (default: ET). ET evaluates for several thresholds, E only for the specified thresholds.

You can customize the script's behavior by providing these command-line arguments when running the script.




```python src/discriminative_cie/discriminative_cie.py```




