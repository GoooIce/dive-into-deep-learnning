from datasets import load_dataset

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from bitnet import replace_linears_in_hf

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

dataset = load_dataset("yelp_review_full")


# Load a model from Hugging Face's Transformers
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Replace Linear layers with BitLinear
replace_linears_in_hf(model)

# model.to("cuda")
# model.to("cpu")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# training_args = TrainingArguments(output_dir="test_trainer")
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", use_cpu=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()