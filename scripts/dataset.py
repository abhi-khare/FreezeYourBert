from datasets import load_dataset
from transformers import AutoTokenizer


def get_yelp_review_dataset(tokenizer):
    dataset = load_dataset("yelp_review_full")

    # splitting train_set into train and validation set
    train_valid = dataset["train"].train_test_split(test_size=50000)

    # processing train_set : extraction, padding, truncation, tokenization
    train_set = train_valid['train']
    train_set = train_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    valid_set = train_valid['test']
    valid_set = valid_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    valid_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    test_set = dataset["test"]
    test_set = test_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_set, valid_set, test_set


def get_trec_dataset(tokenizer):
    dataset = load_dataset("trec")
    dataset = dataset.rename_column("coarse_label", "label")

    # splitting train_set into train and validation set
    train_valid = dataset["train"].train_test_split(test_size=500)

    # processing train_set : extraction, padding, truncation, tokenization
    train_set = train_valid['train']
    train_set = train_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    valid_set = train_valid['test']
    valid_set = valid_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    valid_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    test_set = dataset["test"]
    test_set = test_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_set, valid_set, test_set


def get_yahoo_answers_dataset(tokenizer):
    dataset = load_dataset("yahoo_answers_topics")
    dataset = dataset.rename_column("topic", "label")

    # splitting train_set into train and validation set
    train_valid = dataset["train"].train_test_split(test_size=60000)

    # processing train_set : extraction, padding, truncation, tokenization
    train_set = train_valid['train']
    train_set = train_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    valid_set = train_valid['test']
    valid_set = valid_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    valid_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    test_set = dataset["test"]
    test_set = test_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_set, valid_set, test_set


def get_tweet_eval_dataset(tokenizer):
    dataset = load_dataset("tweet_eval", "sentiment")

    # processing train_set : extraction, padding, truncation, tokenization
    train_set = dataset['train']
    train_set = train_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    valid_set = dataset['validation']
    valid_set = valid_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    valid_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    test_set = dataset["test"]
    test_set = test_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True), batched=True)
    test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_set, valid_set, test_set
