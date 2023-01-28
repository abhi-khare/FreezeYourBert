from datasets import load_dataset
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader

def get_yelp_review_dataset(tokenizer, seed: int):
    dataset = load_dataset("yelp_review_full")

    num_class = 5

    # splitting train_set into train and validation set
    train_valid = dataset["train"].train_test_split(test_size=50000,
                                                    load_from_cache_file=False,
                                                    shuffle=True,
                                                    seed=seed)

    # processing train_set : extraction, padding, truncation, tokenization
    train_set = train_valid['train']
    train_set = train_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                              batched=True,
                              load_from_cache_file=False)
    train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    valid_set = train_valid['test']
    valid_set = valid_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                              batched=True,
                              load_from_cache_file=False)
    valid_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    test_set = dataset["test"]
    test_set = test_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                            batched=True,
                            load_from_cache_file=False)
    test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_set, valid_set, test_set, num_class


def get_trec_dataset(tokenizer, seed: int):
    dataset = load_dataset("trec")
    dataset = dataset.rename_column("coarse_label", "label")
    num_class = 6

    # splitting train_set into train and validation set
    train_valid = dataset["train"].train_test_split(test_size=500,
                                                    load_from_cache_file=False,
                                                    shuffle=True,
                                                    seed=seed)

    # processing train_set : extraction, padding, truncation, tokenization
    train_set = train_valid['train']
    train_set = train_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                              batched=True,
                              load_from_cache_file=False)
    train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    valid_set = train_valid['test']
    valid_set = valid_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                              batched=True,
                              load_from_cache_file=False)
    valid_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    test_set = dataset["test"]
    test_set = test_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                            batched=True,
                            load_from_cache_file=False)
    test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_set, valid_set, test_set, num_class


def get_yahoo_answers_dataset(tokenizer, seed: int):
    dataset = load_dataset("yahoo_answers_topics")
    dataset = dataset.rename_column("topic", "label")
    num_class = 10

    # splitting train_set into train and validation set
    train_valid = dataset["train"].train_test_split(test_size=60000,
                                                    load_from_cache_file=False,
                                                    shuffle=True,
                                                    seed=seed)

    # processing train_set : extraction, padding, truncation, tokenization
    train_set = train_valid['train']
    train_set = train_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                              batched=True,
                              load_from_cache_file=False)
    train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    valid_set = train_valid['test']
    valid_set = valid_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                              batched=True,
                              load_from_cache_file=False)
    valid_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    test_set = dataset["test"]
    test_set = test_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                            batched=True,
                            load_from_cache_file=False)
    test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_set, valid_set, test_set, num_class


def get_tweet_eval_dataset(tokenizer, seed: int):
    dataset = load_dataset("tweet_eval", "sentiment")
    num_class = 3

    # processing train_set : extraction, padding, truncation, tokenization
    train_set = dataset['train']
    train_set = train_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                              batched=True,
                              load_from_cache_file=False)
    train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    valid_set = dataset['validation']
    valid_set = valid_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                              batched=True,
                              load_from_cache_file=False)
    valid_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # processing valid_set : extraction, padding, truncation, tokenization
    test_set = dataset["test"]
    test_set = test_set.map(lambda example: tokenizer(example["text"], padding=True, truncation=True, max_length=128),
                            batched=True,
                            load_from_cache_file=False)
    test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_set, valid_set, test_set, num_class


def get_dataloader(dataset: str, batch_size: int, num_worker: int, seed: int) -> tuple:
    train_set, val_set, test_set, num_class = None, None, None, 0
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    if dataset == 'YR':
        train_set, val_set, test_set, num_class = get_yelp_review_dataset(tokenizer=tokenizer,
                                                                          seed=seed)
    elif dataset == 'YA':
        train_set, val_set, test_set, num_class = get_yahoo_answers_dataset(tokenizer=tokenizer,
                                                                            seed=seed)
    elif dataset == 'TE':
        train_set, val_set, test_set, num_class = get_tweet_eval_dataset(tokenizer=tokenizer,
                                                                         seed=seed)
    elif dataset == 'TREC':
        train_set, val_set, test_set, num_class = get_trec_dataset(tokenizer=tokenizer,
                                                                   seed=seed)

    train = DataLoader(train_set,
                       batch_size=batch_size,
                       num_workers=num_worker,
                       shuffle=True)

    val = DataLoader(val_set,
                     batch_size=batch_size,
                     num_workers=num_worker,
                     shuffle=False)

    test = DataLoader(test_set,
                      batch_size=6 * batch_size,
                      num_workers=num_worker,
                      shuffle=False)

    return train, val, test, num_class
