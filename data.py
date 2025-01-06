from datasets import Dataset


def choose_langugae(df, language):
    if not language:
        return df

    df = df[df['language'] == language]
    return df


def pd_to_dataset(df):
    dataset = Dataset.from_pandas(df)

    return dataset


def preprocess(tokenizer, example, train: bool):
    example["winner"] = "0" if example["winner"] == "model_a" else "1"
    conversations = [
        {"role": "system", "content": "Yor are the assistance that tells which answer is better for the user."},
        {"role": "user", "content": f"prompt: {example['prompt']}, model_a: {example['response_a']}, model_b: {example['response_b']}"},
        {"role": "assistant", "content": example["winner"]}
    ]
    if train:
        text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
    else:
        text = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True)

    return {"text": text}


def map_dataset(df, tokenizer, language):
    dataset = pd_to_dataset(choose_langugae(df, language))
    processed_dataset = dataset.map(lambda x: preprocess(tokenizer, x, True))

    train_test_split = processed_dataset.train_test_split(train_size=0.98, seed=42)
    train_dataset, test_dataset = train_test_split["train"], train_test_split["test"]

    return train_dataset, test_dataset
