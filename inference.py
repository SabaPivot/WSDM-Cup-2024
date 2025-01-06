from data import preprocess
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch

def map_test(test_dataset, tokenizer):
    processed_test = test_dataset.map(lambda x: preprocess(tokenizer, x, train=False))
    return processed_test


def infer(model, tokenizer, test_dataset):
    FastLanguageModel.for_inference(model)
    processed_test_data = map_test(test_dataset=test_dataset, tokenizer=tokenizer)
    tokenized_test, answer = processed_test_data['text'], processed_test_data['winner']

    results = []
    for inputs in tqdm(tokenized_test):
        inputs = torch.tensor(inputs).to('cuda')
        input_length = inputs.shape[-1]
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.3,
        )
        outputs = outputs[:, input_length:]
        results.append(outputs)

    # ToDo: results와 answer 비교