from data import preprocess
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch


def infer(model, test_dataset):
    FastLanguageModel.for_inference(model)
    tokenized_test, answer = test_dataset['text'], test_dataset['winner']

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

    print(len(results), len(answer))

    # ToDo: results와 answer 비교