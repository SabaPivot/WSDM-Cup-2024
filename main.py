from model import load_peft_model
import os, yaml
import pandas as pd
from unsloth.chat_templates import get_chat_template
from data import map_dataset
from trainer import set_trainer

os.environ["WANDB_PROJECT"] = "judge"
with open("trainer.yaml", "r") as file:
    config = yaml.safe_load(file)

def main():
    model_name, train_path, language  = config["model"]["model_path"], config["dataset"]["train_path"], config["dataset"]["language"]
    model, tokenizer = load_peft_model(model_name)
    df = pd.read_parquet(train_path)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama"
    )

    train_dataset, test_dataset = map_dataset(df, tokenizer, language)

    trainer = set_trainer(model, tokenizer, train_dataset)
    trainer.train()

if __name__ == "__main__":
    main()
