from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import yaml

with open("trainer.yaml", "r") as file:
    config = yaml.safe_load(file)
config["training_args"]["bf16"] = is_bfloat16_supported()
config["training_args"]["fp16"] = not is_bfloat16_supported()

def set_trainer(model, tokenizer, train_data):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_data,
        dataset_text_field = "text",
        max_seq_length = config["dataset"]["max_seq_length"],
        dataset_num_proc=config["dataset"]["dataset_num_proc"],
        args=TrainingArguments(**config["training_args"])
        )
    return trainer