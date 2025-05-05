# Converts csv format to json format
import os
import json
from datasets import load_dataset

input_file = "./email_datasets/df_full_email.csv"
output_file = "./email_datasets/df_full_email.json"

def csv_to_json(csv_file, json_file, instruction=None):
    json_data = []

    with open(csv_file, mode='r', encoding='utf-8') as file:
        lines = file.readlines()

        for line in lines[1:]:  # Skip the header row
            line = line.strip()

            first_comma = line.find(",")
            last_comma = line.rfind(",")

            email = line[first_comma + 1:last_comma]  # "Email" column
            class_field = line[last_comma + 1:]  # "Class" column

            # Remove surrounding quotes if present in instruction field
            if email.startswith('"') and email.endswith('"'):
                email = email[1:-1]

            entry = {
                "instruction": instruction,
                "input": email,  # Email
                "output": class_field,
                "answer": ""  # No mapping from CSV
            }
            json_data.append(entry)

    with open(json_file, mode='w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=4)


if __name__ == "__main__":

    INSTRUCTION_BASE = (
        "You are a cybersecurity expert scanning emails for phishing. "
        "You are provided an email. Determine if it is a phishing email or not. "
        "If it is a phishing email, respond with 'Phishing'. "
        "If it is not a phishing email, respond with 'Legit'. "
    )

    csv_to_json(input_file, output_file, instruction=INSTRUCTION_BASE)


    # Shuffle and Split dataset into finetuning set and evaluation set
    input_file = "email_datasets/df_full_email.json"
    finetune_file = "ft-training_set/finetune.json"
    eval_file = "email_datasets/eval.json"

    # Load dataset from JSON file
    dataset = load_dataset("json", data_files=input_file, split="train")

    # Shuffle dataset
    dataset = dataset.shuffle(seed=42)

    # Split into fine-tuning (80%) and evaluation (20%) sets
    split_ratio = 0.8
    finetune_dataset, eval_dataset = dataset.train_test_split(test_size=1-split_ratio).values()

    # Save datasets as JSON files
    finetune_dataset.to_json(finetune_file)
    eval_dataset.to_json(eval_file)

    print("Fine-tuning and evaluation sets successfully created!")