from tqdm import tqdm
import json
import os
import torch
import gc

def infer_llm(sample, prompt, model, tokenizer, device):
    email = sample["input"]
    label= sample["output"]
    #EOS_TOKEN = tokenizer.eos_token #I think this is already a part of it...

    prompt = prompt.format(email) #+ EOS_TOKEN
    print("Email:", email)

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the output
    outputs = model.generate(
        inputs["input_ids"],
        max_length=len(inputs["input_ids"][0]) + 5, ### WE SHOULD MESS WITH THIS TO GET THE RIGHT OUTPUT
        num_return_sequences=1,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )

    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output.split("### Response:")[-1].strip()
    print("Response:", response)

    return response

def extract_class(response):
  """
  This function is designed to extract from the response whether or not the LLM decided the email is 'Legit' or 'Phishing'.

  PLEASE EDIT THIS ONE TO SUIT YOUR PROMPT AND OUTPUT FORMAT!!!!!
  THE ONE I'VE PUT IN IS MEGA NAIEVE AND JUST A PLACE HOLDER!!!
  """
  if 'LEGIT' in response.upper():
    return 'Legit'
  elif 'PHISHING' in response.upper():
    return 'Phishing'
  else:
    return 'Unknown'
  pass

def evaluate_samples(samples, prompt, model, tokenizer, device, output_json_path="evaluation_results.json"):
    TP = TN = FP = FN = Unknown = 0
    all_results = []
    predictions = []

    for idx, sample in enumerate(tqdm(samples)):
        predicted_answer = infer_llm(sample, prompt, model, tokenizer, device)
        predicted_class = extract_class(predicted_answer)
        true_class = sample["output"]

        predictions.append(predicted_answer)

        result = {
            "index": idx,
            "input": sample,
            "predicted_answer": predicted_answer,
            "predicted_class": predicted_class,
            "true_class": true_class
        }

        all_results.append(result)

        # Update counters
        if predicted_class == 'Unknown':
            Unknown += 1
        elif predicted_class == 'Legit' and true_class == "Legit":
            TP += 1
        elif predicted_class == 'Legit' and true_class == "Phishing":
            FP += 1
        elif predicted_class == 'Phishing' and true_class == "Legit":
            FN += 1
        elif predicted_class == 'Phishing' and true_class == "Phishing":
            TN += 1

        # Save after each iteration (includes predictions)
        with open(output_json_path, "w") as f:
            json.dump({
                "results": all_results,
                "predictions": predictions,
                "metrics": {
                    "TP": TP,
                    "TN": TN,
                    "FP": FP,
                    "FN": FN,
                    "Unknown": Unknown
                }
            }, f, indent=2)

        torch.cuda.empty_cache()
        gc.collect()

    return predictions, TP, TN, FP, FN, Unknown
