import re
import os
import gc
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig, GenerationConfig
from utils.inference import evaluate_samples
import argparse

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["WANDB_MODE"] = "disabled"


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="meta-llama/Meta-Llama-3-8B", type=str, help='LLM model name')  
parser.add_argument('--use_finetuned', default=False, type=bool, help='if use finetuned model or not')
parser.add_argument('--augment', default=True, type=bool, help='if to augment or not')
parser.add_argument('--use_few_shot', default=True, type=bool, help='if to use few shot or not')
parser.add_argument('--use_chain_of_thought', default=False, type=bool, help='if to use chain of thought or not')
args = parser.parse_args()


EVAL_FILE = "email_datasets/eval.json"

eval_dataset = json.load(open(EVAL_FILE, "r"))

# get hf_token after generating it from huggingface.co
# and set it as an environment variable
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config)

if args.use_finetuned:
    pretrained_weights_path = "./trained_models/Meta-Llama-3-8B-finetuned"

    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(
        model,
        pretrained_weights_path,
        torch_dtype=torch.float16,
        device_map={"":0},
        quantization_config=bnb_config
    )

# Device check
device = model.device

if args.use_few_shot:
    few_shot_prompt = """
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction: You are a cybersecurity expert scanning emails for phishing. You will be provided an email below. Decide if the email is Phishing or Legit.
    Please find below some examples of emails and their responses in <example> </example>.

    <example>
    ### Email:
    CONFIDENTAL BUSINESS PROPOSALFROM DRPAT ODOGUDEAR SIRI AM DR PAT ODOGU ACCOUNTANT (FOREIGN PAYMENT DEPT) WITH THE FEDERAL MINISTRY OF TRANSPORT LAGOS NIGERIAWITH THE ASSISTANCE OF SOME SENIOR OFFICALS OF THE CENTRAL BANK OF NIGERIA,AND THE MINISTRY OF FINANCE, WE WANT TO TRANSFER THE SOME OF THIRTYFIVE MILLION US DOLLARS OUT OF MY COUNTRY, NIGERIA THE THIRTYFIVE MILLION US DOLLARS IS AN ACCUMULATION OF OVERINVOICED CONTRACTS WHICH HAS ALREADY BEEN EXECUTED AND COMMISSIONED THIS AMOUNT STILL LIES IN THE FEDERAL MINISTRY OF TRANSPORT SUSPENCE ACCOUNT WITH THE CENTRAL BANK OF NIGERIA(CBN)AS CIVIL SERVANTS WE CANNOT OPERATE A FOREIGN ACCOUNT BECAUSE THE CODE OF CONDUCT ACT IN NIGERIA MAKES IT AN OFFENCE FOR ANY PUBLIC OFFICER TO OPERATE FOREIGN ACCOUNT IT IS AS A RESULT OF THIS THAT WE SOLICIT YOUR ASSISTANCE TO MAKE USE OF YOUR PRIVATE/ COMPANYS ACCOUNT TO TRANSFER THE SAID SUMPLEASE NOTE THAT ALL MODALITIES HAS BEEN WORKED OUT FOR A SMOOTH AND HITCHFREE TRANSFER OF THE US$35MILLION INTO YOUR ACCOUNT, WITHIN TEN WORKING DAYS OF GETTING YOUR POSITIVE RESPONSE AND CONSENTYOU WOULD BE ENTITLED TO 30% OF THE $35,000,000:00 FOR PROVIDING US AN ACCOUNT, 65% WOULD COME TO US IN NIGERIA, AND THE REMAINING 5% WOULD BE USED TO OFFSET ALL LOCAL/FOREIGN EXPENDITURE ON THE FOLLOWING GROUNDS(A) THAT WE ARE SATISFIED ON ALL GROUNDS THAT OUR SHARE OF THEFUND WOULD BE GIVEN TO US AFTER TRANSFERENCE(B) THAT THIS TRANSACTION IS TREATED WITH UTMOSTCONFIDENCE,SECRECY AND ABSOLUTE SINCERITY, WHICH IT DEMANDSIF YOU ARE INTERESTED IN THE PROPOSAL, YOU CAN CONTACT ME THROUGH MYE MAIL ADDRESSES:pat11@weedmailcom OR TEL NUMBER:803 305 6016 TO ENABLE US DISCUSS FURTHER DETAILS ON THE TRANSACTION YOU ARE ALSO REQUIRED TO PROVIDE A SECURED TELEPHONE AND FAX FOR THE PURPOSE OF THIS TRANSACTIONEXPECTING TO HEAR FROM YOU SOONESTYOURS FAITHFULLY DR PAT ODOGUYo Yo's shall return, they always do Free YoYo Giveaway, Click http://TheYocomExpress yourself with a super cool email address from BigMailBoxcomHundreds of choices It's freehttp://wwwbigmailboxcom

    ### Response: Phishing
    </example>

    <example>
    ### Email:
    I am traveling overseas and only have sporadic access to my email If you need immediate assistance please call StateDepartment Operations at 2026471512 and ask for Nora Toiv who can assist youcdm

    ### Response: Legit
    </example>

    <example>
    ### Email:
    ""Files Portal monkeyorg "" You Have Recieved New Files monkeyorg File Notification Hello jose, You have new files shared with you on the monkeyorg files portal ( https://binliztwebapp/adobe/downld/indexhtmljose@monkeyorg ) View ( https://binliztwebapp/adobe/downld/indexhtmljose@monkeyorg ) | Download ( https://binliztwebapp/adobe/downld/indexhtmljose@monkeyorg ) These files will expire if not accessed by 10th August, 2021 Thanks filesportal@monkeyorg monkeyorg File Portal monkeyorg File Notification Hello jose, You have new files shared with you on the monkeyorg files portal View | DownloadThese files will expire if not accessed by 10th August, 2021 Thanksfilesportal@monkeyorg monkeyorg File Portal

    ### Response: Phishing
    </example>

    ### Email:
    {}

    ### Response: """

    predictions, TP, TN, FP, FN, unknown = evaluate_samples(eval_dataset, few_shot_prompt, model, tokenizer, device, output_json_path="evaluation_results.json")

if args.use_chain_of_thought:
    chain_of_thought = '''
    ### Instruction:
    You are a cybersecurity expert tasked with scanning emails for signs of phishing. When provided with an email, follow this detailed step-by-step analysis:

    1. Address Analysis:
    Examine the mentioned email address and domain. Determine if it matches known legitimate sources or if it appears spoofed or suspicious.
    2. Content Examination:
    Analyze the language and tone of the email. Look for urgent requests, threatening language, or unusual phrasing that is common in phishing attempts.
    3. Hyperlink & Attachment Inspection:
    Identify any hyperlinks or attachments within the email. Evaluate the URLs for mismatched or misleading domains, and check if the attachments are unexpected or potentially harmful.
    4. Request Verification:
    Check if the email requests sensitive information such as passwords, financial details, or personal data. Assess whether these requests align with standard practices from the supposed sender.
    5. Overall Consistency:
    Evaluate the overall consistency of the email including formatting, grammar, and content coherence.
    6. Final Decision:
    Based on your analysis in steps 1-5, decide if the email is a phishing attempt or a legitimate message. If you're unsure, then it is safer to conclude the email is 'Phishing'.

    After following these steps, provide your final answer. If you conclude the email is a phishing attempt, respond with 'Phishing'. If it is a legitimate email, respond with 'Legit'.

    ### Email:
    {}

    ### Response:
    '''
    predictions, TP, TN, FP, FN, unknown = evaluate_samples(eval_dataset, chain_of_thought, model, tokenizer, device, output_json_path="evaluation_results.json")


print("True Positives TP: ", TP)
print("True Negatives TN: ", TN)
print("False Positives FP: ", FP)
print("False Negatives FN: ", FN)
print("Number of unknown classes: ", unknown)