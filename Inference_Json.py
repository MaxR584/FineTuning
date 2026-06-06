from unsloth import FastLanguageModel
import pandas as pd
import json
import re
from tqdm import tqdm


ADAPTER_DIR = "./qwen_symptom_extraction"
MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 256
BATCH_SIZE = 16

DATASETS = [
    {
        "input": "covid_pushshift_posts_20260521_115353_checkpoint - Copy.json",
        "output": "longcovid_pushshift_1_finetuned_qwen_predictions.csv",
        "name": "long"
    }
]
#─ Load model ───────────────────────────────────────────────────────────
print("Loading fine-tuned model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = ADAPTER_DIR,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)
print("Model loaded.\n")
#─ Helpers ───────────────────────────────────────────────────────────
def extract_symptoms(body_text):
    messages = [
        {"role": "user", "content": str(body_text).strip()}
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    tokens = outputs[0][inputs["input_ids"].shape[1]:]
    token_list = tokenizer.convert_ids_to_tokens(tokens)
    generated = tokenizer.convert_tokens_to_string(token_list).strip()
    match = re.search(r'\[.*?\]', generated, re.DOTALL)
    return match.group() if match else "[]"

def parse_model_output(raw_output):
    if not raw_output:
        return []
    try:
        result = json.loads(raw_output)
        if isinstance(result, list):
            return result
    except:
        pass
    match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except:
            pass
    return []
#─ Run Inference ───────────────────────────────────────────────────────────
for dataset in DATASETS:
    print(f"Processing {dataset['name']}...")
    with open(dataset["input"], 'r', encoding="utf-8") as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data)
    print(f"  Rows: {len(df)}")

    bodies = df["body"].fillna("").astype(str).tolist()

    extracted = []
    raw_outputs = []

    for start in tqdm(range(0, len(bodies), BATCH_SIZE)):
        batch = bodies[start:start + BATCH_SIZE]

        batch_outputs = extract_symptoms(batch)

        for raw in batch_outputs:
            raw_outputs.append(raw)
            extracted.append(json.dumps(parse_model_output(raw)))

    df.loc[:len(extracted)-1, "finetuned_qwen_raw_output"] = raw_outputs
    df.loc[:len(extracted)-1, "finetuned_qwen_extracted_symptoms"] = extracted
    #df["finetuned_qwen_extracted_symptoms"] = extracted
    #df["finetuned_qwen_raw_output"] = raw_outputs

    df.to_csv(dataset["output"], index=False)

    print(f" Saved to {dataset['output']}\n")

    print("All done")
    print("Output files:")
    for dataset in DATASETS:
        print(f" - {dataset['output']}")



    '''
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Skip empty or NaN body text
        if pd.isna(row["body"]) or str(row["body"]).strip() == "":
            extracted.append("[]")
            raw_outputs.append("[]")
            continue

        raw = extract_symptoms(row["body"])
        symptoms = parse_model_output(raw)
        extracted.append(json.dumps(symptoms))
        raw_outputs.append(raw)

    df["finetuned_qwen_extracted_symptoms"] = extracted
    df["finetuned_qwen_raw_output"] = raw_outputs

    df.to_csv(dataset["output"], index=False)
    print(f"  ✓ Saved to {dataset['output']}\n")

print("All done!")
print("Output files:")
for dataset in DATASETS:
    print(f"  - {dataset['output']}")
'''
