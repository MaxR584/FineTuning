"""
STEP 3 — Run Inference with Fine-Tuned Qwen Model
==================================================
Reads Normalized_Testing_data.csv (same data used for all baseline models),
runs the fine-tuned model on the 'body' column, and saves predictions to
finetuned_qwen_predictions.csv for normalization and evaluation.
"""

from unsloth import FastLanguageModel
import pandas as pd
import json
import re
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_DIR    = "./qwen_symptom_extraction"
INPUT_CSV      = "Normalized_Testing_data.csv"
OUTPUT_CSV     = "finetuned_qwen_predictions.csv"
MAX_SEQ_LEN    = 1024
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = """You are a clinical NLP assistant specializing in extracting adverse drug reaction symptoms from patient reviews.

Given a patient review, extract all symptoms and adverse effects the patient explicitly mentions experiencing.

Rules:
- Extract only symptoms the patient directly reports experiencing
- Do not infer or add symptoms that are not explicitly mentioned
- Include informal or colloquial descriptions exactly as they relate to a symptom
- Return a JSON array of symptom strings and nothing else
- If no symptoms are mentioned, return an empty array []

Example input: "i was dizzy and had extreme fatigue after taking the drug"
Example output: ["dizzy", "extreme fatigue"]"""

# ── Load fine-tuned model ─────────────────────────────────────────────────────
print("Loading fine-tuned model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = ADAPTER_DIR,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)
print("Model loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_symptoms(body_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": str(body_text).strip()},
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
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = 0.1,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # Decode and extract first JSON array only
    tokens     = outputs[0][inputs["input_ids"].shape[1]:]
    token_list = tokenizer.convert_ids_to_tokens(tokens)
    generated  = tokenizer.convert_tokens_to_string(token_list).strip()
    match      = re.search(r'\[.*?\]', generated, re.DOTALL)
    generated  = match.group() if match else "[]"

    return generated


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


# ── Load testing data ─────────────────────────────────────────────────────────
print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
print(f"  Rows: {len(df)}")

# ── Run inference ─────────────────────────────────────────────────────────────
results = []

print("\nRunning inference...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    raw_output = extract_symptoms(row["body"])
    symptoms   = parse_model_output(raw_output)

    results.append({
        "id":                                row["id"],
        "body":                              row["body"],
        "finetuned_qwen_extracted_symptoms": json.dumps(symptoms),
        "raw_output":                        raw_output,
    })

# Show first 3 examples
print("\nFirst 3 examples:")
for r in results[:3]:
    print(f"\n  ID:         {r['id']}")
    print(f"  Input:      {r['body'][:80]}...")
    print(f"  Prediction: {r['finetuned_qwen_extracted_symptoms']}")

# ── Save results ──────────────────────────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Saved to {OUTPUT_CSV}")
print("Next step: run normalize_improved.py on this file, then evaluate_models.py")
