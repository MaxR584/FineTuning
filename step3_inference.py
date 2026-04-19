"""
STEP 3 — Run Inference with Fine-Tuned Model
=============================================
What this does:
  - Loads the fine-tuned model (base + LoRA adapters)
  - Runs it on your test set (test.jsonl)
  - Saves extracted symptoms to test_predictions.csv
"""

from unsloth import FastLanguageModel
import pandas as pd
import json
import re
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_DIR    = "./deepseek_symptom_extraction"
TEST_FILE      = "test.jsonl"
OUTPUT_CSV     = "test_predictions.csv"
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

    # Decode with spacing fix
    tokens = outputs[0][inputs["input_ids"].shape[1]:]
    token_list = tokenizer.convert_ids_to_tokens(tokens)
    generated = tokenizer.convert_tokens_to_string(token_list).strip()
    match = re.search(r'\[.*?\]', generated, re.DOTALL)
    generated = match.group() if match else "[]"

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


# ── Load test.jsonl ───────────────────────────────────────────────────────────
print(f"Loading {TEST_FILE}...")
test_records = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        test_records.append(json.loads(line))
print(f"  Test records: {len(test_records)}")

# ── Run inference ─────────────────────────────────────────────────────────────
results = []

print("\nRunning inference...")
for i, record in enumerate(tqdm(test_records)):
    messages     = record["messages"]
    user_text    = next(m["content"] for m in messages if m["role"] == "user")
    ground_truth = next(m["content"] for m in messages if m["role"] == "assistant")

    raw_output = extract_symptoms(user_text)
    symptoms   = parse_model_output(raw_output)

    results.append({
        "id":           i,
        "input":        user_text,
        "ground_truth": ground_truth,
        "prediction":   str(symptoms),
        "raw_output":   raw_output,
    })

    if i < 3:
        print(f"\n[Example {i+1}]")
        print(f"  Input:        {user_text[:80]}...")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Prediction:   {str(symptoms)}")

# ── Save results ──────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Saved to {OUTPUT_CSV}")
