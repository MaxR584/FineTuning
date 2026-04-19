"""
STEP 3 — Run Inference with Fine-Tuned Model
=============================================
What this does:
  - Loads the fine-tuned Qwen model (base + LoRA adapters)
  - Runs it on your test set (the 1098-row CSV)
  - Saves extracted symptoms to a CSV in the same format as your
    existing model outputs, so you can plug it straight into
    evaluate_models.py

Why keep normalization separate?
  The model's job is just symptom extraction (text → symptom strings).
  SNOMED normalization is still handled by your normalize script afterward.
  This keeps the two steps clean and independently improvable.
"""

from unsloth import FastLanguageModel
import pandas as pd
import json
import ast
import re
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_DIR   = "./deepseek_symptom_extraction"   # where you saved the fine-tuned adapter
TEST_CSV      = "Normalized_Testing_data.csv" # your test set — never trained on this
OUTPUT_CSV    = "finetuned_deepseek_extracted.csv"
MAX_SEQ_LEN   = 1024
MAX_NEW_TOKENS = 256   # symptoms list is always short, 256 is plenty

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

# Switch to inference mode — disables dropout, enables optimized kernels
FastLanguageModel.for_inference(model)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_symptoms(body_text):
    """
    Run one patient review through the model and return extracted symptoms.
    
    We format the input using the same chat template as training so the
    model sees exactly the format it was trained on.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": str(body_text).strip()},
    ]

    # Apply chat template — add_generation_prompt=True adds the assistant
    # turn opener so the model knows to start generating
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens  = MAX_NEW_TOKENS,
            temperature     = 0.1,   # low temperature = more deterministic output
            do_sample       = True,
            pad_token_id    = tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the input prompt)
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    return generated

def fix_spacing(text):
    import re
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    return text

def parse_model_output(raw_output):
    if not raw_output:
        return []
    try:
        result = json.loads(raw_output)
        if isinstance(result, list):
            return [fix_spacing(s) for s in result]
    except:
        pass
    match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [fix_spacing(s) for s in result]
        except:
            pass
    return []
# ── Run inference on test set ─────────────────────────────────────────────────
print("Loading test set...")
df = pd.read_csv(TEST_CSV)
print(f"  Rows: {len(df)}")

extracted_symptoms = []
extracted_raw      = []

print("\nRunning inference...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    raw_output = extract_symptoms(row["body"])
    symptoms   = parse_model_output(raw_output)
    extracted_symptoms.append(str(symptoms))
    extracted_raw.append(raw_output)

df["finetuned_qwen_extracted_symptoms"] = extracted_symptoms
df["finetuned_qwen_raw_output"]         = extracted_raw

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Saved to {OUTPUT_CSV}")
print("Next step: run normalize_improved.py on this file, then evaluate_models.py")
