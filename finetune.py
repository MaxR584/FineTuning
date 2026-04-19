from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TRAIN_FILE    = "train.jsonl"
VAL_FILE      = "val.jsonl"
TEST_FILE     = "test.jsonl"
OUTPUT_DIR    = "./deepseek_symptom_extraction"
MAX_SEQ_LEN   = 1024
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

BATCH_SIZE        = 4
GRAD_ACCUM        = 4
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 3
WARMUP_RATIO      = 0.1
WEIGHT_DECAY      = 0.01

print(f"Loading {MODEL_NAME} in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = None,
    load_in_4bit    = True,
)

tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Attaching LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_R,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = LORA_DROPOUT,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} ({100*trainable/total:.2f}% of total)")

print("Loading datasets...")
dataset = load_dataset("json", data_files={
    "train":      TRAIN_FILE,
    "validation": VAL_FILE,
})

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        clean_up_tokenization_spaces = False
    )
    return {"text": text}

dataset = dataset.map(format_chat, remove_columns=["messages"])
print(f"  Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = LEARNING_RATE,
    warmup_ratio                = WARMUP_RATIO,
    weight_decay                = WEIGHT_DECAY,
    lr_scheduler_type           = "cosine",
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 10,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    report_to                   = "none",
    seed                        = 42,
)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = dataset["train"],
    eval_dataset       = dataset["validation"],
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    args               = training_args,
)

print("\nStarting training...")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Epochs:               {NUM_EPOCHS}")
print(f"  LR:                   {LEARNING_RATE}")

trainer_stats = trainer.train()

print(f"\nTraining complete!")
print(f"  Runtime:     {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"  Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"  Final loss:  {trainer_stats.metrics['train_loss']:.4f}")

print(f"\nSaving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Adapter saved.")

print("\nRunning inference on test set for comparison with baseline...")

FastLanguageModel.for_inference(model)

SYSTEM_PROMPT = (
    "You are a clinical NLP assistant specializing in extracting adverse drug "
    "reaction symptoms from patient reviews.\n\n"
    "Given a patient review, extract all symptoms and adverse effects the patient "
    "explicitly mentions experiencing.\n\n"
    "Rules:\n"
    "- Extract only symptoms the patient directly reports experiencing\n"
    "- Do not infer or add symptoms that are not explicitly mentioned\n"
    "- Include informal or colloquial descriptions exactly as they relate to a symptom\n"
    "- Return a JSON array of symptom strings and nothing else\n"
    "- If no symptoms are mentioned, return an empty array []\n\n"
    "Example input: \"i was dizzy and had extreme fatigue after taking the drug\"\n"
    "Example output: [\"dizzy\", \"extreme fatigue\"]"
)

results = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_records = [json.loads(line) for line in f]

print(f"  Test records: {len(test_records)}")

for i, record in enumerate(test_records):
    messages     = record["messages"]
    user_text    = next(m["content"] for m in messages if m["role"] == "user")
    ground_truth = next(m["content"] for m in messages if m["role"] == "assistant")

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = 150,
            temperature    = 0.0,
            do_sample      = False,
            pad_token_id   = tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        spaces_between_special_tokens = True
    ).strip()
    generated = generated.replace("▁", " ").strip()

    results.append({
        "id":           record["messages"][0].get("id", i),
        "input":        user_text,
        "ground_truth": ground_truth,
        "prediction":   generated,
    })

    if i < 5:
        print(f"\n  [Example {i+1}]")
        print(f"  Input:        {user_text[:80]}...")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Prediction:   {generated}")

log_history = trainer.state.log_history
train_steps = [x["step"] for x in log_history if "loss" in x]
train_losses = [x["loss"] for x in log_history if "loss" in x]
val_steps = [x["step"] for x in log_history if "eval_loss" in x]
val_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]

plt.figure(figsize=(10,5))
plt.plot(train_steps, train_losses, label="Train loss")
plt.plot(val_steps, val_losses, label = "Val Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Fine Tuning Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("Loss Curve Saved to loss_curve.png")

results_df = pd.DataFrame(results)
results_df.to_csv("test_predictions.csv", index=False)
print(f"\nAll test predictions saved to test_predictions.csv")
print("Compare the 'prediction' column against your baseline extraction code output.")
