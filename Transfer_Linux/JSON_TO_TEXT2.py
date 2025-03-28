import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def parse_plan_to_text(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Assume only one COA exists (e.g. "coa_id_0")
    coa_key = list(data.keys())[0]
    coa_data = data[coa_key]

    overview = coa_data.get("overview", "")
    name = coa_data.get("name", "Untitled Plan")
    tasks = coa_data.get("task_allocation", [])

    # Build a clean summary of plan details.
    plan_details = (
        f"Plan Name: {name}\n"
        f"Overview: {overview}\n"
        f"Number of Tasks: {len(tasks)}\n"
        "Task Details:\n"
    )
    for i, task in enumerate(tasks):
        plan_details += (
            f"Task {i + 1}:\n"
            f"  - Unit ID: {task['unit_id']}\n"
            f"  - Unit Type: {task['unit_type']}\n"
            f"  - Alliance: {task['alliance']}\n"
            f"  - Position: x={task['position']['x']}, y={task['position']['y']}\n"
            f"  - Command: {task['command']}\n"
        )

    # Clear delimiter and explicit instructions.
    instruction = (
        "\n###\n"
        "Based on the above plan details, generate a completely new paragraph containing exactly ten sentences. "
        "Do not repeat or echo any of the plan details above; instead, synthesize them into a fresh, detailed description of "
        "the plan’s objectives, rationale, and key steps. Output only the new paragraph."
    )

    text_prompt = plan_details + instruction
    return text_prompt

def generate_text_from_opt(json_file_path, model_name="facebook/opt-30b"):
    """
    1) Parse JSON to create a prompt with a clear instruction separator.
    2) Load OPT-30B, ensuring pad_token and attention_mask are set.
    3) Generate a fresh, ten-sentence paragraph.
    """
    text_prompt = parse_plan_to_text(json_file_path)

    # Load the tokenizer and the OPT-30B model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use device_map="auto" to distribute the model over available GPUs.
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()

    # Set pad_token to eos_token to avoid warnings.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Get the device from the model's parameters.
    device = next(model.parameters()).device

    encoding = tokenizer(
        text_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Generation configuration to encourage a ten-sentence paragraph.
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=700,
        min_new_tokens=100,
        no_repeat_ngram_size=2
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            pad_token_id=model.config.pad_token_id
        )

    paragraph = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return paragraph

if __name__ == "__main__":
    json_file = "C:/Users/Hayden/Downloads/Test.json"
    model_choice = "facebook/opt-30b"  # Using OPT-30B

    final_paragraph = generate_text_from_opt(json_file, model_name=model_choice)
    print("\n===== GENERATED PARAGRAPH =====\n")
    print(final_paragraph)