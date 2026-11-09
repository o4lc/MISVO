import json
import os
from datasets import load_dataset


def load_shp_test(max_samples: int = 1000, seed: int = 42) -> list:
    """Load SHP test dataset from HuggingFace (stanfordnlp/SHP)."""
    dataset = load_dataset("stanfordnlp/SHP", split="test")
    dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))
    return [example["history"] for example in dataset]


def load_mbppplus_test(data_path=None, max_samples=378):
    """Load MBPP+ dataset.

    Returns list of dicts (not raw strings) because test metadata
    must travel with prompts for scoring.

    Each dict: {task_id, prompt, test_list, test_imports, code}
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "mbpp_plus", "mbppplus.jsonl")

    entries = []
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            entries.append({
                "task_id": entry["task_id"],
                "prompt": entry["prompt"],
                "test_list": entry["test_list"],
                "test_imports": entry.get("test_imports", []),
                "code": entry.get("code", ""),
            })
            if len(entries) >= max_samples:
                break
    return entries


def get_prompt_text(prompt):
    """Extract the text string from a prompt (may be str or chat-template dict)."""
    if isinstance(prompt, dict) and "text" in prompt:
        return prompt["text"]
    return prompt


def _uses_chat_template(model_name: str) -> bool:
    """Check if a model should use chat template encoding."""
    return ("instruct" in model_name
            or model_name.startswith("lfm")
            or model_name.endswith("_it"))


def format_prompt(prompt, model_name: str, dataset_name: str):
    """Format prompt for the given model following the paper's conventions.

    For chat-template models (instruct / LFM / *_it), SHP prompts are returned
    as a dict with 'messages' (for apply_chat_template) and 'text' (for scoring).
    MBPP+ always uses the EvalPlus plain-text instruct format.
    """
    if dataset_name == "mbpp_plus":
        # EvalPlus instruct prompt format (evalplus/provider/utility.py).
        if isinstance(prompt, dict):
            desc = prompt["prompt"].strip()
            test_list = prompt.get("test_list", [])
            first_assertion = test_list[0] if test_list else ""
            task_prompt = f'"""\n{desc}\n{first_assertion}\n"""'
        else:
            task_prompt = prompt
        instruction = (
            "Please provide a self-contained Python script that solves the "
            "following problem in a markdown code block:\n"
            f"```\n{task_prompt.strip()}\n```"
        )
        response_prefix = (
            "Below is a Python script with a self-contained function that "
            "solves the problem and passes corresponding tests:\n"
            "```python\n"
        )
        return f"User: {instruction}\nAssistant: {response_prefix}"
    elif dataset_name == "shp":
        if _uses_chat_template(model_name):
            return {
                "messages": [{"role": "user", "content": prompt}],
                "text": prompt,
            }
        return f"User: {prompt}\nAssistant:"
    return prompt


def load_dataset_prompts(dataset_name: str, max_samples: int = 1000, seed: int = 42) -> list:
    """Load and return raw prompts for the given dataset."""
    if dataset_name == "shp":
        return load_shp_test(max_samples=max_samples, seed=seed)
    elif dataset_name == "mbpp_plus":
        return load_mbppplus_test(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
