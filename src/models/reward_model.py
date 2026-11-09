from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoTokenizer
from src.utils import logger, free_memory


def _try_local_first(load_fn, *args, **kwargs):
    """Try loading with local_files_only=True first, fall back to network."""
    try:
        return load_fn(*args, local_files_only=True, **kwargs)
    except (OSError, ValueError):
        logger.info("  Not found in cache, downloading...")
        return load_fn(*args, **kwargs)


class RewardModel:
    """Wrapper for Skywork-Reward-V2 sequence-classification reward models
    (the paper uses Skywork/Skywork-Reward-V2-Qwen3-0.6B)."""

    def __init__(self, model_id: str = "Skywork/Skywork-Reward-V2-Qwen3-0.6B", device: str = "cuda",
                 dtype=torch.bfloat16):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def load(self):
        logger.info(f"Loading reward model {self.model_id}...")
        self.tokenizer = _try_local_first(
            AutoTokenizer.from_pretrained, self.model_id,
        )

        self._load_skywork()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # Disable KV cache — not needed for scoring (full forward pass, not autoregressive)
        self.model.config.use_cache = False
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded reward model on {self.device}")

    def _load_skywork(self):
        """Load Skywork-Reward-V2 via AutoModelForSequenceClassification."""
        from transformers import AutoModelForSequenceClassification
        self.model = _try_local_first(
            AutoModelForSequenceClassification.from_pretrained,
            self.model_id, dtype=self.dtype, num_labels=1,
        )

    def unload(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        free_memory()
        logger.info("Unloaded reward model")

    def _format_input(self, prompt: str, response: str) -> str:
        """Format a prompt+response pair for the reward model (chat template)."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False)
        # Strip duplicate BOS token (Qwen3 tokenizer quirk)
        bos = self.tokenizer.bos_token
        if bos and text.startswith(bos):
            text = text[len(bos):]
        return text

    def score(self, prompt: str, response: str) -> float:
        """Score a single prompt+response pair."""
        text = self._format_input(prompt, response)
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits.squeeze().cpu().item()

    def score_batch(self, texts: list) -> list:
        """Score a batch of formatted prompt+response texts."""
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits.squeeze(-1).cpu().tolist()

    def score_responses(self, prompts: list, responses: list, batch_size: int = 4) -> list:
        """Score multiple prompt+response pairs in batches.

        On CUDA OOM, halves batch_size and retries (down to 1). The reduced
        size applies only to the current call; callers pass the original
        batch_size on subsequent prompts.
        """
        all_scores = []
        current_bs = batch_size
        i = 0
        while i < len(prompts):
            batch_prompts = prompts[i:i + current_bs]
            batch_responses = responses[i:i + current_bs]
            texts = [self._format_input(p, r) for p, r in zip(batch_prompts, batch_responses)]
            try:
                scores = self.score_batch(texts)
                if isinstance(scores, float):
                    scores = [scores]
                all_scores.extend(scores)
                i += current_bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_bs = max(1, current_bs // 2)
                logger.warning(f"RM OOM at batch_size={current_bs}, reducing to {new_bs}")
                current_bs = new_bs
                # retry same position with smaller batch
        # Free cached CUDA memory after scoring to reduce fragmentation
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return all_scores


class MultiGPURewardModel:
    """Shards reward model scoring across multiple GPUs.

    Loads one copy of the model per GPU. score_responses() splits work
    evenly and runs each shard in a separate thread (PyTorch releases
    the GIL during CUDA ops, so true GPU parallelism is achieved).

    Duck-type compatible with RewardModel — exposes the same interface
    (load, unload, score, score_responses, tokenizer, model, device).
    """

    def __init__(self, model_id: str, devices: list, dtype=torch.bfloat16):
        self.model_id = model_id
        self.devices = devices
        self.dtype = dtype
        self.shards = [RewardModel(model_id, device=d, dtype=dtype) for d in devices]

    @property
    def tokenizer(self):
        return self.shards[0].tokenizer

    @property
    def model(self):
        return self.shards[0].model

    @property
    def device(self):
        return self.shards[0].device

    def load(self):
        for rm in self.shards:
            rm.load()
        logger.info(f"Loaded reward model on {len(self.shards)} GPUs: {self.devices}")

    def unload(self):
        for rm in self.shards:
            rm.unload()

    def score(self, prompt: str, response: str) -> float:
        return self.shards[0].score(prompt, response)

    def score_responses(self, prompts: list, responses: list, batch_size: int = 4) -> list:
        """Score across GPUs in parallel. Each GPU gets ~1/N of the items.

        On CUDA OOM in any shard, halves per-shard batch_size (minimum 1)
        and retries the entire batch at the current position. The reduced
        size applies only to this call.
        """
        n = len(prompts)
        n_gpus = len(self.shards)
        if n_gpus == 1 or n <= batch_size:
            return self.shards[0].score_responses(prompts, responses, batch_size)

        # per-shard batch size: each shard processes batch_size items,
        # total items consumed per outer step = batch_size * n_gpus
        per_shard_bs = batch_size
        all_scores = []
        i = 0
        while i < n:
            step_size = per_shard_bs * n_gpus
            step_prompts = prompts[i:i + step_size]
            step_responses = responses[i:i + step_size]
            step_n = len(step_prompts)

            # Split into n_gpus roughly equal chunks
            chunk_size = (step_n + n_gpus - 1) // n_gpus
            chunks = []
            for g in range(n_gpus):
                start = g * chunk_size
                end = min(start + chunk_size, step_n)
                if start < step_n:
                    chunks.append((g, step_prompts[start:end], step_responses[start:end]))

            oom_occurred = [False]

            def _score_chunk(args):
                shard_idx, chunk_prompts, chunk_responses = args
                try:
                    return self.shards[shard_idx].score_responses(
                        chunk_prompts, chunk_responses, per_shard_bs
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    oom_occurred[0] = True
                    return None

            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                results = list(pool.map(_score_chunk, chunks))

            if oom_occurred[0]:
                new_bs = max(1, per_shard_bs // 2)
                logger.warning(f"Multi-GPU RM OOM at per_shard_bs={per_shard_bs}, reducing to {new_bs}")
                per_shard_bs = new_bs
                # retry same position
                continue

            for r in results:
                all_scores.extend(r)
            i += step_n

        return all_scores


def create_reward_model(model_id: str, devices: str, dtype=torch.bfloat16):
    """Factory: create RewardModel or MultiGPURewardModel from device string.

    Args:
        devices: single device ("cpu", "cuda", "cuda:1") or
                 comma-separated for multi-GPU ("cuda:1,cuda:2,cuda:3")
    """
    device_list = [d.strip() for d in devices.split(",")]
    if len(device_list) == 1:
        return RewardModel(model_id, device=device_list[0], dtype=dtype)
    return MultiGPURewardModel(model_id, devices=device_list, dtype=dtype)
