import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import logger, free_memory


def _try_local_first(load_fn, *args, **kwargs):
    """Try loading with local_files_only=True first, fall back to network."""
    try:
        return load_fn(*args, local_files_only=True, **kwargs)
    except (OSError, ValueError):
        logger.info("  Not found in cache, downloading...")
        return load_fn(*args, **kwargs)


MODEL_REGISTRY = {
    "gemma3_4b_it": "google/gemma-3-4b-it",
    "llama3_8b_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "lfm25_1b": "LiquidAI/LFM2.5-1.2B-Instruct",
    "phi4": "microsoft/phi-4",
}


# Hidden dimensions for pre-logit space
HIDDEN_DIMS = {
    "gemma3_4b_it": 2560,
    "llama3_8b_instruct": 4096,
    "lfm25_1b": 2048,
    "phi4": 5120,
}


def _load_gemma3(model_id: str, dtype, attn_implementation: str = "sdpa"):
    """Load Gemma3 model, handling the multimodal architecture.

    Gemma3 on HuggingFace is Gemma3ForConditionalGeneration (multimodal).
    We extract the language_model (Gemma3ForCausalLM) for text-only use.
    Structure: full_model.language_model.model (Gemma3TextModel with .norm)
               full_model.language_model.lm_head

    `attn_implementation` is forwarded to HF; "sdpa" (default) goes through
    fused attention kernels. Set to "eager" to work around the HF Gemma3
    hybrid-attention shape-mismatch bug on prompts longer than sliding_window.
    """
    logger.info(f"Loading Gemma3 with attn_implementation={attn_implementation!r}")
    # Try direct CausalLM loading first (works if transformers maps it correctly)
    try:
        model = _try_local_first(
            AutoModelForCausalLM.from_pretrained,
            model_id, dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )
        logger.info("Loaded Gemma3 via AutoModelForCausalLM")
        return model
    except (ValueError, KeyError, OSError):
        pass

    # Fall back to loading the full multimodal model and extracting text component
    logger.info("Loading Gemma3 multimodal model and extracting language model...")
    from transformers import AutoModel
    full_model = _try_local_first(
        AutoModel.from_pretrained,
        model_id, dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    if hasattr(full_model, "language_model"):
        model = full_model.language_model
        # Detach from parent so vision components can be freed
        full_model.language_model = None
        del full_model
        gc.collect()
        return model

    raise RuntimeError(
        f"Cannot extract text model from {model_id}. "
        "Ensure transformers>=4.49.0 is installed."
    )


class BaseLLM:
    """Wrapper around a causal LM with pre-logit hook support for AISP."""

    def __init__(self, model_name: str, device: str = "cuda", dtype=None,
                 attn_implementation: str = None):
        self.model_name = model_name
        self.device = device
        # attn_implementation: "sdpa" | "eager" | "flash_attention_2" | None.
        # None lets each loader pick its default. "eager" is a known workaround
        # for the Gemma3 hybrid-attention shape-mismatch bug on prompts longer
        # than sliding_window (~1024 for gemma-3-4b).
        self.attn_implementation = attn_implementation
        self.enable_thinking = None  # Set from config; passed to apply_chat_template
        # Gemma3 requires bfloat16 for numerical stability
        if dtype is None:
            needs_bf16 = "gemma" in model_name
            self.dtype = torch.bfloat16 if needs_bf16 else torch.float16
        else:
            self.dtype = dtype
        self.model_id = MODEL_REGISTRY[model_name]
        self.hidden_dim = HIDDEN_DIMS[model_name]
        self.model = None
        self.tokenizer = None
        # Reference to the backbone (the part before lm_head)
        self._backbone = None
        self._lm_head = None
        self._final_norm = None  # final norm layer (hook target for perturbation)

    def load(self):
        logger.info(f"Loading {self.model_id} in {self.dtype}...")

        # Enable TF32 for ~2-3x matmul speedup on A100 (negligible precision loss)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = _try_local_first(
            AutoTokenizer.from_pretrained,
            self.model_id, padding_side="left",
        )

        # Verify the tokenizer can actually encode text.  Multimodal models
        # (e.g. Qwen3.5) sometimes ship a tokenizer that AutoTokenizer loads
        # but that produces empty output; fall back to the processor's
        # inner tokenizer in that case.
        if len(self.tokenizer.encode("test")) == 0:
            logger.warning("AutoTokenizer produces empty output; "
                           "falling back to AutoProcessor.tokenizer")
            from transformers import AutoProcessor
            processor = _try_local_first(
                AutoProcessor.from_pretrained, self.model_id,
            )
            self.tokenizer = processor.tokenizer
            self.tokenizer.padding_side = "left"

        if self.tokenizer.model_max_length < 256:
            logger.info(f"  Fixing model_max_length: {self.tokenizer.model_max_length} -> 32768")
            self.tokenizer.model_max_length = 32768

        if "gemma-3" in self.model_id:
            gemma_attn_impl = self.attn_implementation or "sdpa"
            self.model = _load_gemma3(self.model_id, self.dtype,
                                      attn_implementation=gemma_attn_impl)
        else:
            # Try the configured (or SDPA-default) impl; fall back if unsupported
            preferred_attn = self.attn_implementation or "sdpa"
            try:
                self.model = _try_local_first(
                    AutoModelForCausalLM.from_pretrained,
                    self.model_id,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation=preferred_attn,
                )
            except ValueError:
                logger.info(f"  attn_implementation={preferred_attn!r} not supported; "
                            f"loading with default attention")
                self.model = _try_local_first(
                    AutoModelForCausalLM.from_pretrained,
                    self.model_id,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Resolve backbone (model.model) and lm_head
        self._resolve_components()

        # Place model on device. If the model is too large for a single GPU,
        # fall back to device_map="auto" (CPU offloading).
        try:
            self.model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("  Model too large for single GPU; reloading with device_map='auto'")
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            import tempfile
            offload_dir = tempfile.mkdtemp(prefix="hf_offload_")
            self.model = _try_local_first(
                AutoModelForCausalLM.from_pretrained,
                self.model_id,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                offload_folder=offload_dir,
            )
            self._resolve_components()
            self.device = next(self.model.parameters()).device
        self.model.eval()

        logger.info(f"Loaded {self.model_id} on {self.device} (hidden_dim={self.hidden_dim})")

    def _resolve_components(self):
        """Find the backbone (text model with final norm) and lm_head.

        Sets self._backbone, self._lm_head, and self._final_norm.
        The final norm is the layer right before lm_head (RMSNorm/LayerNorm)
        where we register perturbation hooks.
        """
        model = self.model
        # Standard pattern: model.model (backbone with .norm) + model.lm_head
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            self._backbone = model.model
            self._lm_head = model.lm_head
            self._final_norm = model.model.norm
            return
        # LFM2: model.model (backbone with .embedding_norm) + model.lm_head
        if hasattr(model, "model") and hasattr(model.model, "embedding_norm"):
            self._backbone = model.model
            self._lm_head = model.lm_head
            self._final_norm = model.model.embedding_norm
            return
        # Gemma3 multimodal: model.model.language_model (backbone with .norm) + model.lm_head
        if (hasattr(model, "model") and hasattr(model.model, "language_model")
                and hasattr(model.model.language_model, "norm")):
            self._backbone = model.model.language_model
            self._lm_head = model.lm_head
            self._final_norm = model.model.language_model.norm
            return
        # Some models use model.transformer + model.lm_head
        if hasattr(model, "transformer") and hasattr(model, "lm_head"):
            self._backbone = model.transformer
            self._lm_head = model.lm_head
            # Try common norm names
            self._final_norm = getattr(model.transformer, "norm",
                                       getattr(model.transformer, "ln_f", None))
            if self._final_norm is not None:
                return
        raise ValueError(
            f"Cannot resolve backbone/lm_head for {self.model_name}. "
            f"Model type: {type(model).__name__}"
        )

    def unload(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self._backbone = None
            self._lm_head = None
            self._final_norm = None
        free_memory()
        logger.info(f"Unloaded {self.model_name}")

    def compute_prompt_kv_cache(self, prompt_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Pre-compute KV cache for prompt positions 0..L-2.

        The last prompt token is left unprocessed so that model.generate()
        processes it as its first step, triggering perturbation hooks correctly.

        Returns None for models with sliding window attention (e.g. GPT-OSS)
        where passing a pre-computed cache to generate() causes attention
        mask mismatches.

        Returns:
            past_key_values: DynamicCache object (batch=1), or None
        """
        if prompt_ids.shape[1] < 2:
            return None
        # Models with sliding window attention don't support KV cache reuse
        # through generate() — the internal mask setup doesn't account for
        # cached positions correctly.
        if getattr(self.model.config, "sliding_window", None) is not None:
            return None
        # Mask-aware position_ids so each row's RoPE positions start at 0 on its
        # first REAL token. We call the backbone directly here (bypassing
        # generate()'s own prefill), so without this a LEFT-PADDED batch gets
        # absolute positions [0..L-2] — wrong for padded rows, corrupting their
        # attention proportionally to pad length. This silently degraded every
        # wave-batched (multi-prompt) generation; for a single prompt / all-ones
        # mask it reduces to the default [0,1,2,...] (no change). generate()
        # already computes mask-aware positions for the subsequent decode steps.
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            outputs = self._backbone(
                input_ids=prompt_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                position_ids=position_ids[:, :-1],
                use_cache=True,
            )
        return outputs.past_key_values

    def _expand_kv_cache(self, kv_cache, batch_size: int):
        """Create a batch-expanded copy of a batch=1 KV cache.

        Supports standard DynamicCache (.layers) and Qwen3_5DynamicCache
        (key_cache/value_cache/conv_states/recurrent_states lists).

        Returns None for unsupported cache types.
        """
        # --- Standard DynamicCache with .layers (Llama, Gemma, Qwen3.5, etc.) ---
        if hasattr(kv_cache, "layers"):
            from transformers.cache_utils import DynamicCache, DynamicLayer
            expanded = DynamicCache()
            for layer in kv_cache.layers:
                if hasattr(layer, "keys"):
                    # Standard full-attention layer (DynamicLayer)
                    new_layer = DynamicLayer()
                    new_layer.dtype = layer.dtype
                    new_layer.device = layer.device
                    new_layer.keys = layer.keys.expand(batch_size, -1, -1, -1)
                    new_layer.values = layer.values.expand(batch_size, -1, -1, -1)
                    new_layer.is_initialized = True
                    expanded.layers.append(new_layer)
                elif hasattr(layer, "conv_states"):
                    # Linear attention layer (Qwen3.5 hybrid) — expand
                    # conv_states and recurrent_states along batch dim
                    from transformers.cache_utils import LinearAttentionLayer
                    new_layer = LinearAttentionLayer()
                    if getattr(layer, "is_conv_states_initialized", False):
                        cs = layer.conv_states
                        sizes = [batch_size] + [-1] * (cs.dim() - 1)
                        new_layer.lazy_initialization(
                            conv_states=cs.expand(*sizes).contiguous())
                    if getattr(layer, "is_recurrent_states_initialized", False):
                        rs = layer.recurrent_states
                        sizes = [batch_size] + [-1] * (rs.dim() - 1)
                        new_layer.lazy_initialization(
                            recurrent_states=rs.expand(*sizes).contiguous())
                    expanded.layers.append(new_layer)
                else:
                    # Unknown layer type — skip cache expansion entirely
                    logger.warning("Unknown cache layer type %s; skipping cache",
                                   type(layer).__name__)
                    return None
            return expanded

        logger.warning("KV cache type %s not supported for expansion; "
                       "prompt will be recomputed per sample",
                       type(kv_cache).__name__)
        return None

    def _attn_mask_kwarg(self, attention_mask):
        """Return attention_mask kwarg dict for model.generate().

        Omits all-ones masks (no real padding) to let generate() manage
        masks internally — required for models with sliding-window
        attention (e.g. GPT-OSS) where an explicit mask isn't extended
        correctly during autoregressive generation.

        Exception: when pad_token_id == eos_token_id (e.g. Gemma, where we
        alias pad to eos), generate() cannot infer the mask from inputs and
        warns the user. In that case we always pass the mask explicitly.
        """
        if attention_mask is None:
            return {}
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        eos_set = set()
        if isinstance(eos_id, (list, tuple)):
            eos_set = {e for e in eos_id if e is not None}
        elif eos_id is not None:
            eos_set = {eos_id}
        pad_eq_eos = pad_id is not None and pad_id in eos_set
        if attention_mask.all() and not pad_eq_eos:
            return {}
        return {"attention_mask": attention_mask}

    def generate(self, prompt_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_new_tokens: int = 128) -> torch.Tensor:
        """Standard greedy generation."""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_ids,
                **self._attn_mask_kwarg(attention_mask),
                max_new_tokens=max_new_tokens,
                max_length=None,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return outputs

    def generate_with_perturbation(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        perturbation: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> tuple:
        """Generate token-by-token with perturbation added to pre-logits.

        The pre-logit z_t = phi_LLM(y_{<t}) is the output of the backbone
        (after final RMSNorm, before lm_head). We add v_t to z_t then
        compute logits = lm_head(z_t + v_t) and take argmax (greedy).

        Args:
            prompt_ids: (1, seq_len) input token ids
            attention_mask: (1, seq_len) attention mask
            perturbation: (max_new_tokens, hidden_dim) perturbation vectors v_t
            max_new_tokens: max tokens to generate

        Returns:
            (generated_ids, pre_logits) where pre_logits is (num_generated, hidden_dim)
        """
        backbone = self._backbone
        lm_head = self._lm_head
        device = prompt_ids.device

        generated_ids = prompt_ids.clone()
        cur_attention_mask = attention_mask.clone()
        pre_logits_gpu = []
        past_key_values = None

        for t in range(max_new_tokens):
            with torch.no_grad():
                if past_key_values is None:
                    input_ids = generated_ids
                    attn_mask = cur_attention_mask
                else:
                    input_ids = generated_ids[:, -1:]
                    attn_mask = cur_attention_mask

                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                # last_hidden_state is post-norm (after final RMSNorm)
                hidden = outputs.last_hidden_state
                past_key_values = outputs.past_key_values

                z_t = hidden[:, -1, :]  # (1, hidden_dim)
                pre_logits_gpu.append(z_t.squeeze(0))

                # Add perturbation
                if t < perturbation.shape[0]:
                    v_t = perturbation[t].to(device).to(z_t.dtype).unsqueeze(0)
                    z_tilde = z_t + v_t
                else:
                    z_tilde = z_t

                logits = lm_head(z_tilde)  # (1, vocab_size)
                next_token = logits.argmax(dim=-1)  # (1,)

                generated_ids = torch.cat(
                    [generated_ids, next_token.unsqueeze(0)], dim=-1
                )
                cur_attention_mask = torch.cat(
                    [cur_attention_mask,
                     torch.ones(1, 1, device=device, dtype=cur_attention_mask.dtype)],
                    dim=-1,
                )

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Bulk transfer to CPU
        pre_logits = torch.stack(pre_logits_gpu, dim=0).cpu()  # (num_generated, hidden_dim)
        return generated_ids, pre_logits

    def generate_samples_batch(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_samples: int,
        batch_size: int = 32,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Generate multiple samples, chunked by batch_size.

        Uses manual input expansion instead of num_return_sequences for
        broader model compatibility (e.g. Qwen3.5 hybrid attention).
        """
        all_outputs = []
        for start in range(0, num_samples, batch_size):
            cur_batch = min(batch_size, num_samples - start)
            batch_ids = prompt_ids.expand(cur_batch, -1).contiguous()
            batch_mask = attention_mask.expand(cur_batch, -1).contiguous()
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch_ids,
                    **self._attn_mask_kwarg(batch_mask),
                    max_new_tokens=max_new_tokens,
                    max_length=None,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            all_outputs.append(outputs)
        # Pad to same length — different batches may produce different seq lengths
        max_len = max(o.shape[1] for o in all_outputs)
        padded = []
        for o in all_outputs:
            if o.shape[1] < max_len:
                pad = torch.full((o.shape[0], max_len - o.shape[1]),
                                 self.tokenizer.pad_token_id,
                                 dtype=o.dtype, device=o.device)
                o = torch.cat([o, pad], dim=1)
            padded.append(o)
        return torch.cat(padded, dim=0)

    def generate_with_perturbation_batch(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        perturbations: torch.Tensor,
        max_new_tokens: int = 128,
        prompt_kv_cache=None,
        capture_prelogits: bool = False,
    ):
        """Batched generation with per-sample perturbations on pre-logits.

        Uses a forward hook on the final RMSNorm to inject perturbations,
        then delegates to HuggingFace's generate() for batched decoding.

        Args:
            prompt_ids: (1, seq_len) input token ids
            attention_mask: (1, seq_len) attention mask
            perturbations: (batch_size, max_new_tokens, hidden_dim)
            max_new_tokens: max tokens to generate
            prompt_kv_cache: optional pre-computed KV cache from compute_prompt_kv_cache()
            capture_prelogits: if True, also capture pre-logits (before perturbation)

        Returns:
            output_ids: (batch_size, prompt_len + generated_len)
            If capture_prelogits=True, returns (output_ids, pre_logits) where
            pre_logits is (batch_size, gen_steps, hidden_dim) on CPU.
        """
        batch_size = perturbations.shape[0]
        batch_ids = prompt_ids.expand(batch_size, -1).contiguous()
        batch_mask = attention_mask.expand(batch_size, -1).contiguous()

        step = [0]
        prefill_done = [False]
        captured_prelogits_gpu = [] if capture_prelogits else None

        def perturbation_hook(module, input, output):
            t = step[0]
            if not prefill_done[0]:
                # Prefill: output is (batch, prompt_len, hidden_dim) without cache,
                # or (batch, 1, hidden_dim) with cache. Either way, perturb last position.
                prefill_done[0] = True
                if capture_prelogits:
                    captured_prelogits_gpu.append(output[:, -1, :].detach())
                if t < perturbations.shape[1]:
                    bs = output.shape[0]
                    v = perturbations[:bs, t, :].to(output.device).to(output.dtype)
                    output = output.clone()
                    output[:, -1, :] = output[:, -1, :] + v
                step[0] += 1
                return output
            # Autoregressive steps: output is (batch, 1, hidden_dim)
            if capture_prelogits:
                captured_prelogits_gpu.append(output[:, -1, :].detach())
            if t < perturbations.shape[1]:
                bs = output.shape[0]
                v = perturbations[:bs, t, :].unsqueeze(1).to(output.device).to(output.dtype)
                output = output + v
            step[0] += 1
            return output

        generate_kwargs = {}
        if prompt_kv_cache is not None:
            expanded = self._expand_kv_cache(prompt_kv_cache, batch_size)
            if expanded is not None:
                generate_kwargs["past_key_values"] = expanded

        handle = self._final_norm.register_forward_hook(perturbation_hook)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch_ids,
                    **self._attn_mask_kwarg(batch_mask),
                    max_new_tokens=max_new_tokens,
                    max_length=None,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs,
                )
        finally:
            handle.remove()

        if capture_prelogits:
            pre_logits = torch.stack(captured_prelogits_gpu, dim=1).cpu()
            return outputs, pre_logits

        return outputs

    def generate_with_perturbation_sampling(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        perturbation: torch.Tensor,
        n_samples: int,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        prompt_kv_cache=None,
    ) -> tuple:
        """Batched sampling with shared perturbation, capturing pre-logits.

        Uses a forward hook on the final RMSNorm to inject the perturbation
        and capture pre-logits (before perturbation) at each step, then
        delegates to HuggingFace's generate() for efficient batched decoding.

        Args:
            prompt_ids: (1, seq_len) input token ids
            attention_mask: (1, seq_len) attention mask
            perturbation: (max_new_tokens, hidden_dim) shared across all samples
            n_samples: number of parallel samples
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            prompt_kv_cache: optional pre-computed KV cache from compute_prompt_kv_cache()

        Returns:
            (output_ids, pre_logits)
            - output_ids: (n_samples, prompt_len + gen_len)
            - pre_logits: (n_samples, gen_steps, hidden_dim) on CPU
        """
        batch_ids = prompt_ids.expand(n_samples, -1).contiguous()
        batch_mask = attention_mask.expand(n_samples, -1).contiguous()

        step = [0]
        prefill_done = [False]
        # Accumulate pre-logits on GPU to avoid per-step GPU→CPU sync
        captured_prelogits_gpu = []

        def hook_fn(module, input, output):
            t = step[0]
            if not prefill_done[0]:
                prefill_done[0] = True
                # Capture pre-perturbation z_t at last position (stay on GPU)
                captured_prelogits_gpu.append(output[:, -1, :].detach())
                if t < perturbation.shape[0]:
                    v = perturbation[t].to(output.device, dtype=output.dtype)
                    output = output.clone()
                    output[:, -1, :] = output[:, -1, :] + v
                step[0] += 1
                return output
            # Autoregressive: output is (batch, 1, hidden_dim)
            captured_prelogits_gpu.append(output[:, -1, :].detach())
            if t < perturbation.shape[0]:
                v = perturbation[t].to(output.device, dtype=output.dtype)
                output = output + v.view(1, 1, -1)
            step[0] += 1
            return output

        generate_kwargs = {}
        if prompt_kv_cache is not None:
            expanded = self._expand_kv_cache(prompt_kv_cache, n_samples)
            if expanded is not None:
                generate_kwargs["past_key_values"] = expanded

        handle = self._final_norm.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch_ids,
                    **self._attn_mask_kwarg(batch_mask),
                    max_new_tokens=max_new_tokens,
                    max_length=None,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs,
                )
        finally:
            handle.remove()

        # Stack on GPU, then single bulk transfer to CPU
        pre_logits = torch.stack(captured_prelogits_gpu, dim=1).cpu()  # (N, T_gen, d)

        return outputs, pre_logits

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def batch_decode(self, token_ids: torch.Tensor) -> list:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Wave (multi-prompt) generation primitives
    # ------------------------------------------------------------------

    def pad_and_stack_prompts(self, prompt_ids_list, attention_mask_list):
        """Left-pad and stack a list of (1, L_p) prompts into a (P, max_L) batch.

        Left-padding aligns the *right* edge of all prompts so that generated
        tokens land at consistent positions across the batch — required by
        HF generate for correct causal attention across mixed-length prompts.

        Returns:
            (padded_ids, padded_mask, prompt_lens) — padded_ids/mask are
            (P, max_L); prompt_lens is a CPU long tensor of unpadded lengths.
        """
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        P = len(prompt_ids_list)
        max_L = max(ids.shape[1] for ids in prompt_ids_list)
        device = prompt_ids_list[0].device
        ids_dtype = prompt_ids_list[0].dtype
        mask_dtype = attention_mask_list[0].dtype

        padded_ids = torch.full((P, max_L), pad_id, dtype=ids_dtype, device=device)
        padded_mask = torch.zeros((P, max_L), dtype=mask_dtype, device=device)
        prompt_lens = torch.zeros(P, dtype=torch.long)
        for p, (ids, mask) in enumerate(zip(prompt_ids_list, attention_mask_list)):
            L = ids.shape[1]
            padded_ids[p, max_L - L:] = ids[0]
            padded_mask[p, max_L - L:] = mask[0]
            prompt_lens[p] = L
        return padded_ids, padded_mask, prompt_lens

    def _repeat_interleave_kv_cache(self, kv_cache, n_per_prompt: int):
        """Take a (P, ...) cache and return (P*n_per_prompt, ...) by row repeat.

        Unlike `_expand_kv_cache` (broadcast view from batch=1), this does an
        actual repeat_interleave because per-row state will diverge during
        autoregressive generation.
        """
        if hasattr(kv_cache, "layers"):
            from transformers.cache_utils import DynamicCache, DynamicLayer
            expanded = DynamicCache()
            for layer in kv_cache.layers:
                if hasattr(layer, "keys"):
                    new_layer = DynamicLayer()
                    new_layer.dtype = layer.dtype
                    new_layer.device = layer.device
                    new_layer.keys = layer.keys.repeat_interleave(
                        n_per_prompt, dim=0).contiguous()
                    new_layer.values = layer.values.repeat_interleave(
                        n_per_prompt, dim=0).contiguous()
                    new_layer.is_initialized = True
                    expanded.layers.append(new_layer)
                elif hasattr(layer, "conv_states"):
                    from transformers.cache_utils import LinearAttentionLayer
                    new_layer = LinearAttentionLayer()
                    if getattr(layer, "is_conv_states_initialized", False):
                        cs = layer.conv_states.repeat_interleave(
                            n_per_prompt, dim=0).contiguous()
                        new_layer.lazy_initialization(conv_states=cs)
                    if getattr(layer, "is_recurrent_states_initialized", False):
                        rs = layer.recurrent_states.repeat_interleave(
                            n_per_prompt, dim=0).contiguous()
                        new_layer.lazy_initialization(recurrent_states=rs)
                    expanded.layers.append(new_layer)
                else:
                    logger.warning("Unknown cache layer type %s; skipping cache",
                                   type(layer).__name__)
                    return None
            return expanded
        logger.warning("KV cache type %s not supported for wave expansion; "
                       "prompts will recompute prefill",
                       type(kv_cache).__name__)
        return None

    def _wave_perturbation_hook(self, perturbations_per_row, captured_prelogits_gpu):
        """Build a forward hook that injects per-row perturbations and (optionally)
        captures pre-perturbation hidden states.

        perturbations_per_row: (B, T, d) — already row-stacked.
        captured_prelogits_gpu: list to append per-step hidden states to, or None.
        """
        step = [0]
        prefill_done = [False]
        capture = captured_prelogits_gpu is not None

        def hook_fn(module, input, output):
            t = step[0]
            if not prefill_done[0]:
                prefill_done[0] = True
                if capture:
                    captured_prelogits_gpu.append(output[:, -1, :].detach())
                if t < perturbations_per_row.shape[1]:
                    bs = output.shape[0]
                    v = perturbations_per_row[:bs, t, :].to(
                        output.device).to(output.dtype)
                    output = output.clone()
                    output[:, -1, :] = output[:, -1, :] + v
                step[0] += 1
                return output
            if capture:
                captured_prelogits_gpu.append(output[:, -1, :].detach())
            if t < perturbations_per_row.shape[1]:
                bs = output.shape[0]
                v = perturbations_per_row[:bs, t, :].unsqueeze(1).to(
                    output.device).to(output.dtype)
                output = output + v
            step[0] += 1
            return output

        return hook_fn

    def generate_with_perturbation_sampling_wave(
        self,
        prompt_ids_padded: torch.Tensor,
        attention_mask_padded: torch.Tensor,
        perturbations_per_prompt: torch.Tensor,
        n_per_prompt: int,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        prompt_kv_cache=None,
    ) -> tuple:
        """Multi-prompt sampling with per-prompt perturbation (MISVO/RSVO).

        Args:
            prompt_ids_padded: (P, max_L) left-padded prompt token ids
            attention_mask_padded: (P, max_L) attention mask (0 on left padding)
            perturbations_per_prompt: (P, T, d) — each prompt's u, repeated to
                each of its n_per_prompt rows internally.
            n_per_prompt: samples per prompt
            prompt_kv_cache: (P, ...) batched KV cache from compute_prompt_kv_cache
                applied to the padded prompt batch. Will be repeat_interleaved
                internally to (P*n, ...).

        Returns:
            (output_ids, pre_logits) — (P*n, max_L+T_gen) and (P*n, T_gen, d).
            Rows are ordered as [p=0 × n_per_prompt rows, p=1 × n_per_prompt rows, ...].
        """
        P, max_L = prompt_ids_padded.shape
        batch_ids = prompt_ids_padded.repeat_interleave(n_per_prompt, dim=0).contiguous()
        batch_mask = attention_mask_padded.repeat_interleave(n_per_prompt, dim=0).contiguous()
        perturbations = perturbations_per_prompt.repeat_interleave(n_per_prompt, dim=0)

        captured_prelogits_gpu = []
        hook_fn = self._wave_perturbation_hook(perturbations, captured_prelogits_gpu)

        generate_kwargs = {}
        if prompt_kv_cache is not None:
            expanded = self._repeat_interleave_kv_cache(prompt_kv_cache, n_per_prompt)
            if expanded is not None:
                generate_kwargs["past_key_values"] = expanded

        handle = self._final_norm.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch_ids,
                    **self._attn_mask_kwarg(batch_mask),
                    max_new_tokens=max_new_tokens,
                    max_length=None,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs,
                )
        finally:
            handle.remove()

        pre_logits = torch.stack(captured_prelogits_gpu, dim=1).cpu()
        return outputs, pre_logits

    def generate_with_perturbation_batch_wave(
        self,
        prompt_ids_padded: torch.Tensor,
        attention_mask_padded: torch.Tensor,
        perturbations_per_row: torch.Tensor,
        n_per_prompt: int,
        max_new_tokens: int = 128,
        prompt_kv_cache=None,
        capture_prelogits: bool = False,
    ):
        """Multi-prompt greedy generation with per-row perturbation (AISP, BoN-gauss).

        Args:
            prompt_ids_padded: (P, max_L) left-padded prompt ids
            attention_mask_padded: (P, max_L)
            perturbations_per_row: (P*n_per_prompt, T, d) — different perturbation
                per row, ordered consecutively per prompt.
            n_per_prompt: rows per prompt
            prompt_kv_cache: (P, ...) batched KV cache; expanded internally.
            capture_prelogits: if True, also return (P*n, T_gen, d) prelogits.
        """
        P, max_L = prompt_ids_padded.shape
        B = P * n_per_prompt
        assert perturbations_per_row.shape[0] == B, (
            f"perturbations rows {perturbations_per_row.shape[0]} != P*n_per_prompt {B}")

        batch_ids = prompt_ids_padded.repeat_interleave(n_per_prompt, dim=0).contiguous()
        batch_mask = attention_mask_padded.repeat_interleave(n_per_prompt, dim=0).contiguous()

        captured_prelogits_gpu = [] if capture_prelogits else None
        hook_fn = self._wave_perturbation_hook(perturbations_per_row, captured_prelogits_gpu)

        generate_kwargs = {}
        if prompt_kv_cache is not None:
            expanded = self._repeat_interleave_kv_cache(prompt_kv_cache, n_per_prompt)
            if expanded is not None:
                generate_kwargs["past_key_values"] = expanded

        handle = self._final_norm.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch_ids,
                    **self._attn_mask_kwarg(batch_mask),
                    max_new_tokens=max_new_tokens,
                    max_length=None,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs,
                )
        finally:
            handle.remove()

        if capture_prelogits:
            pre_logits = torch.stack(captured_prelogits_gpu, dim=1).cpu()
            return outputs, pre_logits
        return outputs

    def generate_samples_wave(
        self,
        prompt_ids_padded: torch.Tensor,
        attention_mask_padded: torch.Tensor,
        n_per_prompt: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Multi-prompt top-p sampling without perturbation (BoN top-p).

        Returns (P*n_per_prompt, max_L+T_gen) — ordered consecutively per prompt.
        """
        batch_ids = prompt_ids_padded.repeat_interleave(n_per_prompt, dim=0).contiguous()
        batch_mask = attention_mask_padded.repeat_interleave(n_per_prompt, dim=0).contiguous()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch_ids,
                **self._attn_mask_kwarg(batch_mask),
                max_new_tokens=max_new_tokens,
                max_length=None,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return outputs

    def encode_prompt(self, prompt) -> tuple:
        """Encode a prompt to token IDs and attention mask.

        Args:
            prompt: either a plain string or a dict with 'messages' key
                    (list of chat messages for apply_chat_template).
        """
        if isinstance(prompt, dict) and "messages" in prompt:
            if getattr(self.tokenizer, "chat_template", None):
                chat_kwargs = {}
                if self.enable_thinking is not None:
                    chat_kwargs["enable_thinking"] = self.enable_thinking
                result = self.tokenizer.apply_chat_template(
                    prompt["messages"],
                    return_tensors="pt",
                    add_generation_prompt=True,
                    **chat_kwargs,
                )
                # apply_chat_template returns a Tensor or a BatchEncoding
                if isinstance(result, torch.Tensor):
                    token_ids = result
                    attn_mask = torch.ones_like(token_ids)
                else:
                    token_ids = result["input_ids"]
                    attn_mask = result.get(
                        "attention_mask", torch.ones_like(token_ids))
                return (
                    token_ids.to(self.device),
                    attn_mask.to(self.device),
                )
            # No chat template available — fall back to plain text
            logger.warning("Tokenizer has no chat_template; using plain text encoding")
            prompt = prompt["text"]
        encoded = self.tokenizer(prompt, return_tensors="pt")
        ids = encoded["input_ids"]
        if ids.shape[1] == 0:
            raise RuntimeError(
                f"Tokenizer produced 0 tokens. "
                f"model_max_length={self.tokenizer.model_max_length}, "
                f"prompt_len={len(prompt)}, prompt[:200]={prompt[:200]!r}"
            )
        return (
            ids.to(self.device),
            encoded["attention_mask"].to(self.device),
        )
