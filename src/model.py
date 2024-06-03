from threading import Lock

from llama_cpp import Llama

from src.settings import Settings


# TODO: refactor

class LocalLlama:
    _current_model = None

    def __init__(self, settings: Settings) -> None:
        self._current_model = self._load_llama_from_model_settings(settings=settings)

    def call(self) -> Llama:
        return self._current_model

    def free(self):
        if self._current_model:
            del self._current_model

    @staticmethod
    def _load_llama_from_model_settings(settings: Settings) -> Llama:
        kwargs = {}

        create_fn = Llama
        kwargs["model_path"] = settings.model

        _model = create_fn(
            **kwargs,
            n_ctx=2048,
            verbose=True,
            n_gpu_layers=32,
            # Model Params
            # n_gpu_layers=settings.n_gpu_layers,
            # main_gpu=settings.main_gpu,
            # tensor_split=settings.tensor_split,
            # vocab_only=settings.vocab_only,
            # use_mmap=settings.use_mmap,
            # use_mlock=settings.use_mlock,
            # kv_overrides=kv_overrides,
            # # Context Params
            # seed=settings.seed,
            # n_ctx=settings.n_ctx,
            # n_batch=settings.n_batch,
            # n_threads=settings.n_threads,
            # n_threads_batch=settings.n_threads_batch,
            # rope_scaling_type=settings.rope_scaling_type,
            # rope_freq_base=settings.rope_freq_base,
            # rope_freq_scale=settings.rope_freq_scale,
            # yarn_ext_factor=settings.yarn_ext_factor,
            # yarn_attn_factor=settings.yarn_attn_factor,
            # yarn_beta_fast=settings.yarn_beta_fast,
            # yarn_beta_slow=settings.yarn_beta_slow,
            # yarn_orig_ctx=settings.yarn_orig_ctx,
            # mul_mat_q=settings.mul_mat_q,
            # logits_all=settings.logits_all,
            # embedding=settings.embedding,
            # offload_kqv=settings.offload_kqv,
            # # Sampling Params
            # last_n_tokens_size=settings.last_n_tokens_size,
            # # LoRA Params
            # lora_base=settings.lora_base,
            # lora_path=settings.lora_path,
            # # Backend Params
            # numa=settings.numa,
            # # Chat Format Params
            # chat_format=settings.chat_format,
            # chat_handler=chat_handler,
            # # Speculative Decoding
            # draft_model=draft_model,
            # # Tokenizer
            # tokenizer=tokenizer,
            # # Misc
            # verbose=settings.verbose,
        )
        # if settings.cache:
        #     if settings.cache_type == "disk":
        #         if settings.verbose:
        #             print(f"Using disk cache with size {settings.cache_size}")
        #         cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
        #     else:
        #         if settings.verbose:
        #             print(f"Using ram cache with size {settings.cache_size}")
        #         cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)
        #     _model.set_cache(cache)
        return _model


llama_outer_lock = Lock()
llama_inner_lock = Lock()

_local_llama: LocalLlama | None = None


def set_local_llama(settings: Settings):
    global _local_llama
    _local_llama = LocalLlama(settings=settings)


def get_local_llama():
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            return _local_llama
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()
