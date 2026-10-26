LoRA fine-tuning for the first (candidate-generation) model is implemented in `training.py` and `model.py`.
The training command is `python main.py train-lora`; inference automatically uses `outputs/lora_adapter` when
`STAGE1_BACKEND=auto` and a trained adapter is present.

The implementation follows paper Eq. 13–16 and Table I: system/user/assistant role markers, prompt-masked assistant
loss, DeepSeek-R1-Distill-Llama-8B base model, LoRA rank 32 / alpha 160 / dropout 0.4 on `q_proj` and `v_proj`,
AdamW weight decay 0.01, learning rate 1e-4 with cosine schedule, 100 warmup steps, per-device batch size 1,
gradient accumulation 8, bf16, and 15 epochs. Exact source settings that the paper does not specify are kept in
`config.py`.

Training requires an expert-annotated JSONL dataset with non-empty `system`, `user`, and `assistant` string fields,
plus a CUDA GPU supporting bf16. The project refuses to start paper-configured training if that hardware is absent.
The bundled synthetic dataset is only for pipeline smoke tests and is not suitable for asserting model quality.

Stage two is a separately configured, larger refinement LLM (`OLLAMA_STAGE2_MODEL` for Ollama or `STAGE2_MODEL` for
the DeepSeek-compatible API). `outputs/lora_adapter` is generated locally and should not be committed.
