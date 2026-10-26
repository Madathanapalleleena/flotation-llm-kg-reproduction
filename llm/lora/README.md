LoRA fine-tuning (paper Sec. III-B-3, Eq. 13-16, Table I) has been **removed from this build** on request.

Stage 1 of the two-stage generation therefore uses a plain chat model (`STAGE1_MODEL` in `config.py` / `.env`).
When you want it back, this folder is where it goes: `train_lora.py` (JSONL -> tokenised with the [SYSTEM]/[USER]/[ASSISTANT]
format of Eq. 14, response-only loss mask of Eq. 15, rank 32 / alpha 160 / dropout 0.4 on q_proj,v_proj, AdamW wd 0.01,
lr 1e-4 cosine to 0 (Eq. 16), warmup 100, bs 1 x grad-accum 8, bf16, 15 epochs) and the adapter weights.
`data/training/flotation_train.jsonl` already has the system/user/assistant layout that script would consume.
