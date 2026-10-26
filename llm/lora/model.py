"""Local Transformers inference client for a trained stage-1 LoRA adapter."""
from pathlib import Path

import config
from llm.lora.training import render_messages


class LoRALocalLLM:
    kind = "local-lora"
    _cache = {}

    def __init__(self, base_model=config.LORA_BASE_MODEL, adapter_dir=config.LORA_ADAPTER_DIR):
        adapter_dir = Path(adapter_dir)
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"No trained LoRA adapter found at {adapter_dir}. "
                "Run `python main.py train-lora` or set STAGE1_BACKEND=auto."
            )
        import torch
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LoRA inference requires torch, transformers, accelerate, and peft. "
                "Install the optional LoRA dependencies from requirements.txt."
            ) from exc

        self.name = str(adapter_dir)
        self.model_name = f"{base_model}+LoRA:{adapter_dir}"
        key = (str(base_model), str(adapter_dir))
        if key not in self._cache:
            device = config.LOCAL_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
            is_cuda = device.startswith("cuda")
            dtype = (torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported()
                     else torch.float16 if is_cuda else torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
            model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
            self._cache[key] = (tokenizer, model, device)
        self.tokenizer, self.model, self.device = self._cache[key]

    def chat(self, messages, temperature=None):
        import torch

        prompt = render_messages(messages) + "[ASSISTANT]\n"
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=config.LOCAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0][encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
