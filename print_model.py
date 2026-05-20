from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/mnt/data/ai-models/qwen3-0.6b", local_files_only=True)
print(model)