from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)

print(tokenizer.decode(outputs[0]))
# for layer in model.model.layers:
#     print(layer.mlp)
#     break