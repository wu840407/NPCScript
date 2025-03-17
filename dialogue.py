from transformers import AutoModelForCausalLM, AutoTokenizer

# 加載模型和分詞器
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 接收輸入並生成回應
player_input = "你好，你是谁？"
inputs = tokenizer.encode(player_input + tokenizer.eos_token, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("NPC回應:", response)