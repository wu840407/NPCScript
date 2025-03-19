from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加載模型和分詞器（只加載一次）
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(player_input):
    """
    接收玩家輸入並生成 NPC 回應
    """
    # 預處理輸入
    inputs = tokenizer(player_input, return_tensors="pt")
    attention_mask = inputs["attention_mask"]
    
    # 生成回應
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=attention_mask, 
        max_length=50, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解碼並返回回應
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 如果使用 Web 服務，可以加入以下代碼
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def api_generate_response():
    player_input = request.json['input']
    response = generate_response(player_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)