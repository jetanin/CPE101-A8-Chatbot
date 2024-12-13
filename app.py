import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re

device = "cuda"

model_id = "nectec/Pathumma-llm-text-1.0.0"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)


def get_attention_mask(inputs):
    return (inputs != tokenizer.pad_token_id).type(torch.long)


def chatbot(gender, age, weight, heigth, plan, day, level, question):
    global llm_response
    prompt = f"แนะนำการออกกำลังกายสำหรับ{gender} อายุ {age} ปี น้ำหนัก{weight} กิโลกรัม ส่วนสูง{heigth} เซนติเมตร ที่มีแผนการออกกำลังกาย {plan} มีเวลาออกกำลังกาย {day} วันต่อสัปดาห์ ระดับความเชี่ยวชาญ {level} และมีคำถามเพิ่มเติมว่า {question}"

    messages = [
        {"role": "system", "content": f"คุณเป็นผู้เชี่ยวชาญด้านการออกกำลังกาย คุณให้คำตอบที่ครอบคลุมและให้ข้อมูลต่อคำถามของผู้ใช้ {prompt}"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    attention_mask = get_attention_mask(model_inputs.input_ids)

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2000,
        repetition_penalty=1.1
    )
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    pattern = r"system.*?assistant"

    text = response

    result = re.sub(pattern, "", text, flags=re.DOTALL)
    print("response: \n")
    print(response)
    print("\n\nresponse: \n")
    print(result)
    
    return result



interface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Radio(choices=["ชาย", "หญิง"], label="เพศสภาพของคุณ:"),
        gr.Textbox(label="อายุของคุณ"),
        gr.Textbox(label="น้ำหนักของคุณ"),
        gr.Textbox(label="ส่วนสูงของคุณ"),
        gr.Textbox(label="เป้าหมายการออกกำลังกาย (เช่น เพิ่มกล้ามเนื้อ ลดน้ำหนัก):"),
        gr.Slider(minimum=1, maximum=7, label="จำนวนวันที่ว่างออกกำลังกายต่อสัปดาห์ของคุณ (วัน):", step=1),
        gr.Radio(choices=["มือใหม่", "ปานกลาง", "เชี่ยวชาญ"], label="ระดับความเชี่ยวชาญ:"),
        gr.Textbox(label="คำถามเพิ่มเติม:"),
    ],
    outputs="text",
    title="Exercise Consultant Chatbot",
    description="Get personalized exercise recommendations from your AI assistant!",
    elem_id="chat-container"
)

if __name__ == "__main__":
  interface.launch()
