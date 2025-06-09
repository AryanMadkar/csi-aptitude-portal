from flask import Flask, render_template, request, jsonify, session
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import LanguageModelInput
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_PATH = "./TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# === Load model if not present ===
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("📦 Downloading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
    else:
        print("✅ Loading local model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    return tokenizer, model

tokenizer, model = load_model()

# === Custom LangChain-compatible wrapper ===
class CustomLLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(prompt):].strip()

    @property
    def _llm_type(self) -> str:
        return "custom_local_model"

# === LangChain setup ===
llm = CustomLLM()
memory = ConversationBufferMemory(memory_key="chat_history")

template = """
You are a helpful AI. Continue the conversation.

{chat_history}
User: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# === Init session memory ===
@app.before_request
def init_session():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "session_start" not in session:
        session["session_start"] = time.strftime("%Y-%m-%d %H:%M:%S")

# === Routes ===
@app.route("/")
def home():
    return render_template("index.html")  # Connect your frontend here

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Say something first, bruh."})

    response = chain.run(human_input=user_input)

    session["chat_history"].append({
        "user": user_input,
        "bot": response
    })

    return jsonify({
        "response": response,
        "chat_history": session["chat_history"]
    })

@app.route("/reset", methods=["POST"])
def reset():
    session.pop("chat_history", None)
    session.pop("session_start", None)
    memory.clear()
    return jsonify({"status": "Reset done!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
