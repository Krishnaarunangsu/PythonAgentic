import os
import json
import uuid
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# ----------------------------------
# Load Environment Variables
# ----------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# ----------------------------------
# Load Tiny Model (Authenticated)
# ----------------------------------
model_name = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN
)

device = "cpu"
model.to(device)

print("✅ Model downloaded and loaded successfully.")

# ----------------------------------
# A2A Protocol Message Structure
# ----------------------------------
def create_message(sender, receiver, msg_type, payload):
    return {
        "protocol": "A2A-1.0",
        "from": sender,
        "to": receiver,
        "type": msg_type,
        "payload": payload,
        "metadata": {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }
    }

# ----------------------------------
# Worker Agent
# ----------------------------------
class WorkerAgent:

    def __init__(self, name):
        self.name = name

    def handle_message(self, message):

        instruction = message["payload"]["instruction"]
        data = message["payload"]["data"]

        prompt = f"{instruction}:\n{data}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                do_sample=True,
                temperature=0.7
            )

        response_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return create_message(
            sender=self.name,
            receiver=message["from"],
            msg_type="response",
            payload={"result": response_text}
        )

# ----------------------------------
# Supervisor Agent
# ----------------------------------
class SupervisorAgent:

    def __init__(self, name, worker):
        self.name = name
        self.worker = worker

    def process_user_request(self, user_text):

        task_message = create_message(
            sender=self.name,
            receiver=self.worker.name,
            msg_type="task",
            payload={
                "instruction": "Summarize this text",
                "data": user_text
            }
        )

        print("\n📤 Supervisor → Worker")
        print(json.dumps(task_message, indent=2))

        worker_response = self.worker.handle_message(task_message)

        print("\n📥 Worker → Supervisor")
        print(json.dumps(worker_response, indent=2))

        return worker_response["payload"]["result"]

# ----------------------------------
# Run Example
# ----------------------------------
if __name__ == "__main__":

    worker = WorkerAgent("WorkerAgent")
    supervisor = SupervisorAgent("SupervisorAgent", worker)

    user_input = """
    Artificial Intelligence is transforming industries by enabling automation,
    predictive analytics, and intelligent decision making across domains.
    """

    final_output = supervisor.process_user_request(user_input)

    print("\n✅ Final Output to User:\n")
    print(final_output)