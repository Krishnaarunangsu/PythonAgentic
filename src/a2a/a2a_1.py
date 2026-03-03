import json
import uuid
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# Load Local Tiny Model
# -------------------------
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cpu"
model.to(device)

# -------------------------
# A2A Protocol Message
# -------------------------
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

# -------------------------
# Worker Agent
# -------------------------
class WorkerAgent:

    def __init__(self, name):
        self.name = name

    def handle_message(self, message):
        instruction = message["payload"]["instruction"]
        data = message["payload"]["data"]

        prompt = f"{instruction}:\n{data}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=150)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return create_message(
            sender=self.name,
            receiver=message["from"],
            msg_type="response",
            payload={"result": response_text}
        )

# -------------------------
# Supervisor Agent
# -------------------------
class SupervisorAgent:

    def __init__(self, name, worker):
        self.name = name
        self.worker = worker

    def process_user_request(self, user_text):

        # Create A2A task message
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

        # Send to Worker
        worker_response = self.worker.handle_message(task_message)

        print("\n📥 Worker → Supervisor")
        print(json.dumps(worker_response, indent=2))

        return worker_response["payload"]["result"]

# -------------------------
# Run Example
# -------------------------
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