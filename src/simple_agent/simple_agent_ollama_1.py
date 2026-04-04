import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL ="llama3"
prompt=""
request_json={
    "model": MODEL,
    "prompt": prompt,
    "stream": False
}


# --- Tools ---
def calculator(expression):
    """

    :param expression:
    :return:
    """
    try:
        return str(eval(expression))
    except Exception as e:
        print(f"An error occurred: {e}")
        return "ERROR IN CALCULATION"

# --- LLM Call ---
def invoke_llm(prompt):
    """

    :param prompt:
    :return:
    """
    response = requests.post(
        url=OLLAMA_URL,
        json=request_json
    )