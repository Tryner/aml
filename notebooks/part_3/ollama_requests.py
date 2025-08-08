# %%
import requests
import json
# %%
# Configure the payload for the request
# If you downloaded a different model, change the "model" key in the data dictionary
# Promt is the text you want to send to the model
data={
    "model": r"gemma3",
    "prompt": r"Why is the sky blue?",
}
# %%
# Create a session to handle the request,
# since the result is streamed token by token
s = requests.Session()

response_strings = []

with s.post("http://127.0.0.1:11434/api/generate",
            data=json.dumps(data),
            stream=True) as response:
    for line in response.iter_lines():
        if line:
            try:
                response_strings.append(json.loads(line.decode("utf8").replace("'", '"').replace("\'", "\""))["response"])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {line}")
                print(f"Error: {e}")

# %%
print("".join(response_strings))

# %%
