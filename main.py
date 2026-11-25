import transformers

model = "taobao-mnn/MNN-Pocket-0.3B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    device_map="auto",
    dtype="auto"
)

print("Chatbot Started! Type 'exit' to stop.\n")

messages = []

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Bye! Have a nice day. Thank you!")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    output = pipeline(
        messages,
        max_new_tokens=100
    )

    response = output[0]["generated_text"]

    if isinstance(response, list):
        answer = response[-1]["content"]
    else:
        answer = response

    print("Chatbot:", answer)

    messages.append({
        "role": "assistant",
        "content": answer
    })
