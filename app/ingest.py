import os

knowledge_path = "/home/andreas/Documents/Python/local_llm/knowledge"

for file in os.listdir(knowledge_path):
    print(file)
    path = f"{knowledge_path}/{file}"
    with open(path, "r") as f:
        content = f.read()
        print(content)
