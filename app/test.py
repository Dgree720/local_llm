import time, json, requests

url = "http://127.0.0.1:11434/api/generate"
data = {
    "model": "gemma3:1b",
    "prompt": """You are a local RAG assistant running with the Gemma model via Ollama.

Your task: use retrieved context first, then your own reasoning if context is missing or insufficient.

### Rules
1. **Ground in context.** Use it whenever relevant.
2. **If no context or unrelated:** answer generally, but state that context was unused.

## input
Here is the user Query:
hi

And here the retrieved relevant context to answer the query:
Progress440 66 856 482 648

856 128.4 908 180.4 882

908 136.2 1180 408.2 1,044

1180 177 1196 193 1,188

4,825 10,205 4,510 89,100 78,300 66,150 372,557 501,811 440,986 931,392 1,102,464 1,254,528 356,400 313,200 264,600 14,256 12,528 10,584 355,388 1,026,429 1,385,193 1,644,483 1,864,520

11,000 16,500 92,928 232,320 66,000 2,640

12,050 48,600 273,715 684,288 194,400 7,776

0 24 12 52,800 6,000 12,960 20,400 4,800 7,200 104,160

24 48 36 158,400 18,000 38,880 61,200 14,400 21,600 312,480

Now you:""",
    "stream": True,
}

t0 = time.time()
with requests.post(url, json=data, stream=True) as r:
    print("Connected after", round(time.time() - t0, 2), "s")
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        print("Got chunk at", round(time.time() - t0, 2), "s", "->", line)
