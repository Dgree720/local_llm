def make_prompt(user_query, context):
    prompt = f"""
You are a helpful assistant.
Use the provided context if relevant; if not, answer normally.

[CONTEXT]
{context}

[QUESTION]
{user_query}

"""
    return prompt
