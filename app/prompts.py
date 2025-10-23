def make_prompt():
    prompt_template = """
        You are a helpful assistant.
        Use the provided context if relevant; if not, answer normally.

        [QUESTION]
        {user_query}
        
        [CONTEXT]
        {context}
        """
    return prompt_template


def make_prompt2(user_query, context):
    prompt = f"""
You are a helpful assistant.
Use the provided context if relevant; if not, answer normally.

[QUESTION]
{user_query}

[CONTEXT]
{context}
"""
    return prompt
