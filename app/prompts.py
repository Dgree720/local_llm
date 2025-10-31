def make_prompt():
    prompt_template = """You are a helpful AI assistant.
    Answer the Users Question, strictly **using the provided context and cite it accordingly**, if applicable.
    **If however there is no context given, simply answer the user question normally and do not cite anything.**
    
    **User Question:** {user_query}

    **Context:** {context}
    
    Now you:
"""
    return prompt_template
