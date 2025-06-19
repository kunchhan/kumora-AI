def load_kumora_prompt(**kwargs):
    with open("prompts/kumora_prompt.txt", "r") as file:
        template = file.read()
    return template.format(**kwargs)
