def process_chatglm_messages(messages, tools=None):
    _messages = messages
    messages = []
    msg_has_sys = False
    if tools:
        messages.append(
            {
                "role": "system",
                "content": "Answer the following questions as best as you can. You have access to the following tools:",
                "tools": tools,
            }
        )
        msg_has_sys = True

    for m in _messages:
        role, content, func_call = m.role, m.content, m.function_call
        if role == "function":
            messages.append({"role": "observation", "content": content})

        elif role == "assistant" and func_call is not None:
            for response in content.split("<|assistant|>"):
                metadata, sub_content = response.split("\n", maxsplit=1)
                messages.append(
                    {"role": role, "metadata": metadata, "content": sub_content.strip()}
                )
        else:
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            messages.append({"role": role, "content": content})
    return messages
