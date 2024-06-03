import streamlit as st
from st_pages import show_pages_from_config, add_page_title

from src.utils import set_model, llm, set_rag

model_path = 'model_path/model-q2_K.gguf'

st.set_page_config(page_title='🦜 Nexa')


def generate_response(dialogue: str):
    return llm(dialogue, max_tokens=1000, repeat_penalty=1.0)


user = "user\n"
assistant = "assistant\n"
start = "<|im_start|>"
end = "<|im_end|>"
default_promt = "\nОтвечай только на русском языке.\n"

show_pages_from_config()

# st.write("# 🦜 Nexa")
add_page_title()
st.text("Используйте GPT для решения задач")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": assistant, "content": "Привет! Чем могу помочь?"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == assistant:
        with st.chat_message(assistant, avatar="🤖"):
            st.write(msg["content"])
    else:
        with st.chat_message(user, avatar="🐱"):
            st.write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": user, "content": prompt})
    with st.chat_message(user, avatar="🐱"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != assistant:
    with st.chat_message(assistant, avatar="🤖", ):
        with st.spinner("Подождите ..."):
            string_dialogue = ""

            for dict_message in st.session_state.messages:
                if dict_message["role"] == user:
                    string_dialogue += start + user + dict_message["content"] + end
                else:
                    string_dialogue += start + assistant + dict_message["content"] + end

            output = generate_response(dialogue=string_dialogue)

            response = output['choices'][0]['text'].replace('assistant', '')

            placeholder = st.empty()
            placeholder.markdown(response)

    message = {"role": assistant, "content": response}
    st.session_state.messages.append(message)


def main():
    try:
        set_model()
        set_rag()
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
