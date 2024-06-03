import streamlit as st
from langchain_core.prompts import PromptTemplate
from st_pages import add_page_title

from src.utils import llm

st.set_page_config(page_title='🦜 Nexa')

add_page_title()
st.text("Работа с документами")


def generate_response(dialogue: str):
    return llm(dialogue, max_tokens=2000, repeat_penalty=1.0)


PROMPT_TEMPLATE = """<|im_start|>\nuser\nОтветь на вопрос базируясь только на этом контексте:

{context}

---
Ответь на вопрос, используя только контекст: {question}<|im_end|>
"""
prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

uploaded_file = st.file_uploader("Загрузить документ", type=("txt", "md"))

question = st.text_input(
    "Спросите что-нибудь о документе",
    placeholder="Можете ли вы дать мне краткое изложение?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    with st.spinner("Подождите ..."):
        context_text = uploaded_file.read().decode('utf-8')

        prompt = prompt_template.format(context=context_text, question=question)

        response = generate_response(dialogue=prompt)['choices'][0]['text']

        response = response.replace('assistant', '').replace('ssistant', '')

    st.write(response)
