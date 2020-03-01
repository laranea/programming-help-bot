import streamlit as st
from dialogue_manager import DialogueManager

st.title('Programming help bot')

DM = DialogueManager()
DM.create_chitchat_bot()

st.markdown('### Hi, I am a bot. How can I be of help?')

input_ = st.empty()
value = ''
if st.button('Clear'):
    value = ' '

question = input_.text_input('type your question here ...',  value)

if question != '' and question != ' ':
    response = DM.generate_answer(question)
    st.write(response)


