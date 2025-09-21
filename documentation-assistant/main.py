from typing import Set

from backend.core import llm_go
import streamlit as st

prompt = st.text_input("prompt", place_holder= "what is your question")
if (
    ["chat_answers_history"] not in st.session_state,
    ["chat_history"] not in st.session_state,
    ["user_prompt"] not in st.session_state,
):
    st.session_state["chat_history"] = [],
    st.session_state["chat_answers_history"] = [],
    st.session_state["user_prompt"] = []

def source_string(source_urls : Set[str] ) -> str :
    if not source_urls :
        return ""
    source_list=list(source_urls)
    source_list.sort()
    source_string= "sources:\n"
    for i,source in enumerate(source_list):
        source_string += f"{i}. {source}\n"
    return source_string


if prompt:
    with st.spinner("Generating reponse"):
        generated_response = llm_go(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set([doc.metadata["source"] for doc in generated_response["docs"]])
        formatted_response = f":{generated_response['result']}:\n\n\n{source_string(sources)}"
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_history"].append("ai",generated_response["result"])
        st.session_state["chat_history"].append("human",prompt)


if st.session_state["chat_answers_history"] :
    for user_query , generated_response in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
    ):

            st.chat_message('user').write(user_query)
            st.chat_message('assistant').write(generated_response)



