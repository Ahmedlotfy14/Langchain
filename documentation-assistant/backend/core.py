from dotenv import load_dotenv
from typing import Any, Dict, List
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from consts import INDEX_NAME
load_dotenv()


def llm_go(query: str , chat_history: List[Dict[str , Any]] = []) :
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(embedding=embeddings,index_name=INDEX_NAME)
    chat=ChatOpenAI(verbose=True,temperature=0)

    modify_prompt=hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_prompt= hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_doc_chain = create_stuff_documents_chain(chat,retrieval_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, prompt=modify_prompt,retriever=docsearch.as_retriever()
    )

    retreival_chain = create_retrieval_chain(
        retriever=history_aware_retriever,combine_docs_chain=stuff_doc_chain
    )

    result = retreival_chain.invoke(input={"query":query,"chat_history":chat_history})
    return result





