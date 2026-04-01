from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from chromadb import PersistentClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from typing import Optional, List, Tuple


MODEL = "qwen3-embedding:0.6b"
DB_NAME = str(Path(__file__).parent / "chroma_db")
load_dotenv(override=True)


embeddings = OllamaEmbeddings(model=MODEL)
vector_store = Chroma(collection_name="my_collection", embedding_function=embeddings, persist_directory=DB_NAME)

retriever = vector_store.as_retriever()

llm = OllamaLLM(model="qwen3:0.6b", temperature=0)

SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""


def answer_question(question: str, history=None):
    if history is None:
        history = []
    print ("***********************************\n")
    print(f"Received history \n : {history} \n\n")
    print ("\n***********************************\n")
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    print ("***********************************\n")
    print(f"System prompt: \n {system_prompt} \n\n")
    print ("\n***********************************\n")
    
    messages = [SystemMessage(content=system_prompt)]

    for msg in history:
        role = msg["role"]
        text = msg["content"][0]["text"]

        if role == "user":
            messages.append(HumanMessage(content=text))

        elif role == "assistant":
            messages.append(AIMessage(content=text))


    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return str(response)

if __name__ == "__main__":
    gr.ChatInterface(fn=answer_question).launch()
