from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# from langchain_community.llms import Bedrock
from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
import boto3

from typing import Dict
from langchain_core.runnables import RunnablePassthrough

bedrock = boto3.client("bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def get_llama_llm():
    llm = BedrockLLM(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock)
    return llm


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


def get_response_llm(llm, vectorstore_faiss, query):
    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    retriever = vectorstore_faiss.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    retrieval_chain = RunnablePassthrough.assign(context=parse_retriever_input | retriever).assign(
        answer=document_chain)

    return retrieval_chain.invoke({"messages": [HumanMessage(content=query)]})


if __name__ == "__main__":
    vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    llm = get_llama_llm()
    query = "What is RAG token?"
    response = get_response_llm(llm, vectorstore, query)
    print(response['answer'])
