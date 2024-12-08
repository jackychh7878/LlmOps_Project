import os
import json
import sys
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

from QASystem.ingestion import data_ingestion, get_vector_store
from QASystem.retrievalAndGeneration import get_llama_llm, get_response_llm

bedrock = boto3.client("bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def main():
    st.set_page_config("QA with Doc")
    st.header("QA with Doc using langchain and AWSBedrock")

    user_question = st.text_input("Ask a question from the pdf files")

    with st.sidebar:
        st.title("update or create the vector store")
        if st.button("vectors update"):
            with st.spinner("processing..."):
                docs = data_ingestion
                get_vector_store(docs)
                st.success("done")

        if st.button("llama model"):
            with st.spinner("processing..."):
                vectorstore = FAISS.load_local("QASystem/faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llama_llm()
                response = get_response_llm(llm, vectorstore, user_question)
                st.write(response['answer'])
                st.success("Done")


if __name__ == "__main__":
    #this is my main method
    main()
