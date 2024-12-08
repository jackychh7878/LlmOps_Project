import os
import json
import sys
import boto3
import streamlit as streamlit
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS


