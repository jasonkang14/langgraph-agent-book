# %%
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# 환경변수를 불러옴
load_dotenv()


embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
)


# %%
from langchain_pinecone import PineconeVectorStore

index_name = 'income-tax-index'  # 인덱스 이름 설정

# %%
vector_store = PineconeVectorStore.from_existing_index(
                       index_name=index_name,
                       embedding=embedding, )

# %%
from langchain_core.tools.retriever import create_retriever_tool

retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "search_income_tax_law",
    "2025년 대한민국의 소득세법을 검색한 결과를 반환합니다",
)


# %%
from langchain.agents import create_agent

income_tax_agent = create_agent(model='gpt-4o', tools=[retriever_tool])
# %%



