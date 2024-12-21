import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# Streamlit 기본 설정
st.title("Tax Law Chatbot")
st.write("세법 FAQ를 검색하고 답변을 제공합니다.")

# Chroma 데이터베이스 경로
vector_store_path = "vector_store/chroma/tax"

# 임베딩 모델 및 벡터 스토어 초기화
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(vector_store_path):
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embedding_model,
        collection_name="tax"
    )
else:
    st.error("벡터 데이터베이스를 찾을 수 없습니다. 데이터를 먼저 처리해주세요.")
    st.stop()

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini")

# 체인 구성
prompt_template = ChatPromptTemplate.from_template(
    "You are an expert assistant on tax law. Use the following context to answer the question. "
    "If you don't know the answer, say you don't know.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)
chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=llm,
    return_source_documents=True
)

# 사용자 입력 받기
query = st.text_input("질문을 입력하세요:")

if query:
    # 질의응답 실행
    with st.spinner("답변 생성 중..."):
        result = chain.invoke({"query": query})
        print(result)
    
    # 결과 출력
    st.subheader("AI의 답변:")
    st.write(result["result"])

    # 출처 문서 출력
    st.subheader("참고한 문서:")
    for doc in result["source_documents"]:
        st.write(f"- **출처**: {doc.metadata['source']}")
        st.write(doc.page_content)

