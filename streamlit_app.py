import os

# --- FAISS 안정화: faiss/numpy import 전에 설정 (커널·앱 멈춤 예방) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
faiss.omp_set_num_threads(1)

import tempfile
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

# ====================================================================
#  📄🔍 PDF RAG + 웹 검색 챗봇 — 미니 프로젝트 스켈레톤
#
#  오늘 노트북에서 배운 코드를 아래 [TODO]에 조립하면 나만의 웹 앱이 됩니다.
#  - [TODO 1~4] : PDF → retriever → 도구 만들기 (참고: 11.AgentRAG)
#  - [TODO 5~7] : create_agent로 에이전트 구성 (참고: 11.AgentRAG)
#  - Streamlit UI 부분은 완성되어 있으니 그대로 쓰면 됩니다.
#
#  실행: streamlit run streamlit_app.py
# ====================================================================


# --------------------------------------------------------------------
# 1. PDF -> retriever 도구 만들기  (참고: 11.AgentRAG 노트북)
# --------------------------------------------------------------------
def build_pdf_tool(uploaded_files):
    # 업로드된 PDF들을 임시 파일로 저장하며 로드합니다. (이 부분은 그대로 사용)
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        all_docs.extend(PyPDFLoader(tmp_path).load())

    # [TODO 1] 텍스트 분할기를 생성하고 문서를 분할하세요.
    # 조건: chunk_size=1000, chunk_overlap=200
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    # [TODO 2] 분할된 문서와 OpenAIEmbeddings("text-embedding-3-small")로
    #          FAISS 벡터스토어를 생성하세요.
    vector = FAISS.from_documents(
        documents=split_docs,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    # [TODO 3] 벡터스토어를 retriever로 변환하세요. (search_kwargs={"k": 5})
    retriever = vector.as_retriever(search_kwargs={"k": 5})

    # [TODO 4] 아래 함수를 @tool 데코레이터로 감싸고,
    #          1. 도구 설명을 작성하세요. 
    #          2. retriever로 query를 검색한 결과를 반환하도록 완성하세요.
    #          docstring은 에이전트가 도구를 고르는 기준이 되므로 명확하게 쓰세요!
    #TODO
    def pdf_search(query: str) -> str:
        """문서에서 질문을 검색합니다."""
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)

    return pdf_search


# --------------------------------------------------------------------
# 2. create_agent 로 에이전트 구성  (참고: 11.AgentRAG 노트북)
# --------------------------------------------------------------------
def build_agent(tools):
    # [TODO 5] gpt-4o-mini, temperature=0 인 ChatOpenAI 모델을 생성하세요.
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # [TODO 6] 나만의 시스템 프롬프트를 작성하세요.
    # 포함할 것: ① 챗봇의 역할/페르소나  ② 도구 사용 우선순위(pdf_search 먼저)
    #           ③ 답변 언어·말투
    system_prompt = (
        "반드시 pdf_search 먼저 사용하세요. 그 다음에 tavily_search를 사용하세요."
    )

    # [TODO 7] create_agent로 에이전트를 생성해 반환하세요.
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )



# --------------------------------------------------------------------
# 3. Streamlit UI  (완성 코드 — 수정 없이 사용 가능)
# --------------------------------------------------------------------
# 💡 커스터마이즈 아이디어: page_title / page_icon / title 을 내 챗봇 이름으로 바꿔보세요.
st.set_page_config(page_title="RAG + Web Search 챗봇", page_icon="📝")
st.title("📄🔍 PDF RAG + 웹 검색 챗봇")

with st.sidebar:
    st.header("설정")
    openai_key = st.text_input("OpenAI API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")
    uploaded_files = st.file_uploader("PDF 업로드", type="pdf", accept_multiple_files=True)
    build = st.button("에이전트 만들기")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key

if build:
    if not (openai_key and tavily_key and uploaded_files):
        st.warning("API 키 2개와 PDF를 모두 입력하세요.")
    else:
        with st.spinner("문서 인덱싱 및 에이전트 생성 중..."):
            tools = [build_pdf_tool(uploaded_files), TavilySearch(max_results=5)]
            st.session_state.agent = build_agent(tools)
            st.session_state.messages = []  # 새 에이전트를 만들면 대화 기록도 새로 시작
        st.success("준비 완료! 아래에 질문하세요.")

if "agent" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 지금까지의 대화를 화면에 다시 그립니다.
    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    query = st.chat_input("질문을 입력하세요")
    if query:
        st.chat_message("user").write(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                # 이전 대화 전체를 함께 보내야 에이전트가 앞의 대화를 기억합니다.
                result = st.session_state.agent.invoke(
                    {"messages": st.session_state.messages}
                )
            answer = result["messages"][-1].content
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})