import streamlit as st
import tiktoken # 문자의 개수를 나눌때 토근의 개수로 나누기 위해 사용
from loguru import logger # 구동을 한것을 남기기위해 로그 사용

from langchain.chains import ConversationalRetrievalChain # 메모리를 가지고 있는 체인
from langchain.chat_models import ChatOpenAI # LLM 모델을 가져오기위한 ChatOpenAI

from langchain.document_loaders import PyPDFLoader # pdf 파일을 읽기 위한 로더
from langchain.document_loaders import Docx2txtLoader # 워드 파일을 읽어오기 위한 로더
from langchain.document_loaders import UnstructuredPowerPointLoader # ppt 파일을 읽기 위한 로더

from langchain.text_splitter import RecursiveCharacterTextSplitter # 텍스트를 나누기 위한
from langchain.embeddings import HuggingFaceEmbeddings  # 한국어에 특화된 인베딩 모델

from langchain.memory import ConversationBufferMemory # 몇개의 대화를 메모리에 저장할때 필요
from langchain.vectorstores import FAISS # 벡터 스토어 저장할때

# from streamlit_chat import message 메모리 구현을 위한 라이브러리
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main(): # 메인 함수
    st.set_page_config( # 페이지의 상세 사항을 정리 상단탭에 나옴.
    page_title="DirChat",
    page_icon=":books:")

    st.title("_Private Data :red[QA Chat]_ :books:") # 타이틀 제목, _ 기울기, red 텍스트 색깔

    if "conversation" not in st.session_state: # 나중에 사용하기 위해 먼저 여기서 선언 해 줘야 함.
        st.session_state.conversation = None # 세션 스테이트 컨버세이션을 none으로 설정

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None # 쳇 히스토리도 none 으로 설정

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar: # 하위 구성요소들이 집해되기 위해 with문을 사용. 사이드바 적용
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process: # Process 누르면 작동 된다.
        if not openai_api_key: # 오픈API 키가 없을 때 입력하라
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files) # 업로드된 파일을 텍스트로 변환
        text_chunks = get_text_chunks(files_text) # 텍스트로 변환되 파일을 chunk로 나누기
        vetorestore = get_vectorstore(text_chunks) # 벡터 스토어를 이용해 벡터화
        
        # get_conversation_chain 함수를 통해 LMM답변을 할수 있도록 체인을 구성하여 저장
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True
    
    # 체팅 화면을 구현하기위 한 코드
    if 'messages' not in st.session_state: # 초깃값
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
    # 메세지의 내용을 마크 다운 한다. 질문에 대한 컨텐츠를 묶어서 화면에 구현
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 메모리를 갖고 답변을 이어서 하기위한 구현
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic. 질문창을 위한 화면 구현
    if query := st.chat_input("질문을 입력해주세요."): # 초깃값
        st.session_state.messages.append({"role": "user", "content": query}) # 질문이 들어오면 세션 스테이트에 저장

        with st.chat_message("user"): # 쿼리로 질문을 처리를 위한 화면 처리
            st.markdown(query)

        with st.chat_message("assistant"): # 답변에 대한 화면 처리
            chain = st.session_state.conversation

            with st.spinner("Thinking..."): # 로딩화면
                result = chain({"question": query}) # 답변을 저장
                with get_openai_callback() as cb: # 받은 답변을 쳇 히스토리에 저장
                    st.session_state.chat_history = result['chat_history']
                response = result['answer'] # 받은 답변을 변수에 저장
                source_documents = result['source_documents'] # 참고한 문서를 변수에 저장

                st.markdown(response) # 마크다운을 이용해 화면에 출력
                with st.expander("참고 문서 확인"): # expander 화면을 접고 펴는 기능 정의
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history . 어시스턴트가 답변한 기록 저장
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text): # 토근 개수를 기준으로 텍스트를 스프릿하는 함수
    tokenizer = tiktoken.get_encoding("cl100k_base") # cl100k_base 이용해 틱토근 선언
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs): # 업로드된 파일을 텍스트로 변한하는 함수

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue()) # 파일에 저장
            logger.info(f"Uploaded {file_name}") # 로그로 확인
        
        # 각 업로든 된 파일 종류에 따라 스플릿 하기
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents) # 더규먼트 목록에 리스트로 저장
    return doc_list


def get_text_chunks(text): # 여러개의 청크로 스플릿 하는 함수
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, # 청크 사이즈
        chunk_overlap=100, # 겹치는 부분
        length_function=tiktoken_len # 청크를 세는 기준
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks): # 받은 청크를 벡터화
    embeddings = HuggingFaceEmbeddings( # HuggingFaceEmbeddings 의 임베딩 사용
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'}, # 스트림서버는 GPU가 없으므로 CPU로
                                        encode_kwargs={'normalize_embeddings': True} # 질문을 비교하기 위해
                                        )  
    # 벡터스어는  FAISS 사용                                  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key): # 위에서 선언한 것을 다 담기, 질문한 것을 다담아서 답변할수 있도록 하는 함수
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0) # LLM 모델. RAG시스템을 사용하기 때문에 temperature는 0
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'), # 메모리 저장. 이전 답변도 기억하게하는 역할, output_key='answer'는 답변만을 기억하라는 뜻
            get_chat_history=lambda h: h, # h: h 메모리가 들어오는 그대로 히스토리에 넣겠다.
            return_source_documents=True, # LLM 이 참고한 문서를 출력하라
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__': # 이함수를 실행하면 main함수를 실행해라
    main()
