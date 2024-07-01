import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import warnings
import shutil
import atexit

# 임시 디렉토리 설정
temp_dir = tempfile.mkdtemp()
os.environ['LANGCHAIN_CACHE'] = os.path.join(temp_dir, 'langchain.db')

# CrewAI와 관련 라이브러리 임포트 시도
try:
    from crewai import Agent, Task, Crew, Process
except ImportError as e:
    st.error(f"CrewAI 라이브러리가 설치되지 않았습니다: {e}")
    st.code("pip install git+https://github.com/joaomdmoura/crewAI.git")
    st.stop()

# LangChain 라이브러리 임포트 시도
try:
    from langchain.chat_models import ChatOpenAI
except ImportError as e:
    st.error(f"LangChain 관련 라이브러리가 설치되지 않았습니다: {e}")
    st.code("pip install langchain langchain-openai")
    st.stop()

# 경고 무시 설정
warnings.filterwarnings("ignore")

# .env 파일에서 환경 변수 로드 시도
try:
    load_dotenv()
except Exception as e:
    st.warning(f".env 파일을 불러오는 데 실패했습니다: {e}")
    st.info("환경 변수를 직접 설정하거나 .env 파일을 생성해주세요.")

# API 키 설정
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "default_serper_api_key")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "default_openai_api_key")

# 도구 초기화 (도구 없이 진행)
search_tool = None
web_search_tool = None

# LLM 모델 선택 함수
def get_llm(model_name, temperature=0.7):
    return ChatOpenAI(model_name=model_name, temperature=temperature)

# 에이전트 생성 함수
def create_agent(role, goal, backstory, verbose, allow_delegation, llm):
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=verbose,
        allow_delegation=allow_delegation,
        llm=llm
    )
    return agent

# 에이전트 정의
researcher = create_agent(
    role='Researcher',
    goal='주어진 작업에 대한 초기 초안 작성',
    backstory="당신은 열정적인 주니어 연구원입니다. 학습과 기여에 열중하며 태스크를 위임할 수 없습니다.",
    verbose=True,
    allow_delegation=False,
    llm=get_llm("gpt-3.5-turbo-16k", 0.7)
)

analyst = create_agent(
    role='Analyst',
    goal='초기 초안 분석 및 개선',
    backstory="당신은 다른 사람들의 작업을 개선하고 강화하는 경험 많은 분석가입니다.",
    verbose=True,
    allow_delegation=False,
    llm=get_llm("gpt-4", 0.5)
)

critic = create_agent(
    role='Critic',
    goal='분석 결과를 비판적으로 평가하고 피드백 제공',
    backstory="당신은 개선이 필요한 영역을 식별하고 선임 관리자에게 보고하는 날카로운 비평가입니다.",
    verbose=True,
    allow_delegation=False,
    llm=get_llm("gpt-4", 0.3)
)

manager = create_agent(
    role='Senior Manager',
    goal='모든 입력을 종합하여 최종 결과물 생성',
    backstory="당신은 최종 결정을 내리고 사용자에게 결과를 전달하는 경험 많은 관리자입니다.",
    verbose=True,
    allow_delegation=True,
    llm=get_llm("gpt-3.5-turbo-16k", 0.5)
)

# Streamlit UI 설정
st.title("AI 팀 워크플로우")

# 대화 로그 저장을 위한 리스트 초기화
chat_log = []

# 사용자 입력 받기
user_input = st.text_input("작업을 입력하세요 (예: '병원 마케팅 전략을 논의해봅시다'):")

if user_input:
    chat_log.append(f"사용자 입력: {user_input}")

    # Crew 생성
    crew = Crew(
        agents=[researcher, analyst, critic, manager],
        tasks=[
            Task(description=f"다음 작업에 대한 초기 초안 작성: {user_input}", agent=researcher, expected_output="초안 작성"),
            Task(description="연구원이 제공한 초기 초안을 분석하고 개선하세요.", agent=analyst, expected_output="개선된 초안"),
            Task(description="분석 결과를 비판적으로 평가하고 선임 관리자에게 피드백을 제공하세요.", agent=critic, expected_output="피드백"),
            Task(description="모든 입력을 종합하여 사용자를 위한 최종 결과물을 생성하세요.", agent=manager, expected_output="최종 결과물")
        ],
        verbose=2,
        process=Process.sequential
    )

    # Crew 작업 실행
    try:
        with st.spinner('AI 팀이 작업 중입니다...'):
            result = crew.kickoff()
        
        # 결과 표시
        st.markdown("## 최종 결과")
        st.markdown(result)
        chat_log.append(f"최종 결과: {result}")

        # 추가 질문 옵션
        user_question = st.text_input("결과에 대해 질문이 있으신가요?:")
        if user_question:
            chat_log.append(f"사용자 질문: {user_question}")
            with st.spinner('답변을 생성 중입니다...'):
                response = manager.run(f"이전 작업에 대한 다음 질문에 답변하세요: {user_question}")
            st.markdown("## 답변")
            st.markdown(response)
            chat_log.append(f"답변: {response}")

    except Exception as e:
        st.error(f"작업 수행 중 오류가 발생했습니다: {e}")
        chat_log.append(f"오류: {e}")

# 대화 로그를 스트림릿 화면에 출력
if chat_log:
    st.markdown("## 미팅 로그")
    for entry in chat_log:
        st.markdown(entry)

# 미팅 로그를 파일로 저장하여 다운로드할 수 있는 버튼 추가
def save_log():
    with open(os.path.join(temp_dir, "meeting_log.txt"), "w", encoding="utf-8") as f:
        for entry in chat_log:
            f.write(entry + "\n")
    return os.path.join(temp_dir, "meeting_log.txt")

if st.button("미팅 로그 다운로드"):
    log_file_path = save_log()
    with open(log_file_path, "rb") as f:
        st.download_button(
            label="미팅 로그 다운로드",
            data=f,
            file_name="meeting_log.txt",
            mime="text/plain"
        )

# 종료 버튼 추가
if st.button("종료"):
    st.stop()

# 긴 대화를 위한 스크롤 위젯 추가
st.markdown(
    """
    <style>
        .stApp {
            max-height: 100vh;
            overflow-y: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 임시 디렉토리 정리
# 프로그램 종료 시 안전하게 임시 디렉토리를 정리합니다.
def cleanup_temp_dir():
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        st.error(f"임시 디렉토리 정리 중 오류가 발생했습니다: {e}")

# 프로그램 종료 시 임시 디렉토리 정리 등록
atexit.register(cleanup_temp_dir)
