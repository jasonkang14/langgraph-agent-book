# LangGraph 에이전트 구현

이 저장소는 LangGraph를 활용한 에이전트 구현에 대한 실습 코드를 포함하고 있습니다.

## 목차

### Chapter 1: 개발 전 필요한 배경지식
- 1.1 OpenAI 임베딩
- 1.2 OpenAI 임베딩 유사도 검색
- 1.3 Upstage 임베딩

### Chapter 2: LangChain을 활용한 RAG 파이프라인 구성
- 2.1 LLM 답변 생성
- 2.2 ChatOllama
- 2.3 ChatOllama 프롬프트 구체화
- 2.4 ChatHuggingFace
- 2.5 ChatHuggingFace 양자화
- 2.6 Chroma를 활용한 RAG 파이프라인 구축
- 2.7 데이터 전처리 후 RAG 파이프라인 구축
- 2.8 분할정복
- 2.9 LCEL
- 2.10 경량 모델(sLM) 전환 및 활용 방법
- 2.11 LangChain 없이 RAG 파이프라인 구현
- 2.12 Pinecone을 활용한 RAG 파이프라인 구축

### Chapter 3: LangGraph를 활용한 에이전트 구현
- 3.1 LangGraph로 구현하는 워크플로
- 3.2 LangGraph로 구현하는 에이전트
- 3.3 py-zerox를 활용한 데이터 전처리
- 3.4 소득세법 에이전트
- 3.5 Agent Orchestration을 위한 슈퍼바이저
- 3.6 MCP를 활용한 에이전트 구현

### Chapter 4: LangSmith를 활용한 LLM Evaluation
- 4.1 Evaluation을 위한 Dataset 생성
- 4.2 LLM Judge를 활용한 Evaluation

## 환경 설정

이 저장소의 예제 코드를 실행하기 위해서는 다음 단계를 따라주세요:

1. uv 설치 (아직 설치하지 않은 경우 [공식문서](https://github.com/astral-sh/uv)를 참고해주세요)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. 저장소 클론
```bash
git clone https://github.com/jasonkang14/langgraph-book.git
cd langgraph-book
```

3. 의존성 설치
```bash
uv sync
```

4. 각 챕터의 Jupyter Notebook을 순서대로 실행하며 실습을 진행해주세요.

## 주의사항

- 일부 코드는 API 키가 필요할 수 있습니다.
- 각 챕터의 실습 코드는 독립적으로 실행 가능합니다.
- 실습 전 필요한 패키지 설치를 확인해주세요.
