# mcp_server.py
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# 환경변수를 불러온다.
load_dotenv()

# MCP 서버 생성
mcp = FastMCP("house_tax")


# 덧셈 도구 추가
@mcp.tool(
    name="add",  # 도구의 이름 작성
    description="두 숫자를 더합니다",  # 도구의 설명(역할) 작성
)
def add(a: int, b: int) -> int:
    """두 숫자를 더합니다"""  # docstring은 남겨두어도 상관없다.
    return a + b

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# 환경변수를 불러온다.
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o')
small_llm = ChatOpenAI(model='gpt-4o-mini')

# 현재 공정시장가액비율 검색
def get_market_value_rate_search():
    """
    올해의 공정시장가액비율을 찾기 위해 웹 검색을 수행합니다.
    
    반환값:
        str: 현재 공정시장가액비율에 대한 정보가 포함된 검색 결과
    """
    # 검색 도구 초기화
    search = TavilySearch(
        include_answer=True
    )
    # 현재 연도의 공정시장가액비율을 검색한다.
    # datetime.now()로 현재 연도를 동적으로 가져와서 검색어에 포함한다.
    market_value_rate_search = search.invoke(f"{datetime.now().year}년도 공정시장가액비율은?")
    market_value_rate_search = market_value_rate_search['answer']
    return market_value_rate_search


@mcp.tool(
    name="get_market_value_rate",
    description="""사용자의 부동산 상황에 적용되는 공정시장가액비율을 결정합니다.
    
    이 도구는:
    1. 현재 공정시장가액비율 정보가 포함된 검색 결과를 사용
    2. 사용자의 특정 상황(보유 부동산 수, 부동산 가치)을 분석
    3. 적절한 공정시장가액비율을 백분율로 반환

    Args:
        question (str): 부동산 소유에 대한 사용자의 질문
        
    Returns:
        str: 공정시장가액비율 백분율 (예: '60%', '45%')
    """,
)
def get_market_value_rate(question: str) -> str:
    market_value_rate_search = get_market_value_rate_search()

    # 공정시장가액비율 추출을 위한 프롬프트 템플릿 정의
    market_value_rate_prompt = PromptTemplate.from_template("""아래 [Context]는 공정시장가액비율에 관한 내용입니다. 
    당신에게 주어진 공정시장가액비율에 관한 내용을 기반으로, 사용자의 상황에 대한 공정시장가액비율을 알려주세요.
    별도의 설명 없이 공정시장가액비율만 반환해주세요.

    [Context]
    {context}

    [Question]
    질문: {question}
    답변: 
    """)

    # 공정시장가액비율 계산을 위한 체인 정의
    market_value_rate_chain = (
        market_value_rate_prompt
        | small_llm
        | StrOutputParser()
    )
    market_value_rate = market_value_rate_chain.invoke({
        'context': market_value_rate_search, 
        'question': question
    })
    return market_value_rate


if __name__ == "__main__":
    mcp.run(transport="stdio")  # stdio 또는 streamable-http
