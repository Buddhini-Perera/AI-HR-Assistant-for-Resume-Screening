from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from memory import get_memory

def build_agent(resume_tool):
    memory = get_memory()
    llm = ChatOpenAI(temperature=0)

    tools = [
        Tool(
            name="RankCandidates",
            func=lambda x: resume_tool.rank_candidates(),
            description="Ranks resumes based on job description relevance."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=True
    )
    return agent
