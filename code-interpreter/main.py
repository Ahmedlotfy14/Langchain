from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent , AgentExecutor
from langchain_experimental.tools import PythonREPLTool

load_dotenv()


def main():
    print("Hello World")

    instructions = """You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question. 
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        tools=tools,
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    )

    python_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    python_agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 15 QRcodes
                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
        }
    )

    csv_agent_executor : AgentExecutor =create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        csv_path="episode_info.csv",
        verbose=True,
    )

    csv_agent_executor.invoke(
        input={"input": "how many columns are there in file episode_info.csv"}
    )
    csv_agent_executor.invoke(
        input={
            "input": "print the seasons by ascending order of the number of episodes they have"
        }
    )


    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input":original_prompt})

    tools = [
        Tool(name="Python_Agent",
             func=python_agent_executor_wrapper,
             description = """useful when you need to transform natural language to python and execute the python code,
                              returning the results of the code execution
                              DOES NOT ACCEPT CODE AS INPUT"""),
        Tool(name="CSV_Agent",
             func=csv_agent_executor.invoke,
             description="""useful when you need to answer question over episode_info.csv file,
                             takes an input the entire question and returns the answer after running pandas calculations""",
             ),
    ]
    prompt = base_prompt.partial(instructions="")
    all_agent = create_react_agent(
        prompt=prompt,
        tools=tools,
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    )
    all_agent_executor = AgentExecutor(agent=all_agent, tools=tools, verbose=True)

    print(
        all_agent_executor.invoke(
            {
                "input": "which season has the most episodes?",
            }
        )
    )

    print(
        all_agent_executor.invoke(
            {
                "input": "Generate and save in current working directory 15 qrcodes that point to `www.udemy.com/course/langchain`",
            }
        )
    )


if __name__ == "__main__":
    main()
