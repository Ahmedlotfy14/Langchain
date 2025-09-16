from langchain_tavily import TavilySearch

def get_profile_url(name: str):
    search = TavilySearch()
    res = search.run(f"{name}")
    return res


