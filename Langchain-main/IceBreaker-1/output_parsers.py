from typing import Dict , Any , List
from langchain_core.output_parsers import pydanticOutputParser
from pydantic import BaseModel , Field

class Summary(BaseModel):
    summary : str = Field(description="summary")
    facts : List[str] = Field(description="facts")

    def to_dict(self) -> Dict[str , Any]:
        return {"summary":self.summary, "facts":self.facts}

summary_parser = pydanticOutputParser(pydantic_object=Summary)