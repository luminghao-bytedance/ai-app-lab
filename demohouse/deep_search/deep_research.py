import asyncio

from jinja2 import Template
from pydantic import BaseModel, Field
from typing_extensions import Optional

from arkitect.core.component.llm import BaseChatLanguageModel
from arkitect.core.component.llm.model import ArkMessage
from arkitect.core.component.prompts import CustomPromptTemplate

from search_engine import SearchEngine
from search_engine.mock import MockSearchEngine
from search_engine.volc_bot import VolcBotSearchEngine
from prompt import DEEP_RESEARCH_PROMPT
from utils import get_current_date

import re


class DeepResearch(BaseModel):
    search_engine: SearchEngine = Field(default_factory=MockSearchEngine)
    endpoint_id: str = Field(default_factory="")
    template: Template = DEEP_RESEARCH_PROMPT

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    async def arun_research(self, query: str):
        references = ""
        while True:
            llm = BaseChatLanguageModel(
                endpoint_id=self.endpoint_id,
                template=CustomPromptTemplate(template=self.template),
                messages=[
                    ArkMessage(role="user", content=query)
                ],
            )
            output = await self.chat_and_print(llm, references, query)
            requery = self.check_query(output)
            if not requery:
                print("\n----最终回答----")
                print(output)
                break
            print(f'\n搜索：{requery}')
            search_result = await self.search_engine.asearch(requery)
            print(f'\n搜索结果：{search_result}')
            references = references + search_result.raw_content
        return references

    async def chat_and_print(self, llm: BaseChatLanguageModel, references: str, query: str) -> str:
        thinking = False
        final_output = ""
        async for chunk in llm.astream(
                reference=references,  # pass the search result to prompt template
                question=query,
                meta_info=f"当前时间：{get_current_date()}"
        ):
            if chunk.choices[0].delta.reasoning_content:
                if not thinking:
                    print('\n开始思考...')
                    thinking = True
                print(chunk.choices[0].delta.reasoning_content, end="")
            if chunk.choices[0].delta.content:
                if thinking:
                    print('\n开始回答...')
                    thinking = False
                print(chunk.choices[0].delta.content, end="")
                final_output = final_output + chunk.choices[0].delta.content
        return final_output

    def check_query(self, output: str) -> Optional[str]:
        match = re.search(r"我需要搜索(.*)", output)
        if match:
            return match.group(1).strip()
        return None


if __name__ == '__main__':
    dr = DeepResearch(
        search_engine=VolcBotSearchEngine(
            bot_id="{botid}",
            api_key="apiKey"
        ),
        endpoint_id="epID"
    )

    asyncio.run(dr.arun_research('查一下2024国产手机的价格表，并选择其中价格最便宜的手机，搜索一篇关于它的评测文章总结内容'))
