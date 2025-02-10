import asyncio
import logging
from typing import Dict, List, AsyncIterable

from jinja2 import Template
from pydantic import BaseModel, Field
from typing_extensions import Optional

from arkitect.core.component.llm import BaseChatLanguageModel
from arkitect.core.component.llm.model import (
    ArkMessage,
    ArkChatRequest,
    ArkChatResponse,
    ArkChatCompletionChunk,
)
from arkitect.core.component.prompts import CustomPromptTemplate
from arkitect.telemetry.logger import INFO

from search_engine import SearchEngine
from search_engine.mock import MockSearchEngine
from search_engine.volc_bot import VolcBotSearchEngine
from prompt import (
    DEFAULT_INTENTION_PROMPT,
    DEFAULT_PLANNING_PROMPT,
    DEFAULT_QUERY_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
)
from utils import get_current_date, cast_content_to_reasoning_content

import re

"""
References is using to store the references searched so far
"""


class References(BaseModel):
    """
    key: query
    values: list of searched references for this query
    """

    ref_dict: Dict[str, List[str]] = Field(default_factory=dict)

    def add_reference(self, query: str, references: List[str]) -> None:
        if query not in self.ref_dict:
            self.ref_dict[query] = references.copy()
        else:
            extended_references = self.ref_dict.get(query, [])
            extended_references.extend(references)
            self.ref_dict[query] = extended_references

    def to_plaintext(self) -> str:
        output = ""

        for key, value in self.ref_dict.items():
            output += f"\n查询: {key}"
            output += "\n相关资料:"
            output += "\n".join(value)

        return output


"""
DeepResearch 
"""


class DeepResearch(BaseModel):
    search_engine: SearchEngine = Field(default_factory=MockSearchEngine)
    deepseek_r1_endpoint_id: str = Field(default_factory="")
    doubao_endpoint_id: str = Field(default_factory="")
    planning_template: Template = DEFAULT_PLANNING_PROMPT
    intention_template: Template = DEFAULT_INTENTION_PROMPT
    query_rewrite_template: Template = DEFAULT_QUERY_PROMPT
    summary_template: Template = DEFAULT_SUMMARY_PROMPT
    max_planning_rounds: int = 5

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    async def arun_deep_research(
        self, request: ArkChatRequest, question: str
    ) -> ArkChatResponse:
        references = References()
        buffered_reasoning_content = ""

        # 1. run reasoning
        reasoning_stream = self.astream_reasoning(
            request=request,
            question=question,
            references=references,
        )

        async for reasoning_chunk in reasoning_stream:
            buffered_reasoning_content += reasoning_chunk.choices[
                0
            ].delta.reasoning_content

        # 2. run summary
        # append the reasoning content as an assistant message to help summary
        request.messages.append(
            ArkMessage(
                role="assistant",
                content=buffered_reasoning_content,
            )
        )
        resp = await self.arun_summary(
            request=request, question=question, references=references
        )
        # append the reasoning buffer
        resp.choices[0].delta.reasoning_content = (
            buffered_reasoning_content + resp.choices[0].delta.reasoning_content
        )
        return resp

    async def astream_deep_research(
        self, request: ArkChatRequest, question: str
    ) -> AsyncIterable[ArkChatCompletionChunk]:
        references = References()
        buffered_reasoning_content = ""

        # 1. stream reasoning
        reasoning_stream = self.astream_reasoning(
            request=request,
            question=question,
            references=references,
        )

        async for reasoning_chunk in reasoning_stream:
            buffered_reasoning_content += reasoning_chunk.choices[
                0
            ].delta.reasoning_content
            yield reasoning_chunk

        # 2. stream summary
        # append the reasoning content as an assistant message to help summary
        request.messages.append(
            ArkMessage(
                role="assistant",
                content=buffered_reasoning_content,
            )
        )
        summary_stream = self.astream_summary(
            request=request,
            question=question,
            references=references,
        )

        async for summary_chunk in summary_stream:
            yield summary_chunk

    async def astream_reasoning(
        self, request: ArkChatRequest, question: str, references: References
    ) -> AsyncIterable[ArkChatCompletionChunk]:
        planned_rounds = 0
        while planned_rounds < self.max_planning_rounds:
            planned_rounds += 1

            llm = BaseChatLanguageModel(
                endpoint_id=self.deepseek_r1_endpoint_id,
                template=CustomPromptTemplate(template=self.intention_template),
                messages=request.messages,
            )

            stream = llm.astream(
                reference=references.to_plaintext(),  # pass the search result to prompt template
                question=question,
                meta_info=f"当前时间：{get_current_date()}",
            )

            cot_result = ""

            async for chunk in stream:
                if chunk.choices[0].delta.reasoning_content:
                    yield chunk
                elif chunk.choices[0].delta.content:
                    cot_result += chunk.choices[0].delta.content
                    # cast the content into reasoning content
                    yield cast_content_to_reasoning_content(chunk)

            # rewrite query
            llm = BaseChatLanguageModel(
                endpoint_id=self.doubao_endpoint_id,
                template=CustomPromptTemplate(template=self.query_rewrite_template),
                messages=request.messages,
            )
            query_result = ""
            stream = llm.astream(
                reference=references.to_plaintext(),  # pass the search result to prompt template
                question=question,
                cot=cot_result,
                meta_info=f"当前时间：{get_current_date()}",
            )
            async for chunk in stream:
                if chunk.choices[0].delta.reasoning_content:
                    yield chunk
                elif chunk.choices[0].delta.content:
                    query_result += chunk.choices[0].delta.content
                    # cast the content into reasoning content
                    yield cast_content_to_reasoning_content(chunk)

            if "无需搜索" in query_result:
                INFO("intention finished")
                break
            else:
                INFO(f"searching: {query_result}")
                keywords = query_result.split("|")
                content = ""
                for keyword in keywords:
                    INFO(f"searching: {keyword}")
                    search_result = await self.search_engine.asearch(keyword.strip())
                    content += search_result.raw_content
                references.add_reference(query=query_result, references=[content])

    async def arun_summary(
        self, request: ArkChatRequest, question: str, references: References
    ) -> ArkChatResponse:
        llm = BaseChatLanguageModel(
            endpoint_id=self.deepseek_r1_endpoint_id,
            template=CustomPromptTemplate(template=self.summary_template),
            messages=request.messages,
        )

        return await llm.arun(
            reference=references.to_plaintext(),
            question=question,
            meta_info=f"当前时间：{get_current_date()}",
        )

    async def astream_summary(
        self, request: ArkChatRequest, question: str, references: References
    ) -> AsyncIterable[ArkChatCompletionChunk]:
        llm = BaseChatLanguageModel(
            endpoint_id=self.deepseek_r1_endpoint_id,
            template=CustomPromptTemplate(template=self.summary_template),
            messages=request.messages,
        )

        stream = llm.astream(
            reference=references.to_plaintext(),
            question=question,
            meta_info=f"当前时间：{get_current_date()}",
        )

        async for chunk in stream:
            yield chunk

    @classmethod
    def check_query(cls, output: str) -> Optional[str]:
        match = re.search(r"我需要搜索(.*)", output)
        if match:
            return match.group(1).strip()
        return None


logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger(__name__)


async def main():
    dr = DeepResearch(
        search_engine=VolcBotSearchEngine(
            bot_id="bot-20250209103828-hcr48",
            api_key="yourkey",
        ),
        deepseek_r1_endpoint_id="deepseek-ep",
        doubao_endpoint_id="doubao-ep",
    )

    thinking = False
    async for chunk in dr.astream_deep_research(
        request=ArkChatRequest(
            model="test",
            messages=[
                ArkMessage(
                    role="user",
                    content="帮我查一下2024年11月上市的智能手机的价格，并给出一篇有关其中最便宜的一款的网络评测",
                )
            ],
        ),
        question="帮我查一下2024年11月上市的智能手机的价格，并给出一篇有关其中最便宜的一款的网络评测",
    ):
        if chunk.choices[0].delta.reasoning_content:
            if not thinking:
                print("\n----思考过程----\n")
                thinking = True
            print(chunk.choices[0].delta.reasoning_content, end="")
        elif chunk.choices[0].delta.content:
            if thinking:
                print("\n----输出回答----\n")
                thinking = False
            print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
    asyncio.run(main())
