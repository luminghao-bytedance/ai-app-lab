from datetime import datetime
from typing import List

from arkitect.core.component.llm.model import ArkMessage, ArkChatCompletionChunk


def cast_content_to_reasoning_content(chunk: ArkChatCompletionChunk) -> ArkChatCompletionChunk:
    new_chunk = ArkChatCompletionChunk(**chunk.__dict__)
    new_chunk.choices[0].delta.reasoning_content = chunk.choices[0].delta.content
    new_chunk.choices[0].delta.content = ""
    return new_chunk


def get_last_message(messages: List[ArkMessage], role: str):
    """Finds the last ArkMessage of a specific role, given the role."""
    for message in reversed(messages):
        if message.role == role:
            return message
    return None


def get_current_date() -> str:
    return datetime.now().strftime("%Y年%m月%d日")
