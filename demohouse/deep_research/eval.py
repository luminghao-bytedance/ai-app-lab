import asyncio
import openpyxl

from arkitect.core.component.llm.model import ArkChatRequest, ArkMessage
from deep_research_double import DeepResearch
from search_engine.volc_bot import VolcBotSearchEngine


async def get_answer(question):
    dr = DeepResearch(
        search_engine=VolcBotSearchEngine(
            bot_id="bot-20250209103828-hcr48",
            api_key="your key",
        ),
        deepseek_r1_endpoint_id="deepseek-ep",
        doubao_endpoint_id="doubao-ep",
    )

    print(question)
    result = ""
    thinking = False
    async for chunk in dr.astream_deep_research(
        request=ArkChatRequest(
            model="test", messages=[ArkMessage(role="user", content=question)]
        ),
        question=question,
    ):
        if chunk.choices[0].delta.reasoning_content:
            if not thinking:
                print("\n----思考过程----\n")
                result += "\n----思考过程----\n"
                thinking = True
            print(chunk.choices[0].delta.reasoning_content, end="")
            result += chunk.choices[0].delta.reasoning_content
        elif chunk.choices[0].delta.content:
            if thinking:
                print("\n----输出回答----\n")
                result += "\n----输出回答----\n"
                thinking = False
            print(chunk.choices[0].delta.content, end="")
            result += chunk.choices[0].delta.content

    return result


async def get_three_answers(question: str):
    ans1, ans2, ans3 = await asyncio.gather(
        get_answer(question), get_answer(question), get_answer(question)
    )
    return ans1, ans2, ans3


async def compute_answers(row_idx: int, question: str):
    # your logic that returns the three answers
    ans1, ans2, ans3 = await get_three_answers(question)
    return (row_idx, ans1, ans2, ans3)


def collect_questions(sheet):
    questions = {}
    row = 3  # First row of questions
    while True:
        question_cell = sheet[f"E{row}"]
        question = question_cell.value
        if question is None or str(question).strip() == "":
            break
        questions[row] = question
        row += 1
    return questions


async def main():
    workbook = openpyxl.load_workbook("./yali_dataset.xlsx")
    sheet = workbook.active

    questions = collect_questions(sheet)

    tasks = []
    max_concurrency = 3
    semophore = asyncio.Semaphore(max_concurrency)

    async def task_with_concurrency(row_idx, question):
        async with semophore:
            return await compute_answers(row_idx, question)

    for row_idx, question in questions.items():
        t = asyncio.create_task(task_with_concurrency(row_idx, question))
        tasks.append(t)
        break

    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
    for d in done:
        # each completed task returns a (row_idx, ans1, ans2, ans3)
        row_idx, ans1, ans2, ans3 = d.result()
        sheet[f"L{row_idx}"] = ans1
        sheet[f"M{row_idx}"] = ans2
        sheet[f"N{row_idx}"] = ans3

        workbook.save("my_questions.xlsx")


if __name__ == "__main__":
    asyncio.run(main())
