from volcenginesdkarkruntime import Ark

client = Ark(base_url="http://localhost:8888/api/v3")


def main():
    # stream run
    stream_resp = client.chat.completions.create(
        model="ep-1234",  # useless, only for validation
        messages=[
            {
                "role": "user",
                "content": "帮我查一下2024年11月上市的智能手机的价格，并给出一篇有关其中最便宜的一款的网络评测",
            }
        ],
        stream=True,
    )

    thinking = False

    for chunk in stream_resp:
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
    main()
