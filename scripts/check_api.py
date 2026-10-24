"""Check the local LangGraph API without shell-pipe encoding conversions."""

import argparse
import json
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srno", action="store_true", help="Run a retrieval question")
    parser.add_argument("--url", default="http://127.0.0.1:2024")
    args = parser.parse_args()
    question = (
        "SRNO 如何将神经算子用于图像超分辨？请根据论文解释，并引用来源和页码。"
        if args.srno else "你好，请告诉我你能回答哪些领域的问题？"
    )
    print("发送的问题：", question, flush=True)
    body = {
        "assistant_id": "retrieval_graph",
        "input": {"messages": [{"role": "user", "content": question}]},
    }
    request = urllib.request.Request(
        args.url.rstrip("/") + "/runs/wait",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            print("响应 Content-Type：", response.headers.get("Content-Type"))
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"))
        raise
    if "__error__" in result:
        raise RuntimeError(result["__error__"])
    print("\n路由：")
    print(json.dumps(result.get("router"), ensure_ascii=False, indent=2))
    print("\n回答：")
    print(result["messages"][-1]["content"])


if __name__ == "__main__":
    main()
