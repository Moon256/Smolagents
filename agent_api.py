from flask import Flask, request, jsonify
from smolagents import CodeAgent, TransformersModel, FinalAnswerTool
from tools import (
    fetch_latest_news_titles_and_urls,
    extract_news_article_content,
    summarize_news,
    classify_topic
)

app = Flask(__name__)

model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
    torch_dtype="bfloat16",
    trust_remote_code=True,
    max_new_tokens=2000
)

agent = CodeAgent(
    model=model,
    tools=[
        fetch_latest_news_titles_and_urls,
        extract_news_article_content,
        summarize_news,
        classify_topic,
        FinalAnswerTool()
    ],
    additional_authorized_imports=["requests", "bs4"],
    name="news_agent"
)

@app.route("/run-agent", methods=["POST"])
def run_agent():
    data = request.get_json()
    task = data.get("task", "")
    result = agent.run(task)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(port=8000)
