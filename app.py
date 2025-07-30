from flask import Flask, render_template
import json
import os

app = Flask(__name__)

@app.route("/")
def index():
    if not os.path.exists("summaries.json") or os.path.getsize("summaries.json") == 0:
        return "No summaries found. Please run crew_ai_runner.py first."

    try:
        with open("summaries.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return "Error: summaries.json is corrupted. Please re-run crew_ai_runner.py."

    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
