import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
import os
from openai import OpenAI
from crewai import Agent, Task, Crew

# Load API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------- AGENTS -----------------
fetcher_agent = Agent(
    role="Tech News Fetcher",
    goal="Fetch latest technology news headlines and links from a news website.",
    backstory="An expert at gathering trending tech news from reliable sources.",
)

summarizer_agent = Agent(
    role="Summarizer",
    goal="Summarize tech news headlines and links into short, crisp summaries.",
    backstory="Expert at summarizing complex info into engaging summaries."
)

# ----------------- TASKS -----------------

def fetch_news():
    try:
        print("🔎 Fetching latest news...")
        # Using Hacker News API (always stable)
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        top_ids = requests.get(url, timeout=10).json()[:5]  # Top 5 news

        headlines = []
        for story_id in top_ids:
            item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            item = requests.get(item_url, timeout=10).json()
            if item and "title" in item:
                headlines.append({
                    "title": item["title"],
                    "url": item.get("url", f"https://news.ycombinator.com/item?id={story_id}")
                })

        return headlines

    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return []


def summarize_news(news_list):
    summaries = []
    for n in news_list:
        prompt = f"Summarize the following tech news headline in 1-2 sentences:\n\nTitle: {n['title']}\nURL: {n['url']}"
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.7
            )
            ai_summary = response.choices[0].message.content.strip()
        except Exception as e:
            ai_summary = f"Summary unavailable. ({e})"

        summaries.append({
            "title": n["title"],
            "url": n["url"],
            "summary": ai_summary
        })
    return summaries

# ----------------- CREW & EXECUTION -----------------
def run_crew_ai():
    try:
        # Define Tasks with expected_output (now required)
        fetch_task = Task(
            description="Fetch top 5 latest tech news with title and URL.",
            agent=fetcher_agent,
            expected_output="List of dicts with 'title' and 'url' keys",
            async_execution=False
        )

        summarize_task = Task(
            description="Summarize fetched tech news into 1-2 sentences.",
            agent=summarizer_agent,
            expected_output="List of dicts with 'title', 'url', and 'summary' keys",
            async_execution=False
        )

        # Create the Crew and run it
        crew = Crew(
            agents=[fetcher_agent, summarizer_agent],
            tasks=[fetch_task, summarize_task]
        )

        # Manually run the steps using our Python functions
        news = fetch_news()
        if not news:
            print("⚠️ No news fetched. Check website availability.")
            return

        summaries = summarize_news(news)

        # Save results only if valid
        if summaries:
            data = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "summaries": summaries
            }
            with open("summaries.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print("✅ Summaries saved to summaries.json")
        else:
            print("⚠️ Summarization failed.")

    except Exception as e:
        print(f"❌ Error running CrewAI: {e}")


if __name__ == "__main__":
    run_crew_ai()
