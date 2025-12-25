from datetime import datetime, timedelta
from google_play_scraper import reviews, Sort
from langchain.chat_models import ChatOpenAI
import json, os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

APP_ID = "in.swiggy.android"
MASTER_FILE = "data/topics_master.json"
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create master topic file if not exists
if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists(MASTER_FILE):
    json.dump([], open(MASTER_FILE, 'w'))

# 1️⃣ Fetch Reviews for a Given Day
def fetch_reviews(date):
    result, _ = reviews(APP_ID, lang="en", country="in", sort=Sort.NEWEST, count=2000)
    return [{"date": str(r['at'].date()), "review": r['content']}
            for r in result if str(r['at'].date()) == date]

# 2️⃣ Extract Topic using LLM
def extract_topic(text):
    prompt = f"""
    You are an AI that categorizes Swiggy app reviews.
    Extract a short topic (2-4 words maximum) for this review.

    Review: "{text}"

    Return ONLY JSON:
    {{
      "topic": "topic name",
      "type": "issue/request/feedback",
      "reason": "why you chose this topic"
    }}
    """
    try:
        return json.loads(llm.predict(prompt))
    except:
        return None

# 3️⃣ Normalize topics using semantic similarity
def normalize(topic):
    master = json.load(open(MASTER_FILE))
    new_vec = model.encode(topic)

    for t in master:
        if util.cos_sim(new_vec, model.encode(t)) > 0.75:
            return t  # map to existing topic

    # Otherwise add as new topic
    master.append(topic)
    json.dump(master, open(MASTER_FILE, 'w'), indent=4)
    return topic

# 4️⃣ Build Trend Table
def build_trend(records, dates):
    table = defaultdict(lambda: {d:0 for d in dates})
    for r in records:
        table[r['topic']][r['date']] += 1
    return table

# 5️⃣ Export CSV
def export(table):
    df = pd.DataFrame.from_dict(table, orient="index")
    df.to_csv("output/swiggy_trend.csv")
    print("\n📁 Trend Report Generated → output/swiggy_trend.csv")
    return df

# 🚀 Main Execution
def main():
    today = datetime.today()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, -1, -1)]
    
    final = []

    for d in dates:
        print(f"\n🔎 Fetching reviews for: {d}")
        day = fetch_reviews(d)
        for r in day:
            topic_obj = extract_topic(r['review'])
            if topic_obj:
                topic_obj["topic"] = normalize(topic_obj["topic"])
                topic_obj["date"] = d
                final.append(topic_obj)

    table = build_trend(final, dates)
    export(table)

    print("\n🎉 Completed! Submit 'swiggy_trend.csv' from output folder.")

if __name__ == "__main__":
    main()

