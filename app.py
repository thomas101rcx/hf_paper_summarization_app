import os
from datetime import datetime
import requests
import pdfplumber
import io
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import streamlit as st
import google.generativeai as genai
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from typing import Union  

######################################################
#                   CONFIGURATION                    #
######################################################

# Load API keys from Streamlit secrets
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# Configure Google Generative AI
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Pinecone setup
pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = "ai-daily-papers"
if INDEX_NAME not in [idx["name"] for idx in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=INDEX_NAME, dimension=512, spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pinecone_client.Index(INDEX_NAME)

# Embedding model for Pinecone
embedding_model = pipeline(
    "feature-extraction", model="jinaai/jina-embeddings-v2-small-en", 
    tokenizer="jinaai/jina-embeddings-v2-small-en"
)

######################################################
#               PDF PROCESSING FUNCTIONS             #
######################################################

def get_pdf_link(huggingface_url: str) -> Union[str, None]:  # Changed to Union[str, None]
    """Extract PDF link from a Hugging Face paper page."""
    try:
        response = requests.get(huggingface_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "View PDF" in link.text or href.endswith(".pdf"):
                return urljoin(huggingface_url, href) if not href.startswith("http") else href
        return None
    except requests.RequestException as e:
        print(f"Error fetching PDF link from {huggingface_url}: {e}")
        return None

def extract_text_from_pdf(pdf_url: str) -> Union[str, None]:  # Also update this one
    """Download and extract text from a PDF in memory."""
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    except (requests.RequestException, Exception) as e:
        print(f"Error extracting text from {pdf_url}: {e}")
        return None

def summarize_text(text: str, overall: bool = False) -> str:
    """Generate a summary of text using Gemini model."""
    date = datetime.now().strftime("%Y %B %d")
    prompt = (
        f"{'Provide an overall summary of multiple papers' if overall else 'Summarize this paper'} "
        f"for a Data Scientist to quickly understand. Date: {date}.\n\n"
        f"{text}"
    )
    return model.generate_content(prompt).text

######################################################
#                PAPER FETCHING & MEMO               #
######################################################

def fetch_daily_papers(date_str: str, top_n: int = 10) -> list[str]:
    """Fetch top N papers from Hugging Face by upvotes for a given date."""
    url = f"https://huggingface.co/papers?date={date_str}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        papers = []
        for title_elem in soup.find_all("h3"):
            link_elem = title_elem.find_parent("a") or title_elem.find("a")
            if not link_elem or not link_elem.get("href", "").startswith("/papers/"):
                continue
            full_url = urljoin("https://huggingface.co", link_elem["href"])
            if "#community" in full_url:
                continue
            upvotes = int(title_elem.find_previous(string=lambda t: t.strip().isdigit()) or 0)
            papers.append({"URL": full_url, "Upvotes": upvotes})
        
        return [p["URL"] for p in pd.DataFrame(papers).sort_values("Upvotes", ascending=False).head(top_n).to_dict("records")]
    except requests.RequestException as e:
        print(f"Error fetching papers from {url}: {e}")
        return []

def generate_daily_memo(paper_summaries: list[dict]) -> str:
    """Generate a daily memo with overall summary and individual summaries."""
    # Combine all summaries for an overall summary
    all_text = "\n\n".join(p["summary"] for p in paper_summaries)
    overall_summary = summarize_text(all_text, overall=True)
    
    # Format memo with overall summary followed by individual summaries
    memo = f"## Overall Summary\n{overall_summary}\n\n## Individual Paper Summaries\n"
    for paper in paper_summaries:
        memo += f"### [{paper['link']}]({paper['link']})\n{paper['summary']}\n\n"
    return memo

######################################################
#                PINECONE FUNCTIONS                  #
######################################################

def store_in_pinecone(text: str, metadata: dict) -> None:
    """Store text embeddings and metadata in Pinecone."""
    embeddings = embedding_model(text)[0][0]  # Extract first vector (512 dim)
    if len(embeddings) != 512:
        raise ValueError(f"Expected 512 dimensions, got {len(embeddings)}")
    index.upsert([(metadata["id"], embeddings, metadata)])

def load_pinecone_items() -> list[dict]:
    """Load all items from Pinecone, sorted by timestamp."""
    results = index.query(vector=[0.0] * 512, top_k=1000, include_metadata=True, filter={})
    return sorted((m["metadata"] for m in results.get("matches", []) if "type" in m["metadata"]), 
                  key=lambda x: x.get("timestamp", ""))

def group_by_date(items: list[dict]) -> dict:
    """Group Pinecone items by date."""
    data_by_date = {}
    for item in items:
        date = item.get("date")
        if not date:
            continue
        if date not in data_by_date:
            data_by_date[date] = {"memo_content": None, "memo_item": None, "chats": []}
        if item["type"] == "daily_memo":
            data_by_date[date]["memo_content"] = item.get("content", "")
            data_by_date[date]["memo_item"] = item
        elif item["type"] == "chat":
            data_by_date[date]["chats"].append(item)
    for date in data_by_date:
        data_by_date[date]["chats"].sort(key=lambda x: x.get("timestamp", ""))
    return data_by_date

######################################################
#                 STREAMLIT APP                      #
######################################################

def check_and_generate_memo(today: str) -> None:
    """Generate and store memo for today if it doesn’t exist."""
    if index.query(vector=[0.0] * 512, filter={"type": "daily_memo", "date": today}, top_k=1).get("matches"):
        return
    
    links = fetch_daily_papers(today)
    if not links:
        return
    
    paper_summaries = [
        {"link": link, "summary": summary}
        for link in links
        if (pdf_url := get_pdf_link(link)) and (text := extract_text_from_pdf(pdf_url)) and (summary := summarize_text(text))
    ]
    
    if paper_summaries:
        memo = generate_daily_memo(paper_summaries)
        store_in_pinecone(memo, {
            "id": f"memo-{today}", "type": "daily_memo", "date": today, 
            "timestamp": datetime.now().isoformat(), "content": memo
        })

def sidebar_ui(data_by_date: dict) -> None:
    """Render sidebar with date selection."""
    st.sidebar.title("Conversation Dates")
    if not data_by_date:
        st.sidebar.write("No data found.")
        return
    dates = sorted(data_by_date.keys(), reverse=True)
    st.session_state.setdefault("selected_date", dates[0])
    st.session_state["selected_date"] = st.sidebar.selectbox("Select a date:", dates, 
                                                             index=dates.index(st.session_state["selected_date"]))

def main_chat_ui(data_by_date: dict) -> None:
    """Display memo, chat history, and handle new questions."""
    selected_date = st.session_state.get("selected_date")
    if not selected_date or selected_date not in data_by_date:
        st.title("No Date Selected")
        return
    
    date_data = data_by_date[selected_date]
    st.title(f"Conversation for {selected_date}")
    
    # Display memo with overall and individual summaries
    if memo := date_data.get("memo_content"):
        with st.chat_message("assistant"):
            st.markdown(f"**Daily Memo for {selected_date}**")
            st.markdown(memo)
    else:
        with st.chat_message("assistant"):
            st.markdown(f"*(No memo for {selected_date}.)*")
    
    # Display chat history
    for chat in date_data.get("chats", []):
        if q := chat.get("question"):
            with st.chat_message("user"):
                st.write(q)
        if a := chat.get("answer"):
            with st.chat_message("assistant"):
                st.write(a)
    
    # Handle new user input
    if new_input := st.chat_input("Ask about today’s memo..."):
        prompt = f"{memo or ''}\n\n" + "\n".join(f"User: {c['question']}\nAssistant: {c['answer']}" 
                                                 for c in date_data["chats"]) + f"\nUser: {new_input}\nAssistant:"
        answer = model.generate_content(prompt).text
        
        with st.chat_message("user"): st.write(new_input)
        with st.chat_message("assistant"): st.write(answer)
        
        chat_id = f"chat-{selected_date}-{int(datetime.now().timestamp())}"
        store_in_pinecone(new_input + " " + answer, {
            "id": chat_id, "type": "chat", "date": selected_date, 
            "timestamp": datetime.now().isoformat(), "question": new_input, "answer": answer
        })
        date_data["chats"].append({"question": new_input, "answer": answer, "timestamp": datetime.now().isoformat()})
        st.rerun()

def main():
    """Main app logic."""
    today = datetime.now().strftime("%Y-%m-%d")
    check_and_generate_memo(today)
    
    data_by_date = group_by_date(load_pinecone_items())
    st.session_state["data_by_date"] = data_by_date
    
    sidebar_ui(data_by_date)
    main_chat_ui(data_by_date)

if __name__ == "__main__":
    main()