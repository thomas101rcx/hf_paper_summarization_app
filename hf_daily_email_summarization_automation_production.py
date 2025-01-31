import os
import re
import time
from datetime import datetime
import base64
import io
from typing import Dict, Any, List
import requests
import pdfplumber
import streamlit as st
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
import smtplib
# Google Auth / Gmail
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
# Transformers / Embeddings
from transformers import pipeline
# Google Generative AI
import google.generativeai as genai
# Pinecone
import pinecone
from langchain.vectorstores import Pinecone


######################################################
#                      CONFIG                        #
######################################################


os.environ["GEMINI_API_KEY"] =st.secrets["GEMINI_API_KEY"]
os.environ["PINECONE_API_KEY"] =st.secrets["PINECONE_API_KEY"]



SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

######################################################
#                GOOGLE GMAIL AUTH                   #
######################################################

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

######################################################
#               GMAIL FETCHING HELPERS               #
######################################################

def extract_links_from_email(raw_data):
    html_content = requests.utils.unquote(raw_data)
    soup = BeautifulSoup(html_content, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

def fetch_daily_papers_email(creds, query="Daily papers"):
    service = build('gmail', 'v1', credentials=creds)
    query = f"{query} from:daily_papers_digest@notifications.huggingface.co"
    results = service.users().messages().list(userId='me', q=query).execute()
    messages = results.get('messages', [])
    if not messages:
        return None
    # Take the most recent email
    email_id = messages[0]['id']
    message = service.users().messages().get(userId='me', id=email_id).execute()
    payload = message['payload']
    parts = payload.get('parts', [])
    if not parts:
        data = payload['body']['data']
        html = base64.urlsafe_b64decode(data).decode()
        return html
    for part in parts:
        if part['mimeType'] == 'text/html':
            data = part['body']['data']
            html = base64.urlsafe_b64decode(data).decode()
            return html
    return None

######################################################
#          PDF EXTRACTION & SUMMARIZATION            #
######################################################

def get_pdf_link(huggingface_url):
    """Attempt to find a PDF link on the HF page."""
    try:
        response = requests.get(huggingface_url)
        if response.status_code != 200:
            print(f"Failed to fetch {huggingface_url}. Status code: {response.status_code}")
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_link = None
        for link in soup.find_all('a', href=True):
            if "View PDF" in link.text or link.get('href', "").endswith('.pdf'):
                pdf_link = link['href']
                break
        if not pdf_link:
            return None
        if not pdf_link.startswith('http'):
            pdf_link = requests.compat.urljoin(huggingface_url, pdf_link)
        return pdf_link
    except Exception as e:
        print(f"Error while fetching the PDF link from {huggingface_url}: {e}")
        return None

def extract_text_from_pdf_in_memory(pdf_url):
    """Download PDF in-memory and extract text."""
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code != 200:
            print(f"Failed to fetch PDF. Status code: {response.status_code}")
            return None
        pdf_content = io.BytesIO(response.content)
        with pdfplumber.open(pdf_content) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error while extracting text from PDF: {e}")
        return None

def summarize_text(text):
    """Use Gemini-2.0-flash-exp to generate a concise summary."""

    current_dateTime = datetime.now().strftime("%Y %B %d")
    prompt = (
        f"Summarize the following paper for a Data Scientist to quickly "
        f"understand what it is about in a concise manner. Today's date is: {current_dateTime}.\n\n"
        f"Here is the Paper:\n{text}"
    )
    response = model.generate_content(contents=prompt)
    return response.text

######################################################
#                      MEMO CREATION                 #
######################################################

def generate_memo(paper_summaries):
    """
    Generate a one-page daily memo that includes clickable links in Markdown format.
    Expects a list of dicts like: [{'link': ..., 'summary': ...}, ...]
    """
    combined_text = ""
    for paper in paper_summaries:
        combined_text += (
            f"{paper['summary']}\nLink: {paper['link']}\n******\n"
        )

    prompt = (
        "Generate a one-page daily memo based on the following summaries of various AI/ML papers. "
        "Each summary is separated by \"******\". After each summary, include the link as a Markdown "
        "hyperlink `[Paper Link](<url>)`. Keep it concise.\n\n"
        f"{combined_text}"
    )
    response = model.generate_content(contents=prompt)
    return response.text

######################################################
#                  EMAIL SENDING                     #
######################################################

def send_email(recipient, subject, body):
    """
    Example function for sending your memo via email.
    Make sure to replace with your valid credentials.
    """
    try:
        sender_email = "your_email@gmail.com"
        app_password = "your_app_password"
        message = MIMEText(body, "plain")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = recipient

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient, message.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

######################################################
#                 PINECONE SETUP                     #
######################################################

pinecone = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "ai-daily-papers"

# Create the index if not exists (dimensions = 512 for jina-embeddings-v2-small-en)
index_name = "ai-daily-papers"
if index_name not in pinecone.list_indexes()[0]['name']:
    pinecone.create_index(name=index_name, dimension=512,spec=ServerlessSpec(
            cloud = "aws",
      region = "us-east-1",
   ) )
index = pinecone.Index(index_name)

embedding_model = pipeline(
    "feature-extraction",
    model="jinaai/jina-embeddings-v2-small-en",
    tokenizer="jinaai/jina-embeddings-v2-small-en"
)

def store_in_pinecone(text: str, metadata: Dict[str, Any]):
    """
    Store embeddings in Pinecone, ensuring the vector size matches the index dimension (512).
    """
    embeddings = embedding_model(text)
    if isinstance(embeddings, list):
        vector = embeddings[0]  # shape: [1, 512]
    else:
        raise ValueError("Embedding model did not return a list of vectors.")

    if len(vector[0]) != 512:
        raise ValueError(f"Vector size mismatch: expected 512, got {len(vector[0])}")

    vector_id = metadata["id"]
    upsert_response = index.upsert([(vector_id, vector[0], metadata)])
    print("Data stored in Pinecone:", upsert_response)

def load_all_items_from_pinecone() -> List[Dict[str, Any]]:
    """
    Load all items (memos, chat messages, paper summaries, etc.) from Pinecone.
    Returns a list of metadata dicts, sorted by timestamp ascending.
    """
    dummy_vector = [0.0] * 512
    # blank filter => get everything
    results = index.query(
        vector=dummy_vector,
        filter={},  
        top_k=1000,  
        include_metadata=True
    )
    items = []
    if "matches" in results:
        for match in results["matches"]:
            meta = match["metadata"]
            # skip if 'type' is missing entirely
            if "type" not in meta:
                continue
            items.append(meta)
    items.sort(key=lambda x: x.get("timestamp", ""))
    return items

######################################################
#             GROUPING DATA BY DATE                  #
######################################################

def group_data_by_date(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Group items by date. The returned structure is:
    {
       "2025-01-21": {
          "memo_content": "text of memo if any",
          "memo_item": <the entire item for the memo if any>,
          "chats": [ {question, answer, ...}, ... ],
       },
       "2025-01-22": {
          ...
       }
    }
    We'll store only one memo per date (assuming that's the normal usage).
    """
    data_by_date = {}

    for meta in items:
        date = meta.get("date", None)

        # Some items might not have a "date" (e.g. older data or partial store).
        # Skip them or default to "no-date"
        if not date:
            continue

        if date not in data_by_date:
            data_by_date[date] = {
                "memo_content": None,
                "memo_item": None,
                "chats": []
            }

        item_type = meta["type"]
        if item_type == "daily_memo":
            data_by_date[date]["memo_content"] = meta.get("content", "")
            data_by_date[date]["memo_item"] = meta
        elif item_type == "chat":
            data_by_date[date]["chats"].append(meta)
        elif item_type == "paper_summary":
            # We won't display these directly, but you could store them if you like
            pass
        else:
            pass

    # Sort each date's chat by timestamp
    for date_val in data_by_date:
        data_by_date[date_val]["chats"].sort(key=lambda x: x.get("timestamp", ""))

    return data_by_date

######################################################
#                 STREAMLIT FUNCTIONS                #
######################################################

def main():
    # 1. Possibly generate today's memo if we haven't yet.
    check_and_generate_memo_for_today()

    # 2. Load all items from Pinecone and group them by date
    all_items = load_all_items_from_pinecone()
    data_by_date = group_data_by_date(all_items)

    # 3. Store in session_state
    st.session_state["data_by_date"] = data_by_date

    # 4. Build the sidebar to select which date's conversation we want
    sidebar_ui()

    # 5. Show the main chat interface for the selected date
    main_chat_ui()


def check_and_generate_memo_for_today():
    """
    Fetch daily papers from email, generate memo if not existing for today.
    If we already have today's memo in Pinecone, do nothing.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Check if today's memo already exists in Pinecone
    dummy_vector = [0.0] * 512
    filter_query = {"type": "daily_memo", "date": today_str}
    results = index.query(
        vector=dummy_vector,
        filter=filter_query,
        top_k=1,
        include_metadata=True
    )

    # If we find a daily_memo for today, do nothing
    if results and "matches" in results and len(results["matches"]) > 0:
        return  # Already have today's memo

    # If no daily memo for today, try to fetch from Gmail
    creds = authenticate_gmail()
    email_content = fetch_daily_papers_email(creds)
    if not email_content:
        return

    links = extract_links_from_email(email_content)
    if not links:
        return

    paper_summaries = []
    for link in links:
        pdf_url = get_pdf_link(link)
        if not pdf_url:
            continue
        text = extract_text_from_pdf_in_memory(pdf_url)
        if not text:
            continue
        summary = summarize_text(text)
        paper_summaries.append({"link": link, "summary": summary})

    if not paper_summaries:
        return

    # Generate daily memo
    memo_text = generate_memo(paper_summaries)

    # Store the daily memo in Pinecone
    memo_id = f"memo-{today_str}"
    memo_metadata = {
        "id": memo_id,
        "type": "daily_memo",
        "date": today_str,
        "timestamp": datetime.now().isoformat(),
        "content": memo_text
    }
    store_in_pinecone(text=memo_text, metadata=memo_metadata)

    # Optional: store the paper summaries as well
    for paper in paper_summaries:
        paper_metadata = {
            "id": paper["link"],
            "type": "paper_summary",
            "date": today_str,
            "timestamp": datetime.now().isoformat(),
            "summary": paper["summary"],
            "link": paper["link"]
        }
        store_in_pinecone(text=paper["summary"], metadata=paper_metadata)


def sidebar_ui():
    """
    Renders the sidebar. We show the available dates for which we have data (memo/chats).
    The user picks a date to see/resume that day's conversation in the main area.
    """
    st.sidebar.title("Conversation Dates")

    data_by_date = st.session_state.get("data_by_date", {})
    if not data_by_date:
        st.sidebar.write("No conversation data found.")
        return

    # Sort the dates descending or ascending as you prefer
    all_dates = sorted(data_by_date.keys(), reverse=True)

    # If user hasn't selected a date yet, pick the first from the list
    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = all_dates[0]  # default to the newest

    chosen_date = st.sidebar.selectbox("Select a date:", all_dates, 
                                       index=all_dates.index(st.session_state["selected_date"]))

    # Update the session state
    st.session_state["selected_date"] = chosen_date


def main_chat_ui():
    """
    Displays the selected date's memo and chat history using the new Streamlit chat interface.
    The user can ask a new question at the bottom (via st.chat_input).
    """
    data_by_date = st.session_state.get("data_by_date", {})
    selected_date = st.session_state.get("selected_date", None)

    if not selected_date or selected_date not in data_by_date:
        st.title("No Date Selected")
        return

    # Retrieve the memo and chat history for that date
    date_data = data_by_date[selected_date]
    memo_text = date_data.get("memo_content", None)
    chats = date_data.get("chats", [])

    # Optional page title or heading
    st.title(f"Conversation for {selected_date}")

    # 1) If there's a memo, display it at the top in an "assistant" message.
    if memo_text:
        with st.chat_message("assistant"):
            st.markdown(f"**Daily Memo for {selected_date}**")
            st.markdown(memo_text)
    else:
        with st.chat_message("assistant"):
            st.markdown(f"*(No memo found for {selected_date}.)*")

    # 2) Display each prior chat in chronological order
    for chat_entry in chats:
        user_q = chat_entry.get("question", "")
        llm_a = chat_entry.get("answer", "")

        if user_q:
            with st.chat_message("user"):
                st.write(user_q)
        if llm_a:
            with st.chat_message("assistant"):
                st.write(llm_a)

    # 3) st.chat_input is pinned at the bottom automatically.
    new_user_input = st.chat_input("Ask a new question about today's memo...")

    if new_user_input:
        # Build the full context for the LLM: memo + prior Q&As
        conversation_prompt = build_conversation_prompt(memo_text, chats)

        # Add the new user question at the end:
        conversation_prompt += f"\nUser: {new_user_input}\nAssistant:"

        # Get the LLM's response
        response = model.generate_content(contents=conversation_prompt)
        llm_answer = response.text

        # Immediately display these new messages in the UI
        with st.chat_message("user"):
            st.write(new_user_input)
        with st.chat_message("assistant"):
            st.write(llm_answer)

        # Store in Pinecone so it appears on future reload
        chat_id = f"chat-{selected_date}-{int(time.time())}"
        chat_metadata = {
            "id": chat_id,
            "type": "chat",
            "date": selected_date,
            "timestamp": datetime.now().isoformat(),
            "question": new_user_input,
            "answer": llm_answer
        }
        store_in_pinecone(text=new_user_input + " " + llm_answer, metadata=chat_metadata)

        # Also update local session so we don't lose it if user re-runs
        st.session_state["data_by_date"][selected_date]["chats"].append(chat_metadata)

        # Optionally force a rerun so the user doesn't see the prompt repeated in the input
        st.experimental_rerun()



def build_conversation_prompt(memo_text: str, chats: List[Dict[str, Any]]) -> str:
    """
    Build a single prompt for the LLM from the day's memo + prior Q&A.
    You can adjust formatting or how you incorporate them.
    """
    prompt = ""
    if memo_text:
        prompt += f"System: This is the daily memo:\n{memo_text}\n"

    for chat_entry in chats:
        question = chat_entry.get("question", "")
        answer = chat_entry.get("answer", "")
        prompt += f"User: {question}\nAssistant: {answer}\n"

    return prompt


######################################################
#                     RUN APP                        #
######################################################
if __name__ == "__main__":
    main()
