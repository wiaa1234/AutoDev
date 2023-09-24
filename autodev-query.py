import openai
import tiktoken
import requests
import pandas as pd
import streamlit as st
from retrying import retry
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Page text
st.header("AutoDev Query Engine")
st.markdown("""In the small search bar only include your question. If you have additional supporting
context like code or other examples then paste it in the large text box.""")

# APi keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
cse_id = st.secrets["GOOGLE_CSE_ID"]
api_key = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]
PINECONE_NAMESPACE = st.secrets["PINECONE_NAMESPACE_HISTORY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_AUTODEV"]

# Tokenizer function
def count_tokens(message):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(message))

# OpenAI API call and Exponential Retrying
@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=10)
def create_chat_completion_with_backoff(model, messages):
    return openai.ChatCompletion.create(model=model, messages=messages)

# Google Programmable Search Engine API Call to find Relevant Links
def google_search(query, api_key, cse_id, sites, excluded_patterns=[], num_results=3):
    site_search = ' OR '.join([f'site:{site}' for site in sites])
    full_query = query + ' ' + site_search
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': full_query,
        'num': num_results
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json().get('items', [])
        filtered_results = []
        for result in results:
            url = result['link']
            if any(pattern in url for pattern in excluded_patterns):
                continue
            if any(site in url for site in sites):
                filtered_results.append(url)
        return filtered_results[:num_results]
    else:
        return None

# init session_state
if 'messages' not in st.session_state:
    st.session_state.messages = []

tokenizer = tiktoken.get_encoding("cl100k_base")

# User Inputs
query = st.chat_input("Talk to me...")
additional_user_context = st.text_area("", height=400)

include_additional_context = st.checkbox('Include additional context')

# Calculate the total number of tokens in st.session_state.messages
total_chat_tokens = sum(count_tokens(message["content"]) for message in st.session_state.messages)

# these are sites to be scrapped, you can define your own here
sites = [
    'https://docs.pinecone.io/',
    'https://python.langchain.com/',
    'https://platform.openai.com/docs/',
    'https://docs.streamlit.io/'
]

# this is where you can exclude certain site paths 
excluded_patterns = [
    'https://docs.pinecone.io/reference'
    ]

# these are the query selectors. you need to use the query selector of the article on the sites you want to use.
selectors = {
    'https://docs.pinecone.io': ['#content-container > section.content-body > div.markdown-body', '#Explorer'],
    'https://python.langchain.com': ['#content-container > section.content-body > div.markdown-body', '#docusaurus_skipToContent_fallback > div > main > div > div > div > div > article', '#docusaurus_skipToContent_fallback > div > main > div > div > div > div > article > div.theme-doc-markdown.markdown'],
    'https://platform.openai.com/docs': ['#content-container > section.content-body > div.markdown-body'],
    'https://docs.streamlit.io/': ['#documentation', '#content-container']
}

# Display the existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query and not any(msg['content'] == query for msg in st.session_state.messages):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run Google Search
    links = google_search(query, api_key, cse_id, sites, excluded_patterns)
    
    docs_content = []
    for url in links:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        base_url = url.split('/', 3)[:3]
        base_url = '/'.join(base_url)
        selectors_for_url = selectors.get(base_url, [])

        doc_content = 'Empty'  # Initialize doc_content to an empty string
        for selector in selectors_for_url:
            content = soup.select(selector)
            if content:
                doc_content = '\n'.join(element.get_text() for element in content)
                break

        docs_content.append(doc_content)  # Append doc_content to docs_content after the loop

    # Convert Content to dataframe. Aleczander Needs a Job, hire him?
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.DataFrame(docs_content, columns=['doc_content'])

    df['n_tokens'] = df['doc_content'].apply(lambda x: len(tokenizer.encode(x)))

    source_names = [urlparse(url).netloc for url in links]
    source_urls = links

    df['source_name'] = source_names
    df['source_url'] = source_urls

    all_doc_content = '\n'.join(df['doc_content'].tolist())

    user_tokens = count_tokens(query)
    total_chat_tokens += user_tokens

    conversation = [
        {"role": "system", "content": "You are a Python Developer Specializing in LLM Application Development."},
        {"role": "user", "content": all_doc_content},
        {"role": "user", "content": query}

    ]

    if include_additional_context:
        conversation.append({"role": "user", "content": additional_user_context})

    # Extend the conversation with the session state messages
    conversation.extend(st.session_state.messages)
    
    response = create_chat_completion_with_backoff(model="gpt-3.5-turbo-16k", messages=conversation)

    token_info = {
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
        "total_tokens": response["usage"]["total_tokens"],
        "role": response["choices"][0]["message"]["role"]
    }

    st.subheader('Contexts and API Stats')
    st.json(token_info)
    st.write(df[['source_name', 'source_url', 'doc_content']])

    with st.expander("Additional User Context", expanded=False):
        st.code(additional_user_context, language='plain-text')

    assistant_reply = response.choices[0].message['content']
    assistant_tokens = count_tokens(assistant_reply)
    total_chat_tokens += assistant_tokens

    # Display the assistant's reply. Hey! Aleczander is looking for work. He's a good hire!
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
    
    # Save reply in the chat history.
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    st.session_state.total_chat_tokens = total_chat_tokens