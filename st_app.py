import streamlit as st

import cohere
import numpy as np
import time
import re
import pandas as pd
from tqdm import tqdm
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from pytube import extract
from youtube_transcript_api import YouTubeTranscriptApi
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="YTBrief")


# Paste your API key here. Remember to not share publicly
api_key = '1hVmS6rOnPF00b1RiBwsTc2zAJzORDHY7bUov7np'

# Create and retrieve a Cohere API key from dashboard.cohere.ai/welcome/register
co = cohere.Client(api_key)


# Streamlit
st.title(':red[YT]Brief: _Get answers from YouTube Videos in seconds!_')
url = st.text_input('Input URL', placeholder ='YouTube Video URL')
query = st.text_input('Search', placeholder ='What do you want to search for in the video?')

if st.button('Search'):
    st.info('Processing the URL', icon="ℹ️")
    try:
        # Extract YouTube video Id from URL.
        id = extract.video_id(url)
    except:
        st.error('Please enter a valid YouTube video link.', icon="🚨")
    try:   
        # Get YouTube video transcript
        transcript = YouTubeTranscriptApi.get_transcript(id)
        transcript_df = pd.DataFrame(transcript)
    except:
        st.error('The YouTube video doesn"t have a transcript. Please try another one!', icon="🚨")

    st.info('Performing the Cohere AI magic! Please wait..', icon="ℹ️")
    # Get the embeddings
    embeds = co.embed(texts=list(transcript_df['text'][:300]), model='large', truncate='LEFT').embeddings

    # Creating the faiss index
    dimension = np.array(embeds).shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add embeds to Index
    index.add(np.float32(np.array(embeds)))

    # Get the query's embedding
    query_embed = co.embed(texts=[query], model="large", truncate="LEFT").embeddings

    # Retrieve the nearest neighbors
    k=10
    D, I = index.search(np.float32(np.array(query_embed)), k)
    # Format the results
    results = pd.DataFrame(data={'texts': transcript_df.iloc[I[0]]['text'], 'distance': D[0]})
    top_result, result_index = results.iloc[0,:], results.iloc[1,:].name
    timestamp = int(transcript_df.iloc[result_index,:]['start'].round())
    hhmmss_format = time.strftime('%H:%M:%S', time.gmtime(timestamp))
    result_url = f"https://www.youtube.com/watch?v={id}"

    st.success('Semantic search successfully performed!', icon="✅")

    # Displaying the result
    st.subheader('Results')
    st.write(f'Play the video to hear the answer for your search query.')
    st.video(result_url, start_time=timestamp)
    st.write(f'Alternatively, you can start reading the transcript of the video at :blue[{hhmmss_format}]')

else:
    st.write('Please fill the fields above, and click search.')
    
