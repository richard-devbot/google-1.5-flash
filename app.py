from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from PIL import Image
import os
from gemini_helper import to_markdown, get_gemini_response_question, get_gemini_response_image, chat_with_video, summarize_video
import google.generativeai as GenAI  # Add this import statement
import requests
from bs4 import BeautifulSoup


# Load environment variables from `.env`
load_dotenv()

# Configure Gemini API key
GenAI.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Set Streamlit page config
st.set_page_config(page_title="Ricky Richardson AI personal Assitant", page_icon=":robot")
st.image("Logo.png", width=150)
#st.title("Gunde Richardson's 💖   AI Companion: 💫Harnessing the Love-Powered Gemini & Google's LLM 🌟")
st.markdown("<h1 style='width: 110%; text-align: center;'>Gunde Richardson's 💖 AI Companion: 💫Harnessing the Love-Powered gemini-1.5-flash-latest by Google's LLM 🌟</h1>", unsafe_allow_html=True)
st.subheader("This Application can answer any of your questions and generate descriptions for images. Simply type your question or upload an image and click the corresponding button to get a response from the Gemini AI model.")
# Display option to choose between question/chat and image generation
option = st.selectbox(
    "Select question or image Generation?",
    ("🗣️ Question & Chat", "📸 Image Chat", "💻 Code Chat", "🎨 Image Generation from Prompt", "🎬 Video Summarization & Chat"),
    index=None,
    placeholder="Select a method...",
)

st.write('You selected:', option)

# Streamlit app for question and chat
if option == "🗣️ Question & Chat":
    st.header("Gemini Application - Question & Chat")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display chat history
    for i, chat in enumerate(st.session_state['chat_history']):
        if i % 2 == 0:
            message(chat, is_user=True, key=str(i), avatar_style=":human")
        else:
            message(chat, is_user=False, key=str(i), avatar_style=":bot")

    # Get user input
    user_input = st.text_input("You: ", key="input",)

    # Send message to Gemini and update chat history
    if user_input:
        response = get_gemini_response_question(user_input)
        st.session_state['chat_history'].append(user_input)
        st.session_state['chat_history'].append(response)

        # Display response
        message(response, is_user=False, key=str(len(st.session_state['chat_history'])-1), avatar_style="bot")


# Streamlit app for image generation
elif option == "📸 Image Chat":
    st.header("Gemini Application - Image Generation")
    col1, col2 = st.columns(2)
    input_text = col1.text_input("Input Prompt: ", key="input")
    uploaded_image = col2.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col2.image(image, caption="Uploaded Image.", use_column_width=True)

    submit_button = st.button("send")

    if submit_button:
        response = get_gemini_response_image(input_text, image)
        st.subheader("The Response is:")
        st.markdown(response)

elif option == "🎨 Image Generation from Prompt":
    st.header("Gemini Application - Image Generation from Prompt")

    # Get user input for prompt
    prompt = st.text_input("Enter a prompt for image generation:", key="prompt_input")

    # Generate image
    if prompt:
        model = GenAI.GenerativeModel(model_name="gemini-1.5-flash-latest")
        response = model.generate_content(prompt, response_mime_type="image/png")

        # Display the generated image
        st.image(response.image, caption="Generated Image.", use_column_width=True)

# Streamlit app for video summarization and chat
elif option == "🎬 Video Summarization & Chat":
    st.header("Gemini Application - Video Summarization & Chat")

    # Get user input for video URL
    video_url = st.text_input("Enter YouTube video URL:", key="video_url")

    # Get user input for chat
    user_input = st.text_input("You: ", key="chat_input")

    # Summarize the video and chat with it
    if video_url and user_input:
        summary = summarize_video(video_url)
        response = chat_with_video(video_url, user_input)

        # Display the summary and response
        st.subheader("Video Summary:")
        st.markdown(summary)
        st.subheader("Gemini Response:")
        st.markdown(response)

# Code Chat functionality
elif option == "💻 Code Chat":
    st.header("Gemini Application - Code Chat")

    # Get user input for GitHub URL
    github_url = st.text_input("Enter GitHub URL:", key="github_url")

    # Get user input for chat
    user_input = st.text_input("You: ", key="chat_input")

    # Process the GitHub URL and extract code
    if github_url:
        try:
            response = requests.get(github_url)
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all code elements in the repository
            code_elements = soup.find_all("pre", class_="CodeMirror-line")

            # Combine all code elements into a single string
            code = "\n".join([code_element.text.strip() for code_element in code_elements])

            # Combine code and user input as prompt
            prompt = f"{code}\n\n{user_input}"

            # Generate response using Gemini
            model = GenAI.GenerativeModel(model_name="gemini-1.5-flash-latest")
            response = model.generate_content(prompt)

            # Display the response
            st.subheader("Gemini Response:")
            st.markdown(response)

        except Exception as e:
            st.error(f"Error processing GitHub URL: {e}")