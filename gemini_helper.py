from IPython.display import Markdown
from PIL import Image
import textwrap
import google.generativeai as GenAI
from transformers import pipeline
from youtube_dl import YoutubeDL

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

# Function to convert text to Markdown format
def to_markdown(text):
    """
    Convert plain text to Markdown format.

    Args:
        text (str): The plain text.

    Returns:
        Markdown: The formatted text in Markdown.
    """
    text = text.replace(".", " *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


# Function to get responses for questions
def get_gemini_response_question(question):
    """
    Get Gemini response for a given question.
    Args:
        question (str): The input question.
    Returns:
        str: The Gemini response.
    """
    model = GenAI.GenerativeModel(model_name="gemini-pro",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    response = model.generate_content(question)
    return response.text


# Function to get responses for images
def get_gemini_response_image(input_text, image):
    """
    Get Gemini response for a given image.
    Args:
        input_text (str): The input prompt text.
        image (Image): The uploaded image.
    Returns:
        str: The Gemini response.
    """
    model = GenAI.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                  safety_settings=safety_settings,
                                  generation_config=generation_config)
    if input_text != "":
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text

def summarize_video(video_url):
    """
    Summarizes a video using the youtube-dl and transformers libraries.

    Args:
        video_url (str): The URL of the YouTube video.

    Returns:
        str: A summary of the video.
    """

    # Download the video using youtube-dl
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        audio_file = info_dict['formats'][0]['url']

    # Summarize the audio using transformers
    summarizer = pipeline("summarization")
    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()
    summary = summarizer(audio_bytes)[0]['summary_text']

    return summary


def chat_with_video(video_url, user_input):
    """
    Chats with a video using the Gemini model.

    Args:
        video_url (str): The URL of the YouTube video.
        user_input (str): The user's input.

    Returns:
        str: The Gemini response.
    """

    # Get the video summary
    video_summary = summarize_video(video_url)

    # Combine the video summary and user input as the prompt
    prompt = f"{video_summary} {user_input}"

    # Generate the response using Gemini
    model = GenAI.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                  safety_settings=safety_settings,
                                  generation_config=generation_config)
    response = model.generate_content(prompt)

    return response