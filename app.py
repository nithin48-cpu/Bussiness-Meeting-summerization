# from transformers import pipeline
import streamlit as st
import os
from moviepy.editor import *
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import pipeline

uploaded_file = st.file_uploader("Choose a file", type=['mp4', 'mkv', 'OGG', 'MPEG', 'flac'])

# os.chdir('N:\Final_project\Project')
# curr_dir = r'N:\Final_project\Project'


# st.write(os.getcwd())
# print(os.getcwd())
@st.cache_data
def save_file(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())


@st.cache_resource
def load_model():
    model_id = "openai/whisper-small.en"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    # print(os.path.join(os.getcwd(), r'models\wishper'))
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(os.path.join(os.getcwd(), r'models\wishper'))
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


@st.cache_resource
def load_model2():
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    return summarizer


try:
    save_file(uploaded_file)
    st.write("saved sucessfully")

    video = VideoFileClip(uploaded_file.name)
    audio = video.audio
    audio_output_path = 'output.wav'
    audio.write_audiofile(audio_output_path)

    pipe = load_model()
    audio_data = librosa.load(audio_output_path)

    structure = {'array': audio_data[0], 'sampling_rate': 44100}
    results = pipe(structure)['text']
    print(results)
    st.write(results)

    summarizer1 = load_model2()
    result = summarizer1(str(results))[0]['summary_text']

    st.write(result)

except:
    st.write("Some error occured")
