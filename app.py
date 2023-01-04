import gradio as gr
import whisper
import cohere
from deep_translator import GoogleTranslator
from gtts import gTTS
import gtts.langs
#from dotenv import load_dotenv

#load_dotenv()

model = whisper.load_model("base")

LANGUAGES = list(gtts.lang.tts_langs())

def transcribe(api,audio,language):
    co = cohere.Client(api) 

    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)

    #cohere
    response = co.generate(
    model='xlarge',
    prompt=f'This program will generate an introductory paragraph to a blog post given a blog title, audience, and tone of voice.\n--\nBlog Title: Best Activities in Toronto\nAudience: Millennials\nTone of Voice: Lighthearted\nFirst Paragraph: Looking for fun things to do in Toronto? When it comes to exploring Canada\'s largest city, there\'s an ever-evolving set of activities to choose from. Whether you\'re looking to visit a local museum or sample the city\'s varied cuisine, there is plenty to fill any itinerary. In this blog post, I\'ll share some of my favorite recommendations\n--\nBlog Title: Mastering Dynamic Programming\nAudience: Developers\nTone: Informative\nFirst Paragraph: In this piece, we\'ll help you understand the fundamentals of dynamic programming, and when to apply this optimization technique. We\'ll break down bottom-up and top-down approaches to solve dynamic programming problems.\n--\nBlog Title: How to Get Started with Rock Climbing\nAudience: Athletes\nTone: Enthusiastic\nFirst Paragraph:If you\'re an athlete who\'s looking to learn how to rock climb, then you\'ve come to the right place! This blog post will give you all the information you need to know about how to get started in the sport. Rock climbing is a great way to stay active and challenge yourself in a new way. It\'s also a great way to make new friends and explore new places. So, what are you waiting for? Get out there and start climbing!\n--\nBlog Title: {result.text}\nAudience: Engineers\nTone: Enthusiastic\nFirst Paragraph:',
    max_tokens=200,
    temperature=0.8,
    k=0,
    p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=["--"],
    return_likelihoods='NONE')
    #result.text
    reptxt = response.generations[0].text.strip("--")

    #Google models
    translated = GoogleTranslator(source='auto', target=language).translate(reptxt)
    filename = 'result.mp3'
    tts = gTTS(text=translated, lang=language)
    tts.save(filename)
    return filename, translated



gr.Interface(
    title = 'Coherent Speech', 
    fn=transcribe, 
    inputs=[
        gr.inputs.Textbox(lines=1, label="Enter your Cohere API Key"),
        gr.inputs.Audio(source="microphone", type="filepath"), 
            gr.Radio(label="Language", choices=LANGUAGES, value="en")
    ],
    outputs=[gr.Audio(label="Output",type="filepath"),gr.outputs.Textbox(label="Generated Text")],
    live=True).launch()
