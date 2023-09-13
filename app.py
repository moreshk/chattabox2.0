import argparse
import io
import os
import speech_recognition as sr
import openai
from dotenv import load_dotenv  # Import the load_dotenv function

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

# Load environment variables from .env file
load_dotenv()

def limit_conversation_history(conversation: list, limit: int = 5) -> list:
    """Limit the size of conversation history.

    :param conversation: A list of previous user and assistant messages.
    :param limit: Number of latest messages to retain. Default is 3.
    :returns: The limited conversation history.
    :rtype: list
    """
    return conversation[-limit:]

def generate_reply(conversation: list) -> str:
    """Generate a ChatGPT response.
    :param conversation: A list of previous user and assistant messages.
    :returns: The ChatGPT response.
    :rtype: str
    """
    print("Original conversation length:", len(conversation))
    print("Original Conversation", conversation)
    # Limit conversation history
    conversation = limit_conversation_history(conversation)
    
    print("Limited conversation length:", len(conversation))
    print("New Conversation", conversation)
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
                {
                    "role": "system", 
                    "content": (
                    "Your role is a conversational compaion. Keep your responses short and friendly.")
                }

        ] + conversation,
        temperature=1
    )
    return response["choices"][0]["message"]["content"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy_threshold", default=os.environ.get('ENERGY_THRESHOLD', 1000),
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=os.environ.get('RECORD_TIMEOUT', 2),
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=os.environ.get('PHRASE_TIMEOUT', 3),
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default=os.environ.get('DEFAULT_MICROPHONE', 'pulse'),
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # Setup OpenAI API
    openai.api_key = os.environ.get('OPENAI_API_KEY')  # Now it will read from the .env file

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # Initialize SpeechRecognizer
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Microphone setup
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    print("Initializing microphone. Please wait...")
    sleep(3)  # Wait for 3 seconds
    print("Ready to transcribe.\n")

    # Initialize conversation list
    conversation = []

    # Initialize conversation list and first_message_received flag
    conversation = []
    first_message_received = False

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                elif not first_message_received:
                    phrase_complete = True
                    first_message_received = True

                phrase_time = now

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Save the WAV data to a temporary file
                with NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
                    temp_wav_file.write(wav_data.getvalue())
                    temp_wav_file.flush()

                    # Transcribe using OpenAI Whisper API
                    with open(temp_wav_file.name, "rb") as audio_file:
                        transcript = openai.Audio.transcribe("whisper-1", audio_file)
                        text = transcript['text'].strip()

                print(f"Debug: phrase_complete={phrase_complete}, text={text}")  # Debug statement

                if phrase_complete:
                    # Append user's message to conversation and transcription
                    user_message = text
                    conversation.append({"role": "user", "content": user_message})
                    transcription.append(f"User: {user_message}")

                    # Generate a reply based on the conversation
                    try:
                        reply = generate_reply(conversation)
                    except Exception as e:
                        print(f"Error generating reply: {e}")

                    # Append assistant's reply to conversation and transcription
                    conversation.append({"role": "assistant", "content": reply})
                    transcription.append(f"Assistant: {reply}")

                    # Clear the screen
                    # sleep(2)
                    os.system('cls' if os.name == 'nt' else 'clear')

                    # Display the transcribed text and the assistant's reply
                    for line in transcription:
                        print(line)

                else:
                    transcription[-1] = f"User: {text}"

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)

                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
