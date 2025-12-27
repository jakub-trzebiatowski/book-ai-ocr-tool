import wave
from google import genai
from google.genai import types

PROJECT_ID = "book-digitizer-482317"

# 1. Initialize the client
client = genai.Client()

# A list of texts to be converted to speech
prompts = [
    "Say cheerfully: Welcome to the future of AI speech!",
    "This is the second audio file generated in a batch.",
    "And here is a third, just for good measure."
]

# 3. Loop through prompts, generate and save audio for each
for i, prompt in enumerate(prompts):
    # 2. Configure and request TTS content
    # The model must be a Gemini 2.0+ variant supporting native TTS
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"  # Choose from available voices like Kore, Fenrir, etc.
                    )
                )
            )
        )
    )

    # Extract and save the audio blob
    # The audio data is returned as raw PCM (L16) at 24kHz
    audio_part = response.candidates[0].content.parts[0]
    if audio_part.inline_data:
        audio_bytes = audio_part.inline_data.data
        output_filename = f"output_speech_{i}.wav"
        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)  # 24kHz sample rate
            wf.writeframes(audio_bytes)

        print(f"Audio saved to {output_filename}")
