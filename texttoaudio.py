from transformers import AutoProcessor, MusicgenForConditionalGeneration # Import the correct class
from transformers.models.musicgen.modeling_musicgen import MusicgenForConditionalGeneration # Sometimes this explicit import is needed

model_name = "facebook/musicgen-small"

# Use the correct model class
try:
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    text_prompt = input("enter prompt : ")
    inputs = processor(text=text_prompt, sampling_rate=model.config.audio_encoder.sampling_rate, return_tensors="pt") # Adjust sampling_rate key
    audio_values = model.generate(**inputs, max_new_tokens=256) # Adjust max_new_tokens for duration
    sampling_rate = model.config.audio_encoder.sampling_rate # Adjust sampling_rate key

    # You'll likely want to save or play the audio_values here
    # Example: Save to a WAV file (requires the 'scipy' library)
    import scipy.io.wavfile as wav
    wav.write("generated_music.wav", sampling_rate, audio_values[0, 0].numpy()) # Adjust indexing based on output shape

    print("Music generation complete!")

except Exception as e:
    print(f"An error occurred: {e}")




