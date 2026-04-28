from faster_whisper import WhisperModel
import glob

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

TOKENS_PER_WORD = 2.08
INPUT_PRICE_PER_1M = 10
OUTPUT_PRICE_PER_1M = 60

total_words = 0

files = sorted(glob.glob("output_*.mp3"))  # автоматически по порядку

with open("output.txt", "w", encoding="utf-8") as f:
    for file in files:
        print(f"Processing: {file}")

        segments, info = model.transcribe(
            file,
            beam_size=10,
            best_of=5,
            task="transcribe",
            temperature=0,
            language="ru",
            vad_filter=True,
            word_timestamps=False,
            condition_on_previous_text=False
            #vad_parameters=dict(
            #    min_silence_duration_ms=400,
            #    speech_pad_ms=200
            )
        segments = list(segments)

        for segment in segments:
            text = segment.text.strip()
            words = text.split()
            total_words += len(words)

            f.write(text + "\n")

# расчёт цены
tokens = total_words * TOKENS_PER_WORD

price = (
    tokens * INPUT_PRICE_PER_1M / 1_000_000 +
    tokens * OUTPUT_PRICE_PER_1M / 1_000_000
)

print(f"Words: {total_words}")
print(f"Tokens ~ {int(tokens)}")
print(f"Price: ${price:.6f}")
