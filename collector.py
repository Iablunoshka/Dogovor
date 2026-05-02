from faster_whisper import WhisperModel
import argparse
import glob
import os
import shutil
import subprocess
import math
from pathlib import Path


model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

TOKENS_PER_WORD = 2.08
INPUT_PRICE_PER_1M = 10
OUTPUT_PRICE_PER_1M = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split audio with ffmpeg if needed and transcribe it."
    )

    # СНАЧАЛА позиционный аргумент
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=None,
        help="Путь к аудио"
    )

    # ПОТОМ именованный
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=0,
        help="Длина сегмента в минутах"
    )

    return parser.parse_args()


def split_audio_by_minutes(audio_path, chunk_minutes, output_dir):
    if chunk_minutes <= 0:
        return [audio_path]

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Установите ffmpeg или запустите без --chunk-minutes."
        )

    segment_seconds = int(chunk_minutes * 60)

    if segment_seconds <= 0:
        raise ValueError("--chunk-minutes должен быть больше 0.")

    output_pattern = os.path.join(output_dir, "output_%03d.mp3")

    command = [
        "ffmpeg",
        "-y",
        "-i", audio_path,

        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",

        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",

        output_pattern,
    ]

    subprocess.run(command, check=True)

    return sorted(glob.glob(os.path.join(output_dir, "output_*.mp3")))


def get_files_for_transcription(audio_path, chunk_minutes, temp_dir):
    if audio_path:
        audio_path = str(Path(audio_path).expanduser())

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Аудио-файл не найден: {audio_path}")

        return split_audio_by_minutes(audio_path, chunk_minutes, temp_dir)

    return sorted(glob.glob("output_*.mp3"))


def transcribe_files(files,file_name):
    total_words = 0

    with open(f"{file_name}.txt", "w", encoding="utf-8") as f:
        for file in files:
            print(f"Processing: {file}")

            segments, info = model.transcribe(
                file,
                beam_size=10,
                best_of=5,
                task="transcribe",
                temperature=0,
                language=None,
                vad_filter=True,
                word_timestamps=False,
                condition_on_previous_text=False,
                initial_prompt = ("Speech may be mixed: Russian, Ukrainian. Do not translate. Preserve the language of the spoken words.")
                #vad_parameters=dict(
                #    min_silence_duration_ms=400,
                #
            )

            segments = list(segments)

            for segment in segments:
                text = segment.text.strip()
                words = text.split()
                total_words += len(words)

                f.write(text + "\n")

    return total_words


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    # папка для временных файлов
    temp_dir = script_dir / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)

    try:
        files = get_files_for_transcription(
            args.audio_path,
            args.chunk_minutes,
            str(temp_dir)
        )

        if not files:
            raise FileNotFoundError("Файлы для транскрипции не найдены.")

        file_name = Path(args.audio_path).stem
        total_words = transcribe_files(files,file_name)

    finally:
        pass
        # очищаем временные файлы после работы
        for f in temp_dir.glob("output_*.mp3"):
            try:
                f.unlink()
            except:
                pass

    tokens = total_words * TOKENS_PER_WORD

    price = round(tokens * INPUT_PRICE_PER_1M / 1_000_000 + tokens * OUTPUT_PRICE_PER_1M / 1_000_000, 2) if tokens <= 1428 else math.floor((tokens * INPUT_PRICE_PER_1M / 1_000_000 + tokens * OUTPUT_PRICE_PER_1M / 1_000_000) * 10) / 10

    print(f"Words: {total_words}")
    print(f"Tokens ~ {int(tokens)}")
    print(f"Price: €{price}")


if __name__ == "__main__":
    main()
