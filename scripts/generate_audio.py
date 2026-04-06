#!/usr/bin/env -S uv run python

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "audiocraft @ git+https://github.com/facebookresearch/audiocraft.git",
#     "soundfile",
#     "librosa",
#     "torchaudio",
#     "yt-dlp",
#     "pandas"
# ]
# ///

import os
os.environ["HF_HOME"] = "/workspace/UIQ/hf_cache"

import json
import random
import torchaudio
import pandas as pd
import yt_dlp
import soundfile as sf
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import warnings
warnings.filterwarnings("ignore")

def download_audio(youtube_id, start_time, out_path):
    end_time = start_time + 10
    out_base = out_path.replace('.wav', '')
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'outtmpl': out_base + '.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            error_code = ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
            return error_code == 0
        except Exception as e:
            return False

def main():
    out_dir = "/workspace/UIQ/audio_generate_sampling"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading AudioCaps CSV from cdjkim...")
    csv_url = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv"
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        print("Failed to download CSV:", e)
        return
    
    # Shuffle dataset to get random samples
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    samples = []
    print("Downloading original audios from YouTube...")
    for idx, row in df.iterrows():
        if len(samples) >= 5:
            break
            
        youtube_id = row['youtube_id']
        start_time = row['start_time']
        # The test.csv has 5 captions usually. We just take the first caption column.
        caption_cols = [c for c in df.columns if 'caption' in c]
        caption = row[caption_cols[0]]
        
        orig_filename = f"audiogen_{len(samples)}_{youtube_id}_original.wav"
        orig_path = os.path.join(out_dir, orig_filename)
        
        print(f"[{len(samples)+1}/5] Fetching {youtube_id} ({caption})...")
        success = download_audio(youtube_id, start_time, orig_path)
        
        if success and os.path.exists(orig_path):
            samples.append({
                'id': len(samples),
                'youtube_id': youtube_id,
                'caption': caption,
                'orig_path': orig_path,
                'orig_filename': orig_filename
            })
        else:
            print(f"  -> Failed to fetch {youtube_id}. Skipping.")
    
    if len(samples) == 0:
        print("Could not download any audio.")
        return

    print("\nLoading AudioGen model...")
    model = AudioGen.get_pretrained('facebook/audiogen-medium')
    model.set_generation_params(duration=10)
    
    descriptions = [s['caption'] for s in samples]
    print(f"Generating synthetic audio...")
    wavs = model.generate(descriptions)
    
    metadata = []
    
    for idx, (one_wav, sample) in enumerate(zip(wavs, samples)):
        clean_desc = "".join(c if c.isalnum() else "_" for c in sample['caption'])[:30]
        gen_filename = f"audiogen_{idx}_{clean_desc}.wav"
        gen_path = os.path.join(out_dir, f"audiogen_{idx}_{clean_desc}")
        
        # AudioGen write adds .wav automatically
        audio_write(gen_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        
        metadata.append({
            "id": idx,
            "youtube_id": sample['youtube_id'],
            "caption": sample['caption'],
            "generated_audio": gen_filename,
            "original_audio": sample['orig_filename']
        })
            
        print(f"Saved pair [{idx+1}/5]:")
        print(f"  Original: {sample['orig_filename']}")
        print(f"  Generated: {gen_filename}")

    json_path = os.path.join(out_dir, "metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"\nSaved metadata to {json_path}")

if __name__ == "__main__":
    main()
