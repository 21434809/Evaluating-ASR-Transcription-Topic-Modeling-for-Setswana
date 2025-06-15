import os

os.environ["HF_HOME"] = "/ext_data/zion"

import pandas as pd
import torch
import librosa
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Set memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_audio_file(path, target_sr=16000):
    """Load audio file using librosa and resample if needed"""
    try:
        audio_array, sampling_rate = librosa.load(path, sr=target_sr)
        return {"array": audio_array, "sampling_rate": sampling_rate}
    except Exception as e:
        print(f"Error loading audio file {path}: {e}")
        return None

def process_audio_chunks(audio_data, processor, model, device, chunk_size=20):
    """Process audio in chunks to avoid memory issues"""
    sr = audio_data["sampling_rate"]
    array = audio_data["array"]
    chunk_samples = chunk_size * sr
    full_transcription = ""
    
    for i in range(0, len(array), chunk_samples):
        chunk = array[i:i+chunk_samples]
        input_dict = processor(
            chunk, 
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            try:
                logits = model(input_dict.input_values).logits
                pred_ids = torch.argmax(logits, dim=-1)[0]
                full_transcription += processor.decode(pred_ids) + " "
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"GPU memory error, falling back to CPU for chunk {i//chunk_samples}")
                    model = model.to("cpu")
                    input_dict = input_dict.to("cpu")
                    logits = model(input_dict.input_values).logits
                    pred_ids = torch.argmax(logits, dim=-1)[0]
                    full_transcription += processor.decode(pred_ids) + " "
                    model = model.to(device)  # Move back to GPU if possible
                else:
                    raise e
                    
    return full_transcription.strip()

def process_chunked_audio(root_dir, output_csv, stats_csv):
    """Process all audio chunks in the directory structure"""
    # Try GPU first, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load model and processor with Setswana adapter
        model_name = "guymandude/MMS-ASR-ZA-11"
        print("Loading model and setting up Setswana adapter...")
        
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True
        ).to(device)
        
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model.load_adapter("tsn")
        processor.tokenizer.set_target_lang("tsn")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    csv_data = []
    stats_data = []
    supported_formats = ('.mp3', '.wav', '.flac', '.ogg', '.aac')
    
    # Get all numbered folders (folder_1, folder_2, etc.)
    chunk_folders = [d for d in os.listdir(root_dir) 
                    if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('folder_')]
    
    print(f"Found {len(chunk_folders)} chunk folders to process")
    
    for folder in tqdm(sorted(chunk_folders), desc="Processing folders"):
        folder_path = os.path.join(root_dir, folder)
        episode_number = int(folder.split('_')[1])
        
        # Process each audio file in the folder
        for audio_file in os.listdir(folder_path):
            if audio_file.lower().endswith(supported_formats):
                audio_path = os.path.join(folder_path, audio_file)
                chunk_number = int(os.path.splitext(audio_file)[0].split('_')[-1])
                
                try:
                    print(f"\nProcessing: {folder}/{audio_file}")
                    audio = load_audio_file(audio_path)
                    if audio is None:
                        continue
                        
                    duration = len(audio["array"]) / audio["sampling_rate"]
                    
                    # Process in chunks with GPU fallback
                    transcription = process_audio_chunks(audio, processor, model, device)
                    
                    # Save outputs
                    csv_data.append({
                        "Episode Number": episode_number,
                        "Chunk Number": chunk_number,
                        "Duration (seconds)": round(duration, 2),
                        "Transcription": transcription
                    })
                    
                    stats_data.append({
                        "Episode Number": episode_number,
                        "Chunk Number": chunk_number,
                        "Duration (seconds)": round(duration, 2),
                        "Sampling Rate": audio["sampling_rate"],
                        "Audio Length": len(audio["array"]),
                        "File Path": audio_path
                    })
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
    
    # Save transcription data
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
        print(f"\nSuccessfully processed {len(csv_data)} audio chunks")
        
        # Save statistics data
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(stats_csv, index=False, encoding="utf-8")
        print(f"Saved statistics to {stats_csv}")
        
        # Generate visualizations
        generate_visualizations(stats_df)
    else:
        print("\nNo files were processed successfully")

def generate_visualizations(stats_df):
    """Generate visualizations from the statistics data"""
    print("\nGenerating visualizations...")
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. Duration distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=stats_df, x="Duration (seconds)", bins=30)
    plt.title("Distribution of Audio Chunk Durations")
    plt.savefig("visualizations/duration_distribution.png")
    plt.close()
    
    # 2. Duration by episode
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=stats_df, x="Episode Number", y="Duration (seconds)")
    plt.title("Duration Distribution by Episode Number")
    plt.tight_layout()
    plt.savefig("visualizations/duration_by_episode.png")
    plt.close()
    
    # 3. Total duration per episode
    episode_duration = stats_df.groupby("Episode Number")["Duration (seconds)"].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=episode_duration, x="Episode Number", y="Duration (seconds)")
    plt.title("Total Duration per Episode Number")
    plt.tight_layout()
    plt.savefig("visualizations/total_duration_per_episode.png")
    plt.close()
    
    # 4. Number of chunks per episode
    chunk_count = stats_df.groupby("Episode Number").size().reset_index(name="Chunk Count")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=chunk_count, x="Episode Number", y="Chunk Count")
    plt.title("Number of Chunks per Episode Number")
    plt.tight_layout()
    plt.savefig("visualizations/chunks_per_episode.png")
    plt.close()
    
    print("Visualizations saved to 'visualizations' directory")

if __name__ == "__main__":
    root_directory = "chunked_audio"
    output_csv_path = "podcast_transcriptions_chunked.csv"
    stats_csv_path = "podcast_statistics.csv"
    
    if not os.getenv('HF_TOKEN'):
        print("Warning: HF_TOKEN not set. You may need to run: huggingface-cli login")
    
    process_chunked_audio(root_directory, output_csv_path, stats_csv_path)
