import os
import pandas as pd
import torch
import librosa
import csv
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Set memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_audio_file(path, target_sr=16000):
    """Load audio file using librosa and resample if needed"""
    audio_array, sampling_rate = librosa.load(path, sr=target_sr)
    return {"array": audio_array, "sampling_rate": sampling_rate}

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

def process_podcasts(root_dir, output_csv):
    """Process all podcast episodes in the directory structure"""
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
    podcast_folders = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"Found {len(podcast_folders)} podcast folders to process")
    
    for folder in tqdm(podcast_folders, desc="Processing podcasts"):
        folder_path = os.path.join(root_dir, folder)
        mp3_filename = folder.replace(" ", "_") + ".mp3"
        mp3_file = os.path.join(folder_path, mp3_filename)
        
        if not os.path.exists(mp3_file):
            print(f"\nNo MP3 file found for folder {folder}")
            continue
        
        try:
            print(f"\nProcessing: {folder}")
            audio = load_audio_file(mp3_file)
            duration = len(audio["array"]) / audio["sampling_rate"]
            
            # Process in chunks with GPU fallback
            transcription = process_audio_chunks(audio, processor, model, device)
            
            # Save outputs
            txt_file = os.path.join(folder_path, f"{folder}.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            csv_data.append({
                "episode_name": folder,
                "duration_seconds": round(duration, 2),
                "transcription": transcription
            })
            
        except Exception as e:
            print(f"Error processing {folder}: {e}")
            continue
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
        print(f"\nSuccessfully processed {len(csv_data)}/{len(podcast_folders)} folders")
    else:
        print("\nNo files were processed successfully")

if __name__ == "__main__":
    root_directory = os.path.expanduser("/ext_data/zion/pontsho-pilane-motsweding-covid19")
    output_csv_path = os.path.expanduser("~/ext_data/zion/podcast_transcriptions.csv")
    
    # Check for Hugging Face token
    if not os.getenv('HF_TOKEN'):
        print("Warning: HF_TOKEN not set. You may need to run: huggingface-cli login")
    
    process_podcasts(root_directory, output_csv_path)
