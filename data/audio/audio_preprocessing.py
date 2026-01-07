import os
import csv

BASE_AUDIO_DIR = "data/audio/raw"
OUTPUT_CSV = "data/audio/audio_mapping.csv"

def rename_and_map(subfolder, prefix):
    folder_path = os.path.join(BASE_AUDIO_DIR, subfolder)
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".wav")])

    mappings = []

    for idx, filename in enumerate(files, start=1):
        new_name = f"{prefix}_{idx:02d}.wav"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

        mappings.append({
            "label": subfolder,
            "old_filename": filename,
            "new_filename": new_name,
            "path": f"data/audio/raw/{subfolder}/{new_name}"
        })

    return mappings

def main():
    all_mappings = []

    all_mappings.extend(rename_and_map("fake", "audio_fake"))
    all_mappings.extend(rename_and_map("real", "audio_real"))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "old_filename", "new_filename", "path"]
        )
        writer.writeheader()
        writer.writerows(all_mappings)

    print(f"Renommage terminé")
    print(f"Mapping CSV créé : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
