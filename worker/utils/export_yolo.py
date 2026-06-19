import os
import hashlib
import shutil
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def get_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    dest_dir = "/home/sagnik/Projects/docker-composes/manga-library/data/worker/huggingface/models"
    os.makedirs(dest_dir, exist_ok=True)
    
    print("Downloading best.pt from Hugging Face...")
    pt_path = hf_hub_download(
        repo_id="juithealien/manga109-segmentation-bubble",
        filename="best.pt",
        cache_dir="/home/sagnik/Projects/docker-composes/manga-library/data/worker/huggingface"
    )
    print(f"Downloaded model checkpoint to {pt_path}")
    
    print("Loading model via Ultralytics...")
    model = YOLO(pt_path)
    
    print("Exporting model to ONNX with imgsz=1280...")
    # This generates best.onnx in the same folder as pt_path
    onnx_path_temp = model.export(format="onnx", imgsz=1280, simplify=True)
    print(f"Exported to temporary path: {onnx_path_temp}")
    
    dest_path = os.path.join(dest_dir, "yolo11n_bubble.onnx")
    shutil.move(onnx_path_temp, dest_path)
    print(f"Moved ONNX model to final path: {dest_path}")
    
    sha256 = get_sha256(dest_path)
    print(f"SHA-256 Checksum: {sha256}")

if __name__ == "__main__":
    main()
