import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vision_transformers import ViTForClassfication, load_experiment
import math
from torch.nn import functional as F
import os
import json

def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = ViTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    # Use map_location to load the model on CPU if CUDA is not available
    model.load_state_dict(torch.load(cpfile, map_location=torch.device('cpu')))
    return config, model, train_losses, test_losses, accuracies

def load_model(experiment_name, device):
    config, model, _, _, _ = load_experiment(experiment_name)
    model = model.to(device)
    model.eval()
    return model, config

def preprocess_frame(frame, transform):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    # Apply the transformation
    tensor_image = transform(pil_image)
    return tensor_image.unsqueeze(0)  # Add batch dimension

def get_attention_map(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        _, attention_maps = model(image_tensor, output_attentions=True)
    
    # Concatenate the attention maps from all blocks
    attention_maps = torch.cat(attention_maps, dim=1)
    # Select only the attention maps of the CLS token
    attention_maps = attention_maps[:, :, 0, 1:]
    # Average the attention maps of the CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)
    
    # Reshape and resize the attention map
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    
    return attention_maps[0].cpu().numpy()

def apply_attention_to_frame(frame, attention_map):
    # Resize attention map to match frame size
    attention_map_resized = cv2.resize(attention_map, (frame.shape[1], frame.shape[0]))
    
    # Normalize attention map to 0-1 range
    attention_map_normalized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
    
    # Apply colormap to attention map
    attention_heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_normalized), cv2.COLORMAP_JET)
    
    # Blend original frame with attention heatmap
    output_frame = cv2.addWeighted(frame, 0.7, attention_heatmap, 0.3, 0)
    
    return output_frame

def process_video(input_video_path, output_video_path, model, config, device):
    # Set up video capture
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Set up image transformation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        tensor_frame = preprocess_frame(frame, transform)
        
        # Get attention map
        attention_map = get_attention_map(model, tensor_frame, device)
        
        # Apply attention to frame
        output_frame = apply_attention_to_frame(frame, attention_map)
        
        # Write the frame
        out.write(output_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = 'vit-with-100-epochs'  # Update this to match your saved model's name
    input_video_path = 'adolfo.mp4'  # Update this to your input video path
    output_video_path = 'output_video_with_attention.mp4'
    
    # Load the model
    model, config = load_model(experiment_name, device)
    
    # Process the video
    process_video(input_video_path, output_video_path, model, config, device)
    
    print(f"Video processing complete. Output saved to {output_video_path}")

if __name__ == "__main__":
    main()