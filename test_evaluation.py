import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
from PIL import Image
from generation_evaluation import my_sample
from utils import *
from model import *
from dataset import *
from pytorch_fid.fid_score import calculate_fid_given_paths


def get_label_and_losses(model, model_input, device):
    num_classes = 4  # Assuming there are 4 classes
    batch_size = model_input.size(0)
    all_losses = torch.zeros((batch_size, num_classes)).to(device)
    predicted_labels = torch.zeros(batch_size, dtype=torch.int64).to(device)

    # Iterate over each image in the batch
    for i in range(batch_size):
        single_image = model_input[i].unsqueeze(0)  # Add batch dimension
        # Iterate over each label
        for label in range(num_classes):
            expanded_label = torch.tensor([label], dtype=torch.int64).to(device)
            model_output = model(single_image, expanded_label)
            all_losses[i, label] = discretized_mix_logistic_loss(single_image, model_output)
        
        # Find the label with the minimum loss for this image
        predicted_label = torch.argmin(all_losses, dim=1)
        predicted_labels[i] = predicted_label
    return predicted_labels, all_losses

def append_fid_to_csv(csv_path, fid_score):
    df = pd.read_csv(csv_path)
    fid_row = pd.DataFrame([{"id": "fid", "label": fid_score}])
    df_final = pd.concat([df, fid_row], ignore_index=True)
    df_final.to_csv(csv_path, index=False)

def save_images(tensor, images_folder_path, filename):
    os.makedirs(images_folder_path, exist_ok=True)
    filename = os.path.basename(filename)
    image_path = os.path.join(images_folder_path, f"{filename}.jpg")
    img = Image.fromarray((tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), mode='RGB')
    img.save(image_path)

def process_images_and_update_csv(model, data_loader, device, csv_path, images_folder, npy_path):
    # Load DataFrame
    df = pd.read_csv(csv_path, header=None, names=['id', 'label'])
    all_losses_array = []

    # Use a list to collect updates
    updated_rows = []
    # Iterate over image filenames in the dataframe
    for index, row in df.iterrows():
        filename = row['id']  # Adjust column name as necessary
        row['id'] = os.path.basename(filename)
        image_path = os.path.join(images_folder, filename)
        
        model_input , _= data_loader.__getitem__(index)
        model_input = model_input.unsqueeze(0).to(device)

        # Predict labels and calculate losses
        predicted_labels, losses = get_label_and_losses(model, model_input, device)
        all_losses_array.append(losses)
        row['label'] = predicted_labels.item()  # Update the label in the DataFrame

        # Generate and save image
        model_output = model(model_input,predicted_labels)  # Assuming model returns generated image tensor
        sample_t = rescaling_inv(model_output)  # Normalize or scale the tensor as necessary
        save_images(sample_t.squeeze(0), images_folder, filename)  # Save the generated image

        # Collect updated row
        updated_rows.append(row)

    # Reconstruct DataFrame
    df_updated = pd.DataFrame(updated_rows)
    # Save all losses to a numpy array file and update the CSV
    np.save(npy_path, np.array(all_losses_array))
    df_updated.to_csv('test_results.csv', index=False)
    print("Updated CSV and saved losses.")

if __name__ == '__main__':

    ref_data_dir = "data/test"
    gen_data_dir = "samples"
    npy_path = 'test_logits'

    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelCNN(nr_resnet=2, nr_filters=30, input_channels=3, nr_logistic_mix=15)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth', map_location=device))
    model.to(device).eval()
    
    csv_path = 'data/test.csv'
    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataset = CPEN455Dataset(root_dir='data', mode='test', transform=ds_transforms)

    process_images_and_update_csv(model, dataset, device, csv_path, gen_data_dir, npy_path)

    # FID score calculation and appending to CSV
    paths = [gen_data_dir, ref_data_dir]
    fid_score = calculate_fid_given_paths(paths, batch_size, device, dims=2048)
    print("FID score calculated:", fid_score)
    append_fid_to_csv(csv_path, fid_score)

    print("Updated CSV and saved losses.")
