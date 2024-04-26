import os
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
from generation_evaluation import my_sample
from utils import *
from model import PixelCNN
from dataset import CPEN455Dataset
from pytorch_fid.fid_score import calculate_fid_given_paths

def get_label_and_losses(model, model_input, device):
    num_classes = 4  # Assuming there are 4 classes
    batch_size = model_input.size(0)
    all_losses = torch.zeros((batch_size, num_classes)).to(device)
    
    # Iterate over each image in the batch
    for i in range(batch_size):
        single_image = model_input[i].unsqueeze(0)  # Add batch dimension
        # Iterate over each label
        for label in range(num_classes):
            expanded_label = torch.tensor([label], dtype=torch.int64).to(device)
            model_output = model(single_image, expanded_label)
            all_losses[i, label] = discretized_mix_logistic_loss(single_image, model_output)
    
    predicted_labels = torch.argmin(all_losses, dim=1)
    return predicted_labels, all_losses.detach().cpu().numpy()

def append_fid_to_csv(csv_path, fid_score):
    df = pd.read_csv(csv_path)
    fid_row = pd.DataFrame([{"id": "fid", "label": fid_score}])
    df_final = pd.concat([df, fid_row], ignore_index=True)
    df_final.to_csv(csv_path, index=False)

def update_csv_and_save_losses(data_loader, model, device, csv_path, npy_path):
    # Assuming the existing CSV has correct filename in the first column.
    df_existing = pd.read_csv(csv_path, header=None, names=['id', 'label'])
    filenames = df_existing['id'].tolist()
    predicted_labels_list = []
    all_losses_list = []

    for batch_idx, (model_input, _) in enumerate(tqdm(data_loader, desc="Processing batches")):
        model_input = model_input.to(device)
        predicted_labels, losses = get_label_and_losses(model, model_input, device)
        all_losses_list.append(losses)
        predicted_labels_list.extend(predicted_labels.cpu().numpy())

    if len(predicted_labels_list) == len(filenames):
        df_existing['label'] = predicted_labels_list  # Update labels
        df_existing.to_csv('test_results.csv', index=False, header=True)  # Save with column headers
        all_losses_array = np.concatenate(all_losses_list, axis=0)
        np.save(npy_path, all_losses_array)
    else:
        print("Error: Mismatch in the lengths of filenames and predictions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, default='data', help='Location for the dataset')
    parser.add_argument('-c', '--csv_path', type=str, default='data/test.csv', help='Path to the CSV file to update')
    parser.add_argument('-n', '--npy_path', type=str, default='test_logits.npy', help='Path to save the losses .npy file')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()

    ref_data_dir = "data/test"
    gen_data_dir = "samples"
    gen_data_dir_fid = 'samples_fid'

    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)
    
    if not os.path.exists(gen_data_dir_fid):
        os.makedirs(gen_data_dir_fid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model setup
    model = PixelCNN(nr_resnet=2, nr_filters=30, input_channels=3, nr_logistic_mix=15)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth', map_location=device))
    model.to(device).eval()
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, 15)

    # Data loader setup
    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataset = CPEN455Dataset(root_dir=args.data_dir, mode='test', transform=ds_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Run the update and save process
    update_csv_and_save_losses(dataloader, model, device, args.csv_path, args.npy_path)
    
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, 15)
    my_sample(model=model, gen_data_dir=gen_data_dir_fid, sample_op = sample_op)
    paths = [gen_data_dir_fid, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir_fid)), len(os.listdir(ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, 128, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir_fid))
    except:
        print("Dimension {:d} fails!".format(192))

    print("Updated CSV and saved losses.")
