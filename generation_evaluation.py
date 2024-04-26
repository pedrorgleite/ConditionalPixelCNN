'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify the remaining code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 12-18)
2. Modify how you call your sample function(line 31)
'''
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import * 
from dataset import *
import os
import torch
NUM_CLASSES = len(my_bidict)

# This function should save the generated images to the gen_data_dir, which is fixed as 'samples'
sample_op = lambda x : sample_from_discretized_mix_logistic(x, 5)
def my_sample(model, gen_data_dir, sample_batch_size=50, obs=(3, 32, 32), sample_op=sample_from_discretized_mix_logistic):
    for label_name in my_bidict.keys():
        print(f"Generating for Label: {label_name}")
        sample_t = sample(model, sample_batch_size, obs, sample_op, label_name)
        sample_t = rescaling_inv(sample_t)  # Assuming normalization needs to be reversed
        # Create directory for each label if not exists
        save_images(sample_t, gen_data_dir, label = label_name)
    print("Generation complete for all labels.")
    
if __name__ == "__main__":
    ref_data_dir = "data/test"
    gen_data_dir = "samples"
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)
    #Begin of your code
    #Load your model and generate images in the gen_data_dir    
    
    model = PixelCNN(nr_resnet=2, nr_filters=30, input_channels=3, nr_logistic_mix=15)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, 15)
    model = model.to(device)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth',map_location=torch.device(device) ))
    model = model.eval()
    my_sample(model=model, gen_data_dir=gen_data_dir, sample_op = sample_op)
    #End of your code
    paths = [gen_data_dir, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir))
    except:
        print("Dimension {:d} fails!".format(192))
        
    print("Average fid score: {}".format(fid_score))
