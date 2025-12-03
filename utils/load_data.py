import pandas as pd
import numpy as np
import os
from glob import glob

# Load skin lesion data from the HAM10000 dataset
# The dataset contains images of skin lesions along with their metadata.
def load_skin_data(dataset_dir, metadata_path, lesion_type_dict=None) -> pd.DataFrame:
    
    imageid_path_dict = {
    os.path.splitext(os.path.basename(x))[0]: x
    for x in glob(os.path.join(dataset_dir, 'HAM10000_images_part_*', '*.jpg'))
}
    
    df = pd.read_csv(metadata_path)
    df['path'] = df['image_id'].map(imageid_path_dict.get)
    df["cell_type"] = df["dx"].map(lesion_type_dict.get) if lesion_type_dict else df["dx"]
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
    
    
    return df

