import os
import glob
import h5py
import shutil
import argparse
from tqdm import tqdm

def load_processed_hdf5(hdf_path):
    """Load data from an HDF5 file."""
    with h5py.File(hdf_path, 'r') as f:
        return {
            'signal': f['signal'][:],
            'offending_channel': f['offending_channel'][:],
            'label': f['label'][:],
        }

def parse_patient_id_from_filename(fname):
    """Extract patient ID from filename based on dataset structure."""
    base = os.path.basename(fname)
    parts = os.path.splitext(base)[0].split('_')
    return parts[0] if parts[0].isdigit() else parts[1]



def main():
    """Main function to process files and create few-shot splits."""
    # Convert unseen classes to float

    unseen_classes = [2,3,4]
    
    train_dir = "edf/processed_train"
    eval_dir = "edf/processed_eval"
    
    output_train = "edf/few_shot_train_3"
    output_eval = "edf/few_shot_eval_3"
    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_eval, exist_ok=True)

    
    all_files = glob.glob(os.path.join(train_dir, "*.h5")) + glob.glob(os.path.join(eval_dir, "*.h5"))
    patient_dict = {}
    
    for fpath in tqdm(all_files, desc="Processing files"):
        data = load_processed_hdf5(fpath)
        label = float(data['label'][0])
        patient_id = parse_patient_id_from_filename(fpath)
        
        if patient_id not in patient_dict:
            patient_dict[patient_id] = {'files': [], 'labels': set()}
        
        patient_dict[patient_id]['files'].append(fpath)
        patient_dict[patient_id]['labels'].add(label)
    
    train_patients, eval_patients = [], []

    for pid, info in tqdm(patient_dict.items(), desc="Assigning patients to train/eval"):
        # If patient has any unseen class, put them in eval
        if len(info['labels'].intersection(unseen_classes)) > 0:
            eval_patients.append(pid)
        else:
            train_patients.append(pid)
    
    # Copy files to new directories
    for pid in tqdm(train_patients, desc="Copying train files"):
        for fpath in patient_dict[pid]['files']:
            shutil.copy(fpath, output_train)
    
    for pid in tqdm(eval_patients, desc="Copying eval files"):
        for fpath in patient_dict[pid]['files']:
            shutil.copy(fpath, output_eval)
    
    print("New few-shot split created:")
    print(f"Training patients: {len(train_patients)}")
    print(f"Evaluation patients: {len(eval_patients)}")
        

if __name__ == "__main__":
    main()
