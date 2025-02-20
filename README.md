## FFSL Seizure Detection

This repository contains the code for the Master's Thesis:  
**"Multiclass Epileptic Seizure Detection Using Few-Shot Learning with a Federated Learning Extension."**

---

### Dataset: TUH EEG Corpus

#### Request Access
To obtain the **TUH EEG Corpus**, you need to **request access** by following these steps:

1. Fill out [this form](https://isip.piconepress.com/projects/nedc/forms/tuh_eeg.pdf).
2. Follow the instructions provided and **email** the signed form to:  
   📩 **help@nedcdata.org**  
   _(Include "Download The TUH EEG Corpus" in the subject line.)_


#### Download the Dataset

**For the TUH EEG Events Corpus (TUEV):**  
The dataset contains **EEG segments labeled into six classes**:
1. **SPSW** - Spike and Sharp Wave  
2. **GPED** - Generalized Periodic Epileptiform Discharges  
3. **PLED** - Periodic Lateralized Epileptiform Discharges  
4. **EYEM** - Eye Movement  
5. **ARTF** - Artifact  
6. **BCKG** - Background  

📥 **Download the dataset via `rsync`:**
```bash
rsync -auxvL --delete nedc-eeg@www.isip.piconepress.com:data/eeg/tuh_eeg_seizure/v2.0.0/ ./data/datasets/TUEV/
```

Your username and password should have been sent via email after completing the request process.

After downloading, place the "edf" folder inside the repository directory.

 

###  **Data Preprocessing**
Before model training, for dataset preprocessing run the command:
```bash
python3 data_preprocessing.py --root_dir "path_to_dataset_directory"
```
To create the dataset split for Few-Shot Learning (FSL), run
```bash
python3 few_shot_datasplit.py
```

### **Model Training**

Train the three classifiers using the following commands:

1. FAISS
```bash
python3 fsl_faiss.py
```
2. FCNN
```bash
python3 fsl_fcnn.py
```
3. Transformer
```bash
python3 fsl_transformer.py
```
results are stored in directory `results/results_{classifier}`

### Federated Few-Shot Learning (FFSL)
As Federated Few-Shot Learning (FFSL) is part of a company project, the code cannot be made public.
However, if you are interested, feel free to send me a request

As federated few-shot learning is part of company project, the code cannot be disclosed publically, but if you ar interested send me a request.

✉️ Contact
For any questions, feel free to reach out!

📧 Email: eksysoykova@gmail.com \
🔗 GitHub Issues: Open an Issue

