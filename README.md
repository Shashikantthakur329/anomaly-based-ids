
# IDS using transformer model

- This is an NIDS(Network based Intrusion detection system) in which live incoming  network packets are processed and classified as anamolous or not. 
- This model uses tarnsformer model, which has excellence capability in processing sequential data like network packets, etc.
- It is trained on well known dataset NF-UNSW-nf-v2, in which attacks like Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms are detected.
- Accuracy of model is 97%, with f1-score of 90.54%.
- Model is also performing well on JIIS dataset, which aims to detect communications present in covert channels. Its accuracy in detection is 89%, with f1-score of 85%.


## Run the code
- Dataset of JIIS23 is contained in /transformer_model/dataset/JIIS23-23-dataset-main/case1/ folder.
- To run the code, run main.py file.
- The data_reader.py file reads data present in sql folder.

## Capture live data:
- For capturing the live flow of data, you need to use nProbe tool and save the captured data in mysql database.
- Captured flow should be in UNSW-NF-v2 format.
- run data_reader.py file to do live capturing.

