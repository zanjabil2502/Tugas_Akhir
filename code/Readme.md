# COVID-19 prediction from Cough Sound


As a first step,

1. extract the Compare dataset into the 'dataset' folder and rename the folder as 'Compare-Data'. Then extract the CCS data in the 'Compare-Data' folder into that folder.

2. Extract the Musan dataset into the 'dataset' folder and rename the folder as 'Musan-Data'.

The next step is as follows

3. Run the code '1_Download_Dataset.ipynb', this code will download the Coswara and Coughvid datasets, then extract them into the 'dataset' folder

4. Run the code '2_Create_CSV_Files.ipynb', this code will create csv data regarding the Coswara, Coughvid, and Compare datasets and will be saved into the 'csv_files' folder

5. Run the code '3_Normalized.ipynb', this code will normalize the dataset to be used, and the normalization results will be in each folder in the dataset as in 'dataset/Coswara-Data/wav_normalized'. This code will also merge the 3 normalized datasets into one folder named 'dataset/Merge-Data' and also create CSV data related to the merged data in the 'csv_files/normalized_data' folder. In addition, CSV data will be created regarding train data, eval data, and test data into the 'csv_files/experiment_data' folder.

6. Run the code '4_Training.ipynb', this code will train the combined dataset using 5 random seeds. In this code, seeds 9,30,41,42, and 46 are used because they have high yields. The training results will be saved in the 'results' folder.

7. Run the code '5_Testing.ipynb', this code will test the model that has been trained using the test data. This test is carried out using 5 seeds that have been trained previously.
