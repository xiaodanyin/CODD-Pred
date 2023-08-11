# Method for constructing target prediction dataset
## The process is as follows:
    - Extract data: 0.extract_data_from_GOSTAR.sql
    - Data cleaning: 1.codd_target_class.py; 
                     2.add_target_class.py
                     3.target_split.py
                     4.smiles_canonical.py
                     5.merge_table.py
                     6.multi_class_dataset.py
                     7.split_dataset.py
                     8.convert_data_to_chemprop.py                                   
## GOSTAR compound ID
    - The GOSTAR IDs (gvk_id, str_id) of compounds used for target prediction modeling can be found under the './target_data/split_data' folder.
