import json

if __name__ == "__main__":
    with open('./org_merge_dataset/std_name2uniprot_id_all_delete_55_target.json', 'r', encoding='utf-8') as f:
        std_name2uniprot_id = json.load(f)
       
    with open('./merge_dataset/merge_data_pos_40_neg_30_tasks.json', 'r', encoding='utf-8') as f:
        std_name_ls = json.load(f)     

    target_640_std_name2uniprot_id = {}
    our_target_uniprot_id = []
    not_in_ls = []
    for std_name in std_name_ls:
        if std_name in std_name2uniprot_id:
            target_640_std_name2uniprot_id[std_name] = std_name2uniprot_id[std_name]
            our_target_uniprot_id.append(std_name2uniprot_id[std_name])
        else:
            not_in_ls.append(std_name)
    print(len(not_in_ls))   # 0

    with open('./merge_dataset/target_640_std_name2uniprot_id.json', 'w') as f:
        json.dump(target_640_std_name2uniprot_id, f)

    with open('./merge_dataset/target_640_uniprot_id.json', 'w') as f:
        json.dump(our_target_uniprot_id, f)