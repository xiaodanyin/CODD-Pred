import pandas as pd
from tqdm import tqdm



if __name__ == "__main__":
    cpd_df = pd.read_csv('../target_prediction_data/human_target_100p_uom.csv')
    print(len(cpd_df))

    tgt_classification_df = pd.read_csv('./our_target_classification_level1_3.csv')

    std_name_tgt_class_dict = {}
    for std_name, class_l1, class_l2, class_l3 in tqdm(
        zip(
            tgt_classification_df['standard_name'].tolist(), 
            tgt_classification_df['level_1'].tolist(),
            tgt_classification_df['level_2'].tolist(),
            tgt_classification_df['level_3'].tolist(),
            ), total=len(tgt_classification_df)
    ):
        std_name_tgt_class_dict[std_name] = f'{class_l1}&{class_l2}&{class_l3}'

    print(len(std_name_tgt_class_dict))

    target_class = []
    level_1 = []
    level_2 = []
    level_3 = []

    for tgt_name in tqdm(cpd_df['standard_name'].tolist()):
        if tgt_name in std_name_tgt_class_dict:
            tgt_class = std_name_tgt_class_dict[tgt_name]
            target_class.append(tgt_class)
            level_1.append(tgt_class.split("&")[0])
            level_2.append(tgt_class.split("&")[1])
            level_3.append(tgt_class.split("&")[2])            
        else:
            target_class.append('')
            level_1.append('')
            level_2.append('')
            level_3.append('')   

    assert len(target_class) == len(cpd_df)

    cpd_df['target_class'] = target_class
    cpd_df['level_1'] = level_1
    cpd_df['level_2'] = level_2
    cpd_df['level_3'] = level_3

    no_class = cpd_df.loc[cpd_df['target_class'] == ""]
    print(len(no_class))

    cpd_df.to_csv('../target_prediction_data/human_target_100p_uom_with_target_class.csv', index=False)




