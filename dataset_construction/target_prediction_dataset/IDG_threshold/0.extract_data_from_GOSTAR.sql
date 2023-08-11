-- Use the following script to extract the  target small molecule structure-activity data from the GOSTAR database(https://www.gostardb.com/) through MySQL:


select gvk_id,str_id,sub_smiles,mol_weight,activity_prefix,activity_type,assay_type,activity_uom,activity_value,common_name,enzyme_cell_assay,`source`,standard_name from gostar.all_activity_gostar,gostar.structure_details where gostar.all_activity_gostar.gvk_id=gostar.structure_details.gvk_id


-- Put the extracted data into the target_prediction_data folder