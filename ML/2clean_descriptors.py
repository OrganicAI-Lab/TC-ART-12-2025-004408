import os, shutil, sys
import pandas as pd
import networkx as nx

def remove_nans(descriptor_path, list_descriptor_input):   
    uncleaned_folder = f"{descriptor_path}uncleaned/"
    os.makedirs(uncleaned_folder, exist_ok=True)
    os.makedirs(f"{descriptor_path}removed_IDs/", exist_ok=True)

    for index,descriptor_input in enumerate(list_descriptor_input):  
        sys.stdout.write(f"\rRemoving NaNs {descriptor_input} ({index+1}/{len(list_descriptor_input)})                            ")
        sys.stdout.flush()     
         
        #moveold csv into uncleaned folder
        csv_path = f"{descriptor_path}{descriptor_input}.csv"
        uncleaned_csv_path = f"{uncleaned_folder}{descriptor_input}_uncleaned.csv"
        shutil.move(csv_path, uncleaned_csv_path)
        df = pd.read_csv(uncleaned_csv_path)
    
        #remove nans
        nan_mask = df.isna().any(axis=1)
        
        df_removed = df.loc[nan_mask].copy()
        df_clean = df.loc[~nan_mask].copy()
        
        removed_csv_path = f"{descriptor_path}removed_IDs/{descriptor_input}_removed_nan_rows.csv"
        df_removed.to_csv(removed_csv_path, index=False)

        # save cleaned dataframe back to original location
        df_clean.to_csv(csv_path, index=False)

    return uncleaned_folder

def correlations(project_path, list_descriptor_input, descriptor_column_start, desired_manual_descriptors_list, threshold, uncleaned_folder, descriptor_path):
    os.makedirs(os.path.join(project_path,"results"), exist_ok=True)
    descriptor_column_start = descriptor_column_start-1
    
    for index,descriptor_input in enumerate(list_descriptor_input):
        print(f"Processing {descriptor_input}...")
        os.makedirs(f"{project_path}results/correlations/{descriptor_input}", exist_ok=True)
        csv_path = f"{descriptor_path}{descriptor_input}.csv"
        uncleaned_correlation_csv_path = f"{uncleaned_folder}{descriptor_input}_uncleaned_correlation.csv"
        shutil.move(csv_path, uncleaned_correlation_csv_path)
        
        df_full = pd.read_csv(uncleaned_correlation_csv_path)
        df_index = df_full.iloc[:, :descriptor_column_start]
        df_features = df_full.iloc[:, descriptor_column_start:]
        #save column positions to ensure the features will be saved in the same order again and iterated over in the correct order
        column_position = {col: i for i, col in enumerate(df_features.columns)}
        original_column_order = df_features.columns.tolist()
        
        #save handpicked descriptors
        if descriptor_input.startswith("manual"):
            manual_cols = [col for col in desired_manual_descriptors_list if col in df_full.columns]
            df_handpicked_descriptors = df_full[manual_cols].copy()
    
        #%%Keep only columns that have at least two unique values (i.e. mix of 0s and 1s)
        df_filtered_zero = df_features.loc[:, (df_features != 0).any(axis=0)]
        df_filtered_one = df_filtered_zero.loc[:, (df_filtered_zero != 1).any(axis=0)]
    
        print(f"Total features retained after 0/1 filtering: {df_filtered_one.shape[1]}/{df_features.shape[1]}")
        
        #%%Correlation matrix
        corr_matrix = df_filtered_one.corr().abs()
    
        #Graph where each feature is a node, and an edge exists if correlation > threshold
        G = nx.Graph()
        for col in corr_matrix.columns:
            G.add_node(col)
    
        edge_count = 0
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > threshold:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j])
                    edge_count += 1
                    sys.stdout.write(f"\rEdge added: {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]} (corr={corr_value:.4f})                            ")
                    sys.stdout.flush()
    
        print(f"\nTotal edges added: {edge_count}")
    
        list_correlated_groups = list(nx.connected_components(G))
    
        #list of all feature names, so we can remove ones we dont want from it and list removed features
        list_features_to_keep = list(df_filtered_one.columns)
        list_removed_features = []
        
        #Remove all but one feature per correlated group, first feature is kept
        for group in list_correlated_groups:
            list_group = list(group)
            
            if len(list_group) > 1:
                #Sort the group by their positions in the original column order
                list_group_sorted = sorted(list(list_group), key=lambda x: column_position[x])
                feature_to_keep = list_group_sorted[0]
    
                for feature in list_group_sorted[1:]:
                    max_correlation = corr_matrix.loc[feature, feature_to_keep]
                    list_removed_features.append((feature, feature_to_keep, max_correlation))
                    list_features_to_keep.remove(feature)
    
        print(f"Total features retained after correlation filtering: {len(list_features_to_keep)}/{df_features.shape[1]}")
    
        #Save removed features with correlation coefficients
        df_removed = pd.DataFrame(list_removed_features, columns=["Removed Feature", "Retained Feature", "Correlation Coefficient"])
        df_removed.to_csv(f"{project_path}/results/correlations/{descriptor_input}/removed_features.csv", index=False)
        
        #order columns the same as original and filter df to only include the selected features
        features_to_keep_df = [col for col in original_column_order if col in list_features_to_keep]
        df_retained = df_filtered_one[features_to_keep_df]
        
        #%%add back in handpicked descriptors
        if descriptor_input.startswith("manual"):
            #copy handpicked back into df_retained, as they are sections of a bigger df, I have to .copy first to create their own df
            df_retained = df_retained.copy()
            df_handpicked_descriptors = df_handpicked_descriptors.copy()
            list_missing_columns = [c for c in df_handpicked_descriptors.columns if c not in df_retained.columns]
            for c in list_missing_columns:
                df_retained[c] = df_handpicked_descriptors[c].values
        
        #column ordered like original and only if still exist
        df_retained = df_retained[[col for col in original_column_order if col in df_retained.columns]]
        df_cleaned = pd.concat([df_index, df_retained], axis=1)
        df_cleaned.to_csv(csv_path, index=False)
    
        cleaned_correlation_matrix = df_cleaned.iloc[:, descriptor_column_start:].corr()
        cleaned_correlation_matrix.to_csv(f"{project_path}/results/correlations/{descriptor_input}/correlation_matrix.csv")

def main():
#######################################
#parameters
    threshold = 0.8
    desired_manual_descriptors_list = ["dou", "mw", "atom_count", "sp2c_count", "sp3c_count", "heteroatom_count", "bond_count", "h_bond_count", "rotatable_bond_count", "conjugated_bond_count", "cyclic_bond_count", "ring_count", "aromatic_ring_count"]
    descriptor_column_start = 4 #not 0 based

    project_path = "/path/to/your/project/"
    descriptor_path = f"{project_path}Descriptors/"
#######################################

    #removes Descriptors, as they should all be retained
    list_descriptor_input = [file[:-4] for file in os.listdir(descriptor_path) if file.endswith(".csv") and file[:-4] != "manual"]
    print(f"Descriptors found: {len(list_descriptor_input)} {list_descriptor_input}")

    uncleaned_folder = remove_nans(descriptor_path, list_descriptor_input)
    correlations(project_path, list_descriptor_input, descriptor_column_start, desired_manual_descriptors_list, threshold, uncleaned_folder, descriptor_path)

if __name__ == "__main__":
    main()
