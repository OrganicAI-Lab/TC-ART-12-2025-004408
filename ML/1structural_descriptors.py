import os, sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, MACCSkeys, Descriptors, rdFingerprintGenerator

def extract_smiles(smiles_path, smiles_column, ID_column, hole_reorganisation_energy_column, electron_reorganisation_energy_column):
    csv_dataframe=pd.read_csv(smiles_path)
    ID_list=csv_dataframe.iloc[:, ID_column-1].tolist()
    smiles_list= csv_dataframe.iloc[:, smiles_column-1].tolist()
    hole_reorganisation_energy_list=csv_dataframe.iloc[:, hole_reorganisation_energy_column-1].tolist() if hole_reorganisation_energy_column is not None else None
    electron_reorganisation_energy_list=csv_dataframe.iloc[:, electron_reorganisation_energy_column-1].tolist() if electron_reorganisation_energy_column is not None else None
    print("Smiles extracted = "+str(len(smiles_list)))

    return(smiles_list, ID_list, hole_reorganisation_energy_list, electron_reorganisation_energy_list)

def get_mol_file(smiles):
    mol_file = Chem.MolFromSmiles(smiles)
    mol_file_h = Chem.AddHs(mol_file)

    return mol_file, mol_file_h

def morgan(mol_file, morgan):
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    morgan_fp = morgan_generator.GetFingerprint(mol_file)
    morgan.append(list(morgan_fp))

def rdkit(mol_file, rdkit):
    rdkit.append(Descriptors.CalcMolDescriptors(mol_file))
    
def daylight(mol_file, daylight):
    daylight_fp= Chem.RDKFingerprint(mol_file)
    daylight.append(list(daylight_fp))

def maccs(mol_file, maccs):
    maccs_fp= MACCSkeys.GenMACCSKeys(mol_file)
    maccs.append(list(maccs_fp))

def manual(mol_file, mol_file_h, manual_descriptors_data_list, desired_manual_descriptors_list):
    for desired_manual_descriptor in desired_manual_descriptors_list:
        if desired_manual_descriptor == "dou":
            atom_counts = {'C': 0, 'Si': 0, 'P': 0, 'B': 0, 'H': 0, 'N': 0, 'X': 0}

            for atom in mol_file_h.GetAtoms():
                sym = atom.GetSymbol()
                if sym in atom_counts:
                    atom_counts[sym] += 1
                elif sym in ['F', 'Cl', 'Br', 'I']:
                    atom_counts['X'] += 1

            C  = atom_counts['C']
            Si = atom_counts['Si']
            H  = atom_counts['H']
            N  = atom_counts['N']
            X  = atom_counts['X']

            # Valence: C/Si = 4, P = 5, B = 3, N = 3, X = 1, H = 1
            dbe = (C + Si) - (H + X)/2 + (N)/2 + 1
            dou_number=round(dbe,0)
            manual_descriptors_data_list[desired_manual_descriptor].append(dou_number)

        #atom counts
        elif desired_manual_descriptor == "atom_count": 
            manual_descriptors_data_list[desired_manual_descriptor].append(mol_file_h.GetNumAtoms())

        elif desired_manual_descriptor == "heteroatom_count": 
            manual_descriptors_data_list[desired_manual_descriptor].append(rdMolDescriptors.CalcNumHeteroatoms(mol_file_h))

        elif desired_manual_descriptor == "sp2c_count": 
            sp2_carbons = 0  
            for atom in mol_file.GetAtoms():
                if atom.GetAtomicNum() == 6:  #for all carbons, check hybridisation
                    if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                        sp2_carbons += 1
            manual_descriptors_data_list[desired_manual_descriptor].append(sp2_carbons)

        elif desired_manual_descriptor == "sp3c_count": 
            sp3_carbons = 0    
            for atom in mol_file.GetAtoms():
                if atom.GetAtomicNum() == 6:  #for all carbons, check hybridisation
                    if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
                        sp3_carbons += 1
            manual_descriptors_data_list[desired_manual_descriptor].append(sp3_carbons)
            
        #molecular weight
        elif desired_manual_descriptor == "mw": 
            manual_descriptors_data_list[desired_manual_descriptor].append(rdMolDescriptors.CalcExactMolWt(mol_file_h))
    
        #bonds
        elif desired_manual_descriptor == "bond_count": 
            manual_descriptors_data_list[desired_manual_descriptor].append(mol_file_h.GetNumBonds())

        elif desired_manual_descriptor == "rotatable_bond_count": 
            manual_descriptors_data_list[desired_manual_descriptor].append(rdMolDescriptors.CalcNumRotatableBonds(mol_file_h))

        elif desired_manual_descriptor == "cyclic_bond_count": 
            manual_descriptors_data_list[desired_manual_descriptor].append(sum(1 for bond in mol_file_h.GetBonds() if bond.IsInRing()))

        elif desired_manual_descriptor == "conjugated_bond_count": 
            manual_descriptors_data_list[desired_manual_descriptor].append(sum(1 for bond in mol_file_h.GetBonds() if bond.GetIsConjugated()))
        
        elif desired_manual_descriptor == "h_bond_count": 
            including_h_bonds = mol_file_h.GetNumBonds()
            not_including_h_bonds = mol_file.GetNumBonds()
            manual_descriptors_data_list[desired_manual_descriptor].append(including_h_bonds-not_including_h_bonds)

        #rings
        elif desired_manual_descriptor == "ring_count": 
            manual_descriptors_data_list[desired_manual_descriptor].append(rdMolDescriptors.CalcNumRings(mol_file_h))

        elif desired_manual_descriptor == "aromatic_ring_count": 
            manual_descriptors_data_list[desired_manual_descriptor].append(rdMolDescriptors.CalcNumAromaticRings(mol_file_h))

    return manual_descriptors_data_list

def save_as_csv(target, descriptor_path, desired_manual_descriptors_list, desired_descriptor, data, descriptor_dfs, ID_list, hole_reorganisation_energy_list, electron_reorganisation_energy_list):
    #first columns with ID and reorganisation energies
    base = {"ID" : ID_list}
    if "hole" in target:
        base["Hole Reorganisation Energy"] = hole_reorganisation_energy_list
    if "electron" in target:
        base["Electron Reorganisation Energy"] = electron_reorganisation_energy_list
    base_df = pd.DataFrame(base)

    #add descriptors solo
    if desired_descriptor in ["morgan", "daylight", "maccs"]:
        descriptor_df = pd.DataFrame(data, columns=[f"{desired_descriptor}_{i}" for i in range(len(data[0]))])
    elif desired_descriptor == "rdkit":
        descriptor_df = pd.DataFrame(data)
    elif desired_descriptor == "manual":
        descriptor_df = pd.DataFrame({desired_manual_descriptor: data[desired_manual_descriptor] for desired_manual_descriptor in desired_manual_descriptors_list})
    descriptor_dfs[desired_descriptor] = descriptor_df

    df = pd.concat([base_df, descriptor_df], axis=1)
    df.to_csv(f"{descriptor_path}{desired_descriptor}.csv", index = False)

    return base_df, descriptor_dfs

def save_combo_csv(descriptor_path, desired_descriptors_list, base_df, descriptor_dfs):
    #combination of descriptors
    descriptor_combinations = [["rdkit", "maccs"],["rdkit", "morgan"],["rdkit", "daylight"],["rdkit", "morgan", "daylight", "maccs"],["manual", "maccs"],["manual", "morgan"],["manual", "daylight"],["manual", "morgan", "daylight", "maccs"],["morgan", "daylight", "maccs"], ["manual", "rdkit"]]

    for combo in descriptor_combinations:
        if not set(combo).issubset(descriptor_dfs):
            continue

        combo_df = pd.concat([base_df] + [descriptor_dfs[name] for name in combo],axis=1)
        combo_df.to_csv(f"{descriptor_path}{"_".join(combo)}.csv", index=False)

def main():
#######################################
    #Define parameters here
    target = ["electron", "hole"] # ["hole", "electron"] or ["electron", "hole"] or ["electron"] or ["hole"]
    desired_descriptors_list = ["morgan", "daylight" , "maccs", "rdkit", "manual"] #choices: ["morgan", "daylight" , "maccs", "rdkit", "manual"]
    desired_manual_descriptors_list = ["dou", "mw", "atom_count", "sp2c_count", "sp3c_count", "heteroatom_count", "bond_count", "h_bond_count", "rotatable_bond_count", "conjugated_bond_count", "cyclic_bond_count", "ring_count", "aromatic_ring_count"] #choices: ["dou", "mw", "atom_count", "sp2c_count", "sp3c_count", "heteroatom_count", "bond_count", "h_bond_count", "rotatable_bond_count", "conjugated_bond_count", "cyclic_bond_count", "ring_count", "aromatic_ring_count"]

    ID_column=2 #NOT 0 based
    project_path="/path/to/your/project/"
    smiles_path=f"{project_path}test.csv"

    smiles_column=ID_column + 1
    hole_reorganisation_energy_column = ID_column + 2 #put None if not targeted
    electron_reorganisation_energy_column = ID_column + 3 #put None if not targeted
#######################################

    descriptor_path=project_path+"Descriptors/"    
    os.makedirs(descriptor_path, exist_ok=True)

    smiles_list, ID_list, hole_reorganisation_energy_list, electron_reorganisation_energy_list = extract_smiles(smiles_path, smiles_column, ID_column, hole_reorganisation_energy_column, electron_reorganisation_energy_column)
    descriptor_data_lists = {desired_descriptor: [] for desired_descriptor in desired_descriptors_list}
    manual_descriptors_data_list = {desired_manual_descriptor: [] for desired_manual_descriptor in desired_manual_descriptors_list}
    descriptor_dfs = {}

    for index,smiles in enumerate(smiles_list):   
        sys.stdout.write(f"\rExtracting descriptors {index + 1}/{len(smiles_list)}")
        sys.stdout.flush()

        mol_file, mol_file_h = get_mol_file(smiles)

        for desired_descriptor in desired_descriptors_list:
            if desired_descriptor == "manual":
                manual_descriptors_data_list = manual(mol_file, mol_file_h, manual_descriptors_data_list, desired_manual_descriptors_list)
            else:
                globals()[desired_descriptor](mol_file, descriptor_data_lists[desired_descriptor])

    for desired_descriptor in desired_descriptors_list:
        if desired_descriptor == "manual":
            base_df, descriptor_dfs = save_as_csv(target, descriptor_path, desired_manual_descriptors_list, "manual", manual_descriptors_data_list, descriptor_dfs, ID_list, hole_reorganisation_energy_list, electron_reorganisation_energy_list)
        else:
            base_df, descriptor_dfs = save_as_csv(target, descriptor_path, desired_manual_descriptors_list, desired_descriptor, descriptor_data_lists[desired_descriptor], descriptor_dfs, ID_list, hole_reorganisation_energy_list, electron_reorganisation_energy_list)

    save_combo_csv(descriptor_path, desired_descriptors_list, base_df, descriptor_dfs)

    print("\nDescriptor csv files saved")

if __name__ == "__main__":
    main()

