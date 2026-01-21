# TC-ART-12-2025-004408
TC-ART-12-2025-004408 Submission
This repository contains scripts for database generation using the Gaussian software for quantum calculations, generating different machine learning descriptors and testing different machine learning algorithms.

Workflow overview
1. Run quantum_calculations/1input_files.py. This code will generate 
	1) gaussian com files for all geometric files in a folder
	2) an array slurm job submission file 
	3) a configuration file for the array job
	Following this, the jobs will be automatically submitted.

	The user needs to adjust 
	1) project_path, xyz_path and desired charge (-1 for electron transport and 1 for hole transport) in the main function
	2) the slurm_template for their HPC system
	3) if desired, the calculation parameters can be adjusted in the main function


2. Once calculations are completed, run quantum_calculations/2extract_energies.py. This code will generate 
	1) gaussian com files for log files with error l103, l502 and l9999
	2) an array slurm job submission file for these error com files and the next energy com files for successful runs 
	3) a configuration file for this error files array job and next energy array job
	4) in case of the completed energy level being = 4, a reorganisation_energies.csv will be generated with all reorganisation energies calculated
	Following this, the jobs will be automatically submitted. This can be rerun until all calculations were successfully completed.

	The user needs to adjust 
	1) project_path, the completed energy level and desired charge (-1 for electron transport and 1 for hole transport) in the main function in the main function
	2) the slurm_template for their HPC system
	3) if desired, the calculation parameters can be adjusted in the main function


3. Once all calculations have been completed and a .csv with the IDs, smiles and reorganisation energies is present, run ML/1structural_descriptors.py. This code will generate
	1) descriptor comma separated files for each desired descriptor and descriptor combination

	The user needs to adjust 
	1) target (electron/hole/both), 
	2) desired descriptors and desired manual descriptors
	3) project path and path to the reorganisation energy file and column indices


4. Run ML/2clean_descriptors.py. This code will generate
	1) cleaned descriptor comma separated files
	2) correlation matrices for each descriptor set
	3) comma separated files of removed IDs due to high correlation or missing values

	The user needs to adjust
	1) thresholds for removing IDs with high correlation
	2) desired manual descriptors
	3) project path and path to the descriptor comma separated files folder


5. Run ML/3ML.py. This code will generate
	1) optimisation.json with the best hyperparameters for each descriptor set and machine learning algorithm tested
	2) A predicted_vs_actual.csv and scatter plot of predicted vs actual values for each descriptor set and machine learning algorithm tested
	3) A results.csv with R2, RMSE and MAE for each descriptor set and machine learning algorithm tested

	The user needs to adjust the following in the ML/3parameters_ML.json
	1) project path and column indices
	2) desired descriptors and models to test
	3) desired targets to predict (electron/hole/both)
	4) If desired, hyperparameter testing ranges can be adjusted

	
6. To test graph neural networks, run ML/GNN.py. This will generate
	1) best_hyperparameters.json with the best hyperparameters
	2) final_model_results.json with R2 and RMSE for each target
	The user needs to adjust the following in the ML/3parameters_ML.json
	1) project path
	2) column names for smiles and targets
