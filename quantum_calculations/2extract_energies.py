""" This code will generate 
        1) gaussian com files for log files with error l103, l502 and l9999
        2) an array slurm job submission file for these error com files 
        3) a configuration file for this error files array job
        4) an array slurm job submission file for the next energy com files 
        5) a configuration file for this next energy array job
    Following this, the jobs will be automatically submitted.

    The user needs to adjust 
        1) project_path, xyz_path and completed_energy_level in the main function
        2) the slurm_template for their HPC system
        3) if desired, the calculation parameters can be adjusted in the main function"""

import os, os.path, shutil, fileinput,sys

def gaussian_error_determination(project_path, cpus_per_task, memory, theory_level, completed_energy_level, charge):  
    #folder and list setup
    com_path=f"{project_path}/E_{completed_energy_level}/com/"
    log_path=f"{project_path}/E_{completed_energy_level}/log/"
    os.makedirs(log_path, exist_ok=True)
    error_path=f"{project_path}/E_{completed_energy_level}/error/"
    os.makedirs(error_path, exist_ok=True)
    
    molecule_list = [file[:-8] for file in os.listdir(f"{com_path}") if file.endswith(".log")]
    successful_molecule_list=[]; fixed_error_molecule_list=[]; error_molecule_list=[]; error_lists = {"103": [], "301": [], "301_H": [], "301_basis_set": [], "502": [], "9999": []}

    #define log files depending on errors in file or normal termination
    for molecule in range(len(molecule_list)):
        sys.stdout.write(f"\rSorting error files {molecule + 1}/{len(molecule_list)}        ")
        sys.stdout.flush()

        with open(f"{com_path}{molecule_list[molecule]}_E_{completed_energy_level}.log", "r") as log_file:
            log_file_content = log_file.read()

        lines = [line.strip() for line in log_file_content.splitlines() if line.strip()]
        last_line = lines[-1]

        if last_line.startswith("Normal termination of Gaussian"):
            shutil.move(f"{com_path}{molecule_list[molecule]}_E_{completed_energy_level}.log", f"{log_path}{molecule_list[molecule]}_E_{completed_energy_level}.log")
            successful_molecule_list.append(molecule_list[molecule])
            error="None"

        #l103 (internal coordinates have inherent limitations, and this problem may occur when several atoms line up exactly during the optimization process.) error molecules (moving, fixing+listing) 
        elif ("Error termination via Lnk1e in /opt/software/gaussian/g16/l103.exe" in log_file_content or "Error termination via Lnk1e in /opt/software/gaussian/g16_avx/g16/l103.exe" in log_file_content):
            error="103"

        #l301 (missing H) or atom not in basis set error molecules, moving+listing ONLY, basis set needs to be changed or missing H need to be added) 
        elif ("Error termination via Lnk1e in /opt/software/gaussian/g16/l301.exe" in log_file_content or "Error termination via Lnk1e in /opt/software/gaussian/g16_avx/g16/l301.exe" in log_file_content):
            if "The combination of multiplicity" in log_file_content:
                error="301_H"
            elif "Atomic number out of range" in log_file_content:
                error="301_basis_set"
            else:
                error="301"

        #l502 (scf convergence failure) error molecules (moving, fixing+listing) 
        elif ("Error termination via Lnk1e in /opt/software/gaussian/g16/l502.exe" in log_file_content or "Error termination via Lnk1e in /opt/software/gaussian/g16_avx/g16/l502.exe" in log_file_content):
            error="502"

        #l9999 (need longer/more cycles, only give extra cycles once, if not fixed the issue might be something else) error molecules (moving, fixing+listing) 
        elif ("Error termination via Lnk1e in /opt/software/gaussian/g16/l9999.exe" in log_file_content or "Error termination via Lnk1e in /opt/software/gaussian/g16_avx/g16/l9999.exe" in log_file_content):
            error="9999"

        #other error molecules molecules and molecules still running (moving+listing !!ONLY!!)                                                                                 
        else:
            #shutil.move(f"{com_path}{molecule_list[molecule]}_E_{completed_energy_level}.log", f"{error_path}{molecule_list[molecule]}_E_{completed_energy_level}.log")
            error_molecule_list.append(molecule_list[molecule])
            error="other"

        #move and list log file based on error
        if error in ["103", "301", "301_H", "301_basis_set", "502", "9999"]:
            error_subfolder_path=f"{error_path}/l{error}/"
            os.makedirs(error_subfolder_path, exist_ok=True)
            shutil.move(f"{com_path}{molecule_list[molecule]}_E_{completed_energy_level}.log", f"{error_subfolder_path}{molecule_list[molecule]}_E_{completed_energy_level}.log")
            error_lists[error].append(molecule_list[molecule])

            #edit next energy calculation to read chk_2 instead of chk
            with fileinput.input(f"{project_path}E_{completed_energy_level+1}/com/{molecule_list[molecule]}_E_{completed_energy_level+1}.com", inplace=True) as com_file:
                for line in com_file:
                    print(line.replace(f"{completed_energy_level}.chk", f"{completed_energy_level}_2.chk"), end="")
                
            #edit current energy calculation to try fix the error and change chk to chk_2
            if error in ["103", "301_H", "502"]:
                with fileinput.input(f"{com_path}{molecule_list[molecule]}_E_{completed_energy_level}.com", inplace=True) as com_file:
                    for line in com_file:
                        if error == "103":
                            #should fic the problem of several atoms lining up exactly, but may take longer
                            line=line.replace("opt", "opt=cartesian")
                        elif error == "502":
                            #Using Fermi broadening, quadratic convergence and not using the default incremental fock can help with convergence, bt increases computational time. In most cases this should solve the issue.
                            line = line.replace(f" {theory_level} ", f" {theory_level} SCF=Fermi SCF=Noincfock SCF=QC ")
                        line=line.replace(f"{completed_energy_level}.chk", f"{completed_energy_level}_2.chk")
                        print(line, end="")
            #error 9999 needs a new com file without the coordinates, as it reads the geometry from the chk file and continues the calculation from there.
            elif error == "9999":
                if completed_energy_level == 2 or completed_energy_level == 3:
                    multiplicity = f"{charge},2"
                if completed_energy_level == 1 or completed_energy_level == 4:
                    multiplicity = "0,1"
                if completed_energy_level == 2 or completed_energy_level == 4:
                    job_line = f"#p {theory_level} freq=HPModes guess=read geom=check"
                if completed_energy_level == 1 or completed_energy_level == 3:
                    job_line = f"#p opt {tehory_level} guess=read geom=check freq=HPModes"
                gaussian_template = f"""%nprocshared={cpus_per_task}
%mem={memory}
%oldchk={project_path}/E_{completed_energy_level}/chk/{molecule_list[molecule]}_E_{completed_energy_level}.chk
%chk={project_path}/E_{completed_energy_level}/chk/{molecule_list[molecule]}_E_{completed_energy_level}_2.chk
{job_line}

{molecule_list[molecule]}_2

{multiplicity}

"""
                with open(f"{com_path}{molecule_list[molecule]}_E_{completed_energy_level}.com", "w") as com_file:
                    com_file.write(gaussian_template)

            #list the errors that were automatically attempted to be fixed and do not need manual inspection
            if error in ["103", "502", "9999"]:
                fixed_error_molecule_list.append(molecule_list[molecule])

    print(f"\nERROR l301 (missing H & other) - manually add H & check = {error_lists['301_H']} & {error_lists['301']}")
    print(f"ERROR l301 (atoms not in basis set) - adjust basis set = {len(error_lists['301_basis_set'])}")
    print(f"ERROR (other) - check log files in {com_path}= {error_molecule_list}")
    print(f"Number of l9999, l502 & l103 - already rerunning, check if fixed once job completed = {len(error_lists['9999'])}, {len(error_lists['502'])} & {len(error_lists['103'])}")
    print(f"Number of successful molecules = {len(successful_molecule_list)}")

    return fixed_error_molecule_list, successful_molecule_list

def create_sh_file(project_path, molecule_list, cpus_per_task, error_name, completed_energy_level, target, partition):
    #folder setup
    slurm_output_path = os.path.join(f"{project_path}/slurm_out/")
    
    #configuration file for array job             
    with open (f"{project_path}/E_{completed_energy_level}/config{error_name}.txt", "w") as config_file:
        config_file.write("ArrayTaskID    Sample\n")
        for index, molecule in enumerate(molecule_list):
            config_file.write(f"{index}               {molecule}\n")
    
    #.sh file
    slurm_template = f"""#!/bin/bash
#SBATCH --export=ALL
#SBATCH --partition={partition}
#SBATCH --account=your/account
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time=48:00:00
#SBATCH --job-name=E_{completed_energy_level}_{target}{error_name}
#SBATCH --output={slurm_output_path}/{error_name}_E_{completed_energy_level}-%j.out

#SBATCH --array=0-{len(molecule_list)-1}
config={project_path}/E_{completed_energy_level}/config{error_name}.txt
molecule=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {{print $2}}' $config)
 
module purge
module load gaussian
 
#=========================================================
# Prologue script to record job details 
/opt/software/scripts/job_prologue.sh
#----------------------------------------------------------
 
g16 {project_path}/E_{completed_energy_level}/com/${{molecule}}_E_{completed_energy_level}.com
 
#=========================================================
# Epilogue script to record job endtime and runtime 
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------
"""
    
    with open(f"{project_path}/E_{completed_energy_level}/E_{completed_energy_level}{error_name}.sh", "w") as f:
        f.write(slurm_template)
    print(f"SLURM script and config file created in {project_path}")

def extract_energy(completed_energy_level,xyz_path, project_path, successful_molecule_list):
    successful_molecule_list = [file[:-8] for file in os.listdir(f"{project_path}/E_{completed_energy_level}/log/") if file.endswith(".log")]
    print(f"Number of successful molecules = {len(successful_molecule_list)}")
    
    E_1_list = []; E_2_list = []; E_3_list = []; E_4_list = []

    import cclib
    import pandas as pd 
    
    for index, molecule in enumerate(successful_molecule_list):
        for energy in range (4):
            sys.stdout.write(f"\rExtracting E{energy+1} for {index +1}/{len(successful_molecule_list)}                      ")
            sys.stdout.flush()

            #parse energies, band gap, entropy and enthalpy using cclib parser                              
            parser = cclib.io.ccopen(f"{project_path}E_{energy+1}/log/{molecule}_E_{energy+1}.log")
            data = parser.parse()
            E = data.scfenergies[-1]
            
            if energy == 0:
                E_1_list.append(float(E))         
            elif energy == 1:
                E_2_list.append(float(E))
            elif energy == 2:
                E_3_list.append(float(E))
            elif energy == 3:
                E_4_list.append(float(E))
      
        #Calculate energies 
        E_total_list.append(((E_4_list[index]-E_1_list[index])+(E_2_list[index]-E_3_list[index]))/2)

    #write excel sheet
    df = pd.DataFrame({"Reorganisation energy": E_total_list}, index=successful_molecule_list)
    df.to_csv(f"{project_path}Reorganisation_energy.csv")
    
    print(f"Results saved in {project_path}Reorganisation_energy.csv")
    return successful_molecule_list

def main():
#######################################
    #define parameters here
    project_path = "/path/to/your/project/"

    charge = "-1" #-1 for electron transport, 1 for hole transport
    completed_energy_level = 1 #adjust this after each energy calculation has finished running to check for errors and queue up the next energy calculation

    #parameters for next calculations
    theory_level = "b3lyp/3-21g*"
    partition="standard"
    cpus_per_task = "40"
    memory="170GB"
#######################################

    #determine errors in files and prepare input files to fix errors and input files for the next energy calculation
    fixed_error_molecule_list, successful_molecule_list = gaussian_error_determination(project_path, cpus_per_task, memory, theory_level, completed_energy_level, charge)

    
    if len(fixed_error_molecule_list) > 0: #if there were errors submit the calculations to try fix them
        error_name = "_error" #for naming the jobs, leave empty if not an error
        create_sh_file(project_path, fixed_error_molecule_list, cpus_per_task, error_name, completed_energy_level, target, partition)
        os.system(f"sbatch {project_path}/E_{completed_energy_level}/E_{completed_energy_level}{error_name}.sh")

    if completed_energy_level < 4: #run next energy (except for the last energy already being completed)
        error_name = "" #for naming the jobs, leave empty if not an error
        next_energy_level=completed_energy_level+1
        create_sh_file(project_path, successful_molecule_list, cpus_per_task, error_name, next_energy_level, target, partition)
        os.system(f"sbatch {project_path}/E_{next_energy_level}/E_{next_energy_level}{error_name}.sh")
    
    elif completed_energy_level == 4: #when all 4 energies are completed, extract the reorganisation energy
        successful_molecule_list = extract_energy(completed_energy_level,xyz_path, project_path, successful_molecule_list)
        
if __name__ == "__main__":
    main()
