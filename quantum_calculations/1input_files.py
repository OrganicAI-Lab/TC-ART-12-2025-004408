""" This code will generate
        1) folders for h or e target and for E_1 - E_4 with chk and com folder each
        2) .com files for each molecule and E_1 - E_4
        3) an array slurm job submission file 
        4) a configuration file for the array job
    Following this, the jobs will be automatically submitted.

    The user needs to adjust 
        1) project_path in the main function
        2) the slurm_template for their HPC system
        3) if desired, the theory level can be adjusted in the main function"""

import os.path, os, glob, sys

def creatue_input_files(project_path, xyz_path, molecule_list, cpus_per_task, memory, theory_level, charge):
    #folders
    out_path = os.path.join(f"{project_path}/slurm_out/")
    os.makedirs(out_path, exist_ok=True)
    for energy in range(4):
        com_path = os.path.join(f"{project_path}/E_{energy+1}/com/")
        os.makedirs(com_path, exist_ok=True)
        chk_path = os.path.join(f"{project_path}/E_{energy+1}/chk/")
        os.makedirs(chk_path, exist_ok=True)

    for index, molecule in enumerate(molecule_list):
        sys.stdout.write(f"\rReading {molecule}.xyz and writing Gaussian {molecule}.com ({index + 1}/{len(molecule_list)})                                                                     ")
        sys.stdout.flush()

        #read geometry from xyz file
        with open(f"{xyz_path}{molecule}.xyz", "r") as xyz_file:
            lines = xyz_file.readlines()
        xyz_coords_from_xyz = "".join(lines[2:]).strip()
           
        #write gaussian input file
        for energy in range(4):
            if energy == 0:
                xyz_coords=xyz_coords_from_xyz
                chk_line=""
            else:
                xyz_coords=""
                chk_line=f"%oldchk={project_path}/E_{energy}/chk/{molecule}_E_{energy}.chk"

            if energy == 0:
                job = f"#p opt {theory_level} freq=HPModes"
                charge_multiplicity = "0,1"
            elif energy == 1:
                job = f"#p {theory_level} guess=read geom=check"
                charge_multiplicity = f"{charge},2"
            elif energy == 2:
                job = f"#p opt {theory_level} guess=read geom=check freq=HPModes"
                charge_multiplicity = f"{charge},2"
            elif energy == 3:
                job = f"#p {theory_level} guess=read geom=check"
                charge_multiplicity = "0,1"

            gaussian_template = f"""%nprocshared={cpus_per_task}
%mem={memory}
{chk_line}
%chk={project_path}/E_{energy+1}/chk/{molecule}_E_{energy+1}.chk
{job}

{molecule}

{charge_multiplicity}
{xyz_coords}

"""
            
            with open(f"{project_path}/E_{energy+1}/com/{molecule}_E_{energy+1}.com", "w") as com_file:
                com_file.write(gaussian_template)
                
    print(f"\nGaussian .com files created.\n")

def create_sh_file(project_path, molecule_list, cpus_per_task, partition, target):
    #config file            
    with open (f"{project_path}/E_1/config.txt", "w") as config_file:
        config_file.write("ArrayTaskID    Sample\n")
        for index, molecule in enumerate(molecule_list):
            config_file.write(f"{index}               {molecule}\n")

    #.sh file
        slurm_template = f"""#!/bin/bash
# Propagate environment variables to the compute node
#SBATCH --export=ALL
#SBATCH --partition={partition}
#SBATCH --account=nematiaram-esrd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time=48:00:00
#SBATCH --job-name=E_1_{target}
#SBATCH --output={project_path}/slurm_out/E_1-%j.out

#SBATCH --array=0-{len(molecule_list)-1}
config={project_path}/E_1/config.txt
molecule=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {{print $2}}' $config)
 
module purge
module load gaussian
 
#=========================================================
# Prologue script to record job details 
/opt/software/scripts/job_prologue.sh
#----------------------------------------------------------
 
g16 {project_path}/E_1/com/${{molecule}}_E_1.com
 
#=========================================================
# Epilogue script to record job endtime and runtime 
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------
"""

        
    with open(f"{project_path}/E_1/E_1.sh", "w") as slurm_file:
        slurm_file.write(slurm_template)
    print(f"SLURM script and config file created in {project_path}")

def main():
#######################################
    #define parameters here
    project_path = "/path/to/your/project/"
    xyz_path = f"{project_path}/xyz/"

    charge="1" #-1 for electron transport, 1 for hole transport

    theory_level = "b3lyp/3-21g*"
    partition="standard"
    cpus_per_task = "40"
    memory="170GB"
#######################################

    #extract list of molecules from xyz folder 
    molecule_list = [file[:-4] for file in os.listdir(xyz_path) if file.endswith('.xyz')]

    if charge == "-1":
        target = "e"
        project_path = f"{project_path}/{target}/" 
    elif charge == "1":
        target = "h"
        project_path = f"{project_path}/{target}/"
    os.makedirs(project_path, exist_ok=True)

    creatue_input_files(project_path, xyz_path, molecule_list, cpus_per_task, memory, theory_level, charge)
    create_sh_file(project_path, molecule_list, cpus_per_task, partition, target)

    os.system(f"sbatch {project_path}/E_1/E_1.sh")

if __name__ == "__main__":
    main()
