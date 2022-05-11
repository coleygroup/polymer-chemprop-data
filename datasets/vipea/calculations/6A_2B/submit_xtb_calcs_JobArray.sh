#!/bin/bash

#SBATCH --job-name=xtb
#SBATCH --output=slurm-%j-%x-%a.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --array 401-681

iA=0
iB=$SLURM_ARRAY_TASK_ID

echo "     ============================="
echo "     processing folder ${iA}_${iB}"
echo "     ============================="

# Loading the required module
source /etc/profile
module load anaconda/2022a
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/etc/profile.d/conda.sh
conda activate xtb

# xTB settings
export XTBHOME=/home/gridsan/maldeghi/.conda/envs/xtb/share/xtb/
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_STACKSIZE=4G
ulimit -s unlimited

# go to folder
cd ${iA}_${iB}

for poly_type in block alternating random; do
  if [ ! -d $poly_type ]; then
    continue
  fi
  cd $poly_type
  for mol in $( ls poly*mol ); do
    seq_id=$(echo $mol | cut -d"." -f1 | cut -d"_" -f2)
    cfm_id=$(echo $mol | cut -d"." -f1 | cut -d"_" -f3)

    # if result already available, skip
    if [ -f "vipea_${seq_id}_${cfm_id}.log" ]; then
      continue
    fi

    # run xTB opt
    xtb $mol --chrg 0 --uhf 0 --alpb benzene --opt loose > /dev/null  # too much output
    # cp/save minimized structure, which also contain energy
    xtb_opt_mol="xtbopt_${seq_id}_${cfm_id}.mol"
    mv xtbopt.mol $xtb_opt_mol

    # run xTB ipea calcs
    xtb $xtb_opt_mol --chrg 0 --uhf 0 --alpb benzene --norestart --vipea > xtbvipea.out
    # keep only results of interest
    grep "SCC IP (eV)\|SCC EA (eV)" xtbvipea.out > "vipea_${seq_id}_${cfm_id}.log"

    # cleanup xTB files
    rm charges wbo xtbopt.log xtbrestart xtbtopo.mol .xtboptok xtbvipea.out

  done
  cd ../
done
