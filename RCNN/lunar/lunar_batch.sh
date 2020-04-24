#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=lunar_rcnn
#SBATCH --mem=32000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=k.buiten@student.rug.nl
#SBATCH --output=lunar-%j.log
#SBATCH --gres=gpu:k40:2
module load Python/3.6.4-foss-2016a
mpirun python ./lunarRCNN.py