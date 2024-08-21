#!/bin/bash
#!/datasets/work/hb-c-radiation/work/Python_env/miniconda3/envs/myenv_conda_ct_gender/bin python3

#SBATCH --account=OD-231108
#SBATCH --job-name="preprocessing"
#SBATCH --time=1:00:00
#SBATCH --output=preprocessing.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

# Source Conda setup script
source /datasets/work/hb-c-radiation/work/Python_env/miniconda3/etc/profile.d/conda.sh

cd /datasets/work/hb-c-radiation/work/Python_env/miniconda3/envs
conda activate myenv_conda_ct_gender

# Print current date
date

PYTHONPATH="/datasets/work/hb-c-radiation/work/Python_env/miniconda3/envs/myenv_conda_ct_gender/lib/python3.9/site-packages/":"${PYTHONPATH}"
export PYTHONPATH

cd /datasets/work/hb-c-radiation/work/Python_env/myenv_ct_gender/cranial_ct_gender_classification/deployment/AnthroAI/run

srun -n1 python3.9 skull_region_cropping.py \
--image_folder "/datasets/work/hb-radiationtqa/work/Cranial CT data/Cranial CT nifti isotropic" \
--skull_seg_folder "/datasets/work/hb-radiationtqa/work/Cranial CT data/Cranial CT isotropic segmentations torch2" \
--cropped_image_folder "/datasets/work/hb-radiationtqa/work/Cranial CT data/Cranial CT nifti isotropic crop torch2" \
--cropped_skull_seg_folder "/datasets/work/hb-radiationtqa/work/Cranial CT data/Cranial CT isotropic segmentations crop torch2"