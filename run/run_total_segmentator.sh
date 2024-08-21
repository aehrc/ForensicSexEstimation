
export TOTALSEG_WEIGHTS_PATH=../TotalSegmentator_weights/nnunet/results

# Directory containing the image files
input_dir="../Cranial CT data/Cranial CT nifti isotropic"
output_folder="../Cranial CT data/Cranial CT isotropic segmentations"

# Loop through all .nii.gz files in the input directory
for image_file in "${input_dir}"/*.nii.gz; do
    # Extract the base name of the file without the extension
    base_name=$(basename "${image_file}" .nii.gz)

    # Specify the output directory for this file, using only the output_folder and base_name
    output_dir="${output_folder}/${base_name}"

    # Ensure the output directory exists
    mkdir -p "${output_dir}"

    # Run TotalSegmentator command with the current file and its unique output directory
    TotalSegmentator -i "${image_file}" -o "${output_dir}" -ta total
done