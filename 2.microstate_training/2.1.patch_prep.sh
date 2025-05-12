prep_path=$(dirname $(python -c 'import pyprep; print(pyprep.__file__)'))
prep_pipeline_path=$prep_path/prep_pipeline.py
reference_path=$prep_path/reference.py
find_noisy_channels_path=$prep_path/find_noisy_channels.py
echo "Copying prep_pipeline.py to: $prep_pipeline_path"
sudo cp ./prep_pipeline.py $prep_pipeline_path
sudo cp ./reference.py $reference_path
sudo cp ./find_noisy_channels.py $find_noisy_channels_path