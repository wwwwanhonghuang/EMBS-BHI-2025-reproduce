prep_pipeline_path=$(dirname $(python -c 'import pyprep; print(pyprep.__file__)'))/prep_pipeline.py
echo "Copying prep_pipeline.py to: $prep_pipeline_path"
sudo cp ./prep_pipeline.py $prep_pipeline_path
