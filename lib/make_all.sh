# Set the base path
PCFG_CKY_INSIDE_OUTSIDE_BASE_PATH=pcfg-cky-inside-outside

# Run cmake
cd "$PCFG_CKY_INSIDE_OUTSIDE_BASE_PATH"

cmake .

# List of binary names
binary_names=('syntax_analysis' 'phase_convert' 'train_pcfg' 'distributed_training_main' 'distributed_training_moderator')

# Build each binary
for binary_name in "${binary_names[@]}"; do
    make "$binary_name" -j
done
cd -

