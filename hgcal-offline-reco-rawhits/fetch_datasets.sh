BASE_DIR=${1:-$(pwd)/data}
echo "Base directory: $BASE_DIR"

xrootd_prefix=root://cmsdata.phys.cmu.edu/
training_data_dir=/store/group/hgcal-ml-reconstruction/training_datasets/taus_noFineCalo_with_tracksters_50k
evaluation_data_dirs=(
    /store/group/hgcal-ml-reconstruction/eval_parquets/K0L
    /store/group/hgcal-ml-reconstruction/eval_parquets/photons
    /store/group/hgcal-ml-reconstruction/eval_parquets/pions
)

mkdir -p $BASE_DIR/train
mkdir -p $BASE_DIR/eval

# Download training data in $BASE_DIR/train with parallel downloads
echo "Downloading training data..."
xrdcp -f -N --parallel 8 ${xrootd_prefix}${training_data_dir}/* $BASE_DIR/train/

# Download evaluation data in $BASE_DIR/eval with parallel downloads
echo "Downloading evaluation data..."
for eval_dir in "${evaluation_data_dirs[@]}"; do
    xrdcp -f -N --parallel 8 ${xrootd_prefix}${eval_dir}/* $BASE_DIR/eval/
done
