FOLDERS=(
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_100GeV_1k
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_10GeV_1k
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_200GeV_1k
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_25GeV_1k
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_2GeV_1k
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_300GeV_1k
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_3GeV_1k
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_50GeV_1k
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_5GeV_1k
    /store/group/hgcal-ml-reconstruction/training_datasets/photons_noFineCalo_with_tracksters_75GeV_1k
)

BASE_FOLDER="/eos/user/m/mmarcheg/ml4reco/data/"

echo "Copying files from the following folders locally to $BASE_FOLDER:"
for folder in "${FOLDERS[@]}"; do
    echo " - $folder"
    subfolder=$(basename $folder)
    xrdcp -r "root://cmsdata.phys.cmu.edu/$folder" "$BASE_FOLDER/$subfolder/parquet"
done
