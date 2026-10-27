VERBOSE=0
OPTIND=1
while getopts "v" opt; do
    case "$opt" in
        v) VERBOSE=1 ;;
        *) ;;
    esac
done

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGULARITY_IMAGE=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest

IMAGE_EPOCH=$(date -d "$(stat -c %y "$SINGULARITY_IMAGE")" +%s)
COMMIT_EPOCH=$(git -C $REPO_PATH log -1 --format=%ct)

if [ "$VERBOSE" -eq 1 ]; then
    echo -e "Last image update:\t  $(date -d "@$IMAGE_EPOCH")"
    echo -e "Last commit on repo:\t  $(date -d "@$COMMIT_EPOCH")"
fi

if [ "$IMAGE_EPOCH" -lt "$COMMIT_EPOCH" ]; then
    echo "⚠️  WARNING: Singularity image ($SINGULARITY_IMAGE) is older than the last commit on $REPO_PATH."
else
    echo "✅  Singularity image ($SINGULARITY_IMAGE) is up to date with the last commit."
fi

if [[ $(hostname) == "falcon"* ]]; then
    BINDINGS="-B /home/export/$USER -B /cvmfs -B /tmp --nv"
elif [[ $(hostname) == "lxplus"* ]]; then
    BINDINGS="-B /eos/cms/ -B /eos/user/${USER:0:1}/$USER -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc""
else
    BINDINGS="-B /cvmfs -B /tmp"
fi

apptainer shell $BINDINGS --nv $SINGULARITY_IMAGE