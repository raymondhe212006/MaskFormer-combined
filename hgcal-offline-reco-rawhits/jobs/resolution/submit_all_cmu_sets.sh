#!/bin/bash
# Submits one eval job per (particle, energy) of the CMU single-particle sets. Environment: see eval_cmu_set.sh;
# SKIP="photons_E100 pions_E2" skips sets; PARTICLES / ENERGIES override the defaults.
SKIP=${SKIP:-}; PARTICLES=${PARTICLES:-"photons pions K0L"}; ENERGIES=${ENERGIES:-"2 3 5 10 25 50 75 100 200 300"}
HERE=$(dirname "${BASH_SOURCE[0]}")
for p in $PARTICLES; do for E in $ENERGIES; do
  n=${p}_E${E}; case " $SKIP " in *" $n "*) echo "skip $n"; continue;; esac
  echo -n "$n: "; sbatch -J "$n" "$HERE/eval_cmu_set.sh" "$p" "$E" "$@"
done; done
