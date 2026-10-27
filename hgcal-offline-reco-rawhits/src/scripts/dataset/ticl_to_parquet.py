import uproot, numpy as np, awkward as ak
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import awkward as ak
import os, pathlib
#import ROOT
#from utils import *


#inputFolder = "/afs/cern.ch/work/m/mmarcheg/private/ml4reco/example_ticl/CMSSW_15_1_0/src/production_tests/histo"
#fileList    = glob(f"{inputFolder}/*.root")
fileList = ["/home/mmarcheg/work/ml4reco/test_ticl/histo.root"]
Nfiles = len(fileList)
#print("Found", len(fileList), f"ROOT files, will process {Nfiles}")

tmp_store = defaultdict(list)

for f in tqdm(fileList[:Nfiles], desc="reading"):
    with uproot.open(f)["ticlDumper"] as tdir:
        basenames = {key.rsplit(";", 1)[0] for key in tdir.keys()}
        for base in basenames:
            tree = tdir[base] 
            tmp_store[base].append(tree.arrays(library="ak"))

dump_data = {name: ak.concatenate(arr_list) for name, arr_list in tmp_store.items()}
print("Trees read:", list(dump_data))


outdir = pathlib.Path("parquet_out")
outdir.mkdir(exist_ok=True)

# Combine all collections into a single record array
combined_data = ak.zip({name: arr for name, arr in dump_data.items()}, depth_limit=1)

# Save to a single parquet file
output_path = outdir / "ticl_output.parquet"
ak.to_parquet(combined_data, output_path, compression=None)
print("Wrote", output_path, "with collections:", list(dump_data.keys()))
