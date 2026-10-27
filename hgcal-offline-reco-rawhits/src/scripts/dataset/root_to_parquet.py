import uproot
import numpy as np
import awkward as ak
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import os, pathlib
import argparse 
import gc
import pyarrow.parquet as pq
import pyarrow as pa

def parse_args():
    parser = argparse.ArgumentParser(description='Process TICL and ML root files into a single parquet file.')
    parser.add_argument('--TICLfiles', nargs='+', default=["output_ticl.root"], help='List of TICL root files')
    parser.add_argument('--nanoMLfiles', nargs='+', default=["output.root"], help='List of nanoML files')
    parser.add_argument('--outputDir', type=str, default="parquet_out", help='Output directory for parquet files')
    parser.add_argument('--outputFile', type=str, default="output.parquet", help='Output parquet filename')
    parser.add_argument('--compression', type=str, default='lz4', help='Parquet compression algorithm')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of files to process at once')

    return parser.parse_args()

def process_batch(ticl_files, ml_files):
    """Process a batch of files and return the combined data"""
    tmp_store = defaultdict(list)
    
    # Collections to keep from TICL file
    ticl_collections = ['clusters', 'ticlCandidate', 'candidates', 'ticlTrackstersCLUE3DHigh', 'simtrackstersSC', 'associations']
    
    # Read TICL files
    for f in tqdm(ticl_files, desc="  TICL", leave=False):
        with uproot.open(f)["ticlDumper"] as tdir:
            basenames = {key.rsplit(";", 1)[0] for key in tdir.keys()}
            for base in basenames:
                if base in ticl_collections:
                    tree = tdir[base] 
                    tmp_store[base].append(tree.arrays(library="ak"))
                
    # Read ML files
    for f in tqdm(ml_files, desc="  nanoML", leave=False):
        with uproot.open(f)["Events"] as tdir:
            rechit_branches = [
                'RecHitHGC_x', 'RecHitHGC_y', 'RecHitHGC_z',
                'RecHitHGC_energy', 'RecHitHGC_time',
                'RecHitHGC_MergedSimClusterBestMatchIdx',
                'RecHitHGC_MergedSimClusterBestMatchQual',
                'RecHitHGC_detId',
                'RecHitHGC_SimClusterBestMatchIdx',
                'RecHitHGC_SimClusterBestMatchQual',
            ]
            mergedsimcluster_branches = [
                'MergedSimCluster_impactPoint_eta', 'MergedSimCluster_impactPoint_phi',
                'MergedSimCluster_impactPoint_x', 'MergedSimCluster_impactPoint_y',
                'MergedSimCluster_impactPoint_z', 'MergedSimCluster_boundaryEnergy',
                'MergedSimCluster_recEnergy', 'MergedSimCluster_pdgId',
                'MergedSimCluster_trackIdAtBoundary',
                'MergedSimCluster_eventId', 'MergedSimCluster_bunchCrossing',
            ]
            simcluster_branches = [
                'SimCluster_impactPoint_eta', 'SimCluster_impactPoint_phi',
                'SimCluster_impactPoint_x', 'SimCluster_impactPoint_y',
                'SimCluster_impactPoint_z', 'SimCluster_boundaryEnergy',
                'SimCluster_recEnergy', 'SimCluster_pdgId',
                'SimCluster_trackIdAtBoundary',
                'SimCluster_eventId', 'SimCluster_bunchCrossing',
            ]
            # Per-LayerCluster best-match index into the SimCluster collection.
            # Lives in the NanoML file (unlike the LayerClusters themselves, which
            # come from the TICL "clusters" tree) so it's merged in below.
            layercluster_match_branches = [
                'LayerCluster_SimCluster_MatchIdx',
            ]

            rechit_data = tdir.arrays(filter_name=rechit_branches, library="ak")
            tmp_store["RecHitHGC"].append(rechit_data)

            mergedsimcluster_data = tdir.arrays(filter_name=mergedsimcluster_branches, library="ak")
            tmp_store["MergedSimCluster"].append(mergedsimcluster_data)

            simcluster_data = tdir.arrays(filter_name=simcluster_branches, library="ak")
            tmp_store["SimCluster"].append(simcluster_data)

            layercluster_match_data = tdir.arrays(filter_name=layercluster_match_branches, library="ak")
            tmp_store["LayerCluster_SimCluster_MatchIdx"].append(layercluster_match_data)

    # Concatenate arrays for this batch
    batch_data = {name: ak.concatenate(arr_list) for name, arr_list in tmp_store.items()}

    # Merge the NanoML LayerCluster<->SimCluster match index into the TICL "clusters"
    # collection, so downstream code can access it as a plain field of "clusters"
    # (same access pattern as the RecHitHGC<->SimCluster match fields). This assumes
    # the TICL and NanoML files for a batch are event-aligned, same as every other
    # TICL/NanoML collection combined below.
    layercluster_match = batch_data.pop("LayerCluster_SimCluster_MatchIdx", None)
    if layercluster_match is not None and "clusters" in batch_data:
        batch_data["clusters"] = ak.with_field(
            batch_data["clusters"],
            layercluster_match["LayerCluster_SimCluster_MatchIdx"],
            where="LayerCluster_SimCluster_MatchIdx",
        )

    # Combine into single record array
    combined = ak.zip({name: arr for name, arr in batch_data.items()}, depth_limit=1)

    return combined

def main():
    args = parse_args()

    TICLfileList = args.TICLfiles
    MLfileList = args.nanoMLfiles
    
    if len(TICLfileList) != len(MLfileList):
        raise ValueError(f"Number of TICL files ({len(TICLfileList)}) != number of ML files ({len(MLfileList)})")
    
    # Create output directory
    outdir = pathlib.Path(args.outputDir)
    outdir.mkdir(exist_ok=True)
    output_path = outdir / args.outputFile
    
    # Process in batches and append to single parquet file
    batch_size = args.batch_size
    n_files = len(TICLfileList)
    n_batches = (n_files + batch_size - 1) // batch_size
    
    print(f"Processing {n_files} files in {n_batches} batches of up to {batch_size}")
    print(f"Collections to save: clusters, ticlCandidate, candidates, ticlTrackstersCLUE3DHigh, RecHitHGC, MergedSimCluster, SimCluster")
    print(f"Writing incrementally to {output_path}")
    
    parquet_writer = None
    
    for i in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_files)
        
        ticl_batch = TICLfileList[start_idx:end_idx]
        ml_batch = MLfileList[start_idx:end_idx]
        
        print(f"\nBatch {i+1}/{n_batches}: files {start_idx+1}-{end_idx}")
        batch_data = process_batch(ticl_batch, ml_batch)
        
        # Convert to arrow table
        table = ak.to_arrow_table(batch_data)
        
        # Write or append to parquet
        if parquet_writer is None:
            # First batch: create the writer with schema
            parquet_writer = pq.ParquetWriter(
                output_path, 
                table.schema,
                compression=args.compression
            )
        
        parquet_writer.write_table(table)
        print(f"  Appended batch {i+1} to parquet")
        
        # Free memory
        del batch_data, table
        gc.collect()
    
    # Close the writer
    if parquet_writer:
        parquet_writer.close()
    
    print(f"\nDone! Created single parquet file: {output_path}")

if __name__ == "__main__":
    main()
