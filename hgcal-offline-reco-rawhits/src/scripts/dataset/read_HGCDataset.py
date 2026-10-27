import os
import argparse
import yaml
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.dataset.dataset_maskformers import HGCDataset

def plot_incidence_matrix(incidence_matrix, outputfile="incidence_matrix.png"):
    if type(incidence_matrix).__name__ == "Tensor":
        print("Converting incidence matrix from PyTorch tensor to NumPy array for plotting.")
        incidence_matrix = incidence_matrix.cpu().numpy()
        print(f"Incidence matrix shape: {incidence_matrix.shape}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.imshow(incidence_matrix, cmap="viridis", aspect="auto", interpolation="none")
    plt.colorbar(label="Incidence")
    plt.xlabel("RecHit Index")
    plt.ylabel("MergedSimCluster Index")
    plt.title("Incidence Matrix")
    plt.ylim(-0.5, incidence_matrix.shape[0] - 0.5)
    plt.savefig(outputfile, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HGCDataset loading and incidence matrix plotting")
    parser.add_argument("input", type=str, help="Path to the input Parquet file")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the incidence matrix for each event")
    parser.add_argument("--entrystop", type=int, default=100, help="Number of events to process (default: 10)")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed information about inputs and labels")
    args = parser.parse_args()

    config_path = "/afs/cern.ch/work/m/mmarcheg/private/ml4reco/hgcal-offline-reco/configs/test_ticl_collections.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    inputs = config["data"]["inputs"]
    targets = config["data"]["targets"]
    scale_dict_path = config["data"]["scale_dict_path"]
    particle_scale_dict_path = config["data"]["particle_scale_dict_path"]
    incidence_cutval = float(config["data"]["incidence_cutval"])

    dataset = HGCDataset(
        args.input,
        inputs=inputs,
        targets=targets,
        scale_dict_path=scale_dict_path,
        particle_scale_dict_path=particle_scale_dict_path,
        topocluster_collection=config["data"]["topocluster_collection"],
        particle_collection=config["data"]["particle_collection"],
        num_events=args.entrystop,
        incidence_cutval=incidence_cutval,
        verbose=args.verbose
    )

    start_total = time.time()
    for idx in tqdm(range(min(len(dataset), args.entrystop)), desc="Loading events"):
        start = time.time()
        inputs, labels = dataset.__getitem__(idx)
        end = time.time()

        # Print inputs and label shapes for each key
        if args.verbose:
            print(f"Event {idx}:")
            for key in inputs:
                print(f"  Input '{key}': shape {inputs[key].shape}")
            for key in labels:
                print(f"  Label '{key}': shape {labels[key].shape}")
        
        if idx % max(1, args.entrystop // 10) == 0:
            tqdm.write(f"Event {idx} loaded in {end - start:.2f} seconds")
            if args.plot:
                plots_folder = "plots"
                os.makedirs(plots_folder, exist_ok=True)
                filename = os.path.join(plots_folder, f"incidence_matrix_event_{idx}.png")
                print(f"Saving incidence matrix plot for event {idx} to {filename}")
                plot_incidence_matrix(labels["particle_incidence"], outputfile=filename)
    end_total = time.time()
    print(f"Total time to load {args.entrystop} events: {end_total - start_total:.2f} seconds")
    print(f"Average time per event: {(end_total - start_total) / args.entrystop:.3f} seconds")
