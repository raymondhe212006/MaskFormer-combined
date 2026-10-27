import awkward as ak
import vector
import fastjet

vector.register_awkward()

deltaR = 0.4

jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, deltaR)

array = ak.Array(
    [
        {"px": 1.2, "py": 3.2, "pz": 5.4, "E": 23.5},
        {"px": 32.2, "py": 64.21, "pz": 543.34, "E": 755.12},
        {"px": 32.45, "py": 63.21, "pz": 543.14, "E": 835.56},
    ],
    with_name="Momentum4D",
)

print("Input array:")
print(array.tolist())

print("Performing jet clustering...")
cluster = fastjet.ClusterSequence(array, jetdef)
print("Cluster:")
print(cluster)

print("Resulting jets:")
print(cluster.inclusive_jets().tolist())

