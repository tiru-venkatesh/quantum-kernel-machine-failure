Quantum-kernel learning for machine-failure classification

Research question:
On a balanced case-control subset of the UCI AI4I 2020 synthetic predictive-maintenance dataset, how does a four-feature Qiskit ZZ-feature-map fidelity quantum-kernel SVM compare with classical Logistic Regression, RBF-SVM and Random Forest baselines?

The experiment uses four features:
1. Process temperature minus air temperature (K)
2. Rotational speed (rpm)
3. Torque (Nm)
4. Tool wear (min)

Quantum method:
- Four-qubit ZZ feature map with linear entanglement
- Qiskit FidelityQuantumKernel
- ComputeUncompute fidelity with StatevectorSampler
- Precomputed quantum kernel matrix + scikit-learn SVC

Run commands (after setup):
Quick smoke test:
python run_experiment.py --per-class 30 --shots 256 --reps 1 --output results_quick

Full experiment:
python run_experiment.py --per-class 120 --shots 1024 --reps 2 --output results

Dataset:
UCI AI4I 2020 Predictive Maintenance Dataset, ID 601, DOI 10.24432/C5HS5C.
UCI describes this benchmark as synthetic. Cite the source and disclose that it is not measured factory data.

Qiskit uses a local statevector sampler for simulation. This is not physical quantum-hardware execution and does not prove quantum advantage.
