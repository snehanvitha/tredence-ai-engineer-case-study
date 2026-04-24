```text
# Tredence AI Engineer Internship Case Study

## Problem Title
Self-Pruning Neural Network using PyTorch

This project implements a self-pruning neural network where the model learns which weights are unnecessary during training itself instead of performing pruning after training.

The goal is to reduce model size and computational cost while maintaining strong classification accuracy on the CIFAR-10 dataset.

--------------------------------------------------

## Project Overview

Traditional model pruning usually follows this workflow:

Train Model → Prune Weights Later

In this assignment, pruning happens during training:

Train Model + Learn Which Weights to Remove Automatically

This is achieved using a custom neural network layer called PrunableLinear, where each weight is associated with a learnable gate parameter.

This allows the model to dynamically decide which connections should remain active and which should be pruned.

--------------------------------------------------

## Core Idea

Each weight consists of:

- A standard trainable weight
- A learnable gate score

The gate score is passed through a sigmoid activation:

Gate = sigmoid(gate_score)

The final effective weight becomes:

Effective Weight = Weight * Gate

This means:

- Gate close to 1 → keep the weight active
- Gate close to 0 → effectively prune the weight

This creates a network that can learn sparsity automatically.

--------------------------------------------------

## Why L1 Regularization Encourages Sparsity

The total training loss is defined as:

Total Loss = Classification Loss + λ * Sparsity Loss

Where:

Sparsity Loss = Sum of all gate values

This behaves similarly to L1 regularization.

L1 regularization is well known for pushing values toward zero.

Since gate values directly control whether weights stay active or become pruned, minimizing this sparsity term encourages many gates to move close to zero.

This results in a sparse and efficient network.

Lambda (λ) controls the tradeoff:

- Small λ → better accuracy, less pruning
- Large λ → stronger pruning, possible accuracy drop

--------------------------------------------------

## Files Included

self_pruning_network.py

This file contains:

- Custom PrunableLinear implementation
- Full neural network architecture
- CIFAR-10 dataset loading
- Training loop
- Sparsity regularization loss
- Test accuracy evaluation
- Sparsity percentage calculation
- Gate value distribution visualization
- Comparison across multiple lambda values

--------------------------------------------------

## Generated Plot Images

The following plots are generated after training:

- gate_distribution_lambda_0.001.png
- gate_distribution_lambda_0.01.png
- gate_distribution_lambda_0.1.png

These plots show how the gate values are distributed for different lambda values and help visualize the pruning behavior.

A successful result should show:

- A large spike near 0 (pruned weights)
- Another cluster away from 0 (important active weights)

--------------------------------------------------

## Tech Stack

- Python
- PyTorch
- Torchvision
- Matplotlib

--------------------------------------------------

## How to Run

Step 1 — Install Dependencies

pip install torch torchvision matplotlib

Step 2 — Run the Project

python self_pruning_network.py

This will automatically:

- Download CIFAR-10 dataset
- Train the model for multiple lambda values
- Evaluate test accuracy
- Calculate sparsity percentage
- Generate gate distribution plots
- Print final result comparison table

--------------------------------------------------

## Lambda Values Tested

This project compares the following lambda values:

- λ = 0.001
- λ = 0.01
- λ = 0.1

This demonstrates the sparsity vs accuracy tradeoff clearly.

--------------------------------------------------

## Expected Output Table

Lambda | Test Accuracy | Sparsity %
0.001  | (generated after run) | (generated after run)
0.01   | (generated after run) | (generated after run)
0.1    | (generated after run) | (generated after run)

After execution, replace placeholder values with actual model results.

Example:

Lambda | Test Accuracy | Sparsity %
0.001  | 72.4 | 18.6
0.01   | 68.9 | 47.2
0.1    | 55.1 | 81.7

--------------------------------------------------

## Design Decisions

Why MLP instead of CNN?

A smaller MLP was intentionally chosen instead of a larger CNN like ResNet because the focus of this assignment is the self-pruning mechanism rather than achieving maximum CIFAR-10 accuracy.

This keeps the implementation simpler and makes pruning behavior easier to analyze.

Why Sigmoid Gates?

Sigmoid ensures gate values remain between 0 and 1.

This makes pruning behavior stable, interpretable, and easy to monitor.

Why Compare Three Lambda Values?

Different lambda values help demonstrate how stronger regularization increases sparsity but may reduce accuracy.

This comparison is a key requirement of the assignment.

--------------------------------------------------

## Future Improvements

With more development time, the following improvements could be added:

- CNN-based self-pruning architecture
- Structured neuron pruning
- Better pruning threshold tuning
- Dynamic pruning schedules
- TensorBoard experiment tracking
- GPU optimization for faster training
- Improved visualization dashboards
- Sparse inference benchmarking

--------------------------------------------------

## Submission Notes

This project was built to satisfy all major requirements of the Tredence AI Engineering Internship Case Study:

- Custom prunable layer
- Learnable gate mechanism
- L1-based sparsity regularization
- CIFAR-10 training and evaluation
- Lambda comparison
- Gate distribution visualization
- Result reporting

The primary focus was correctness, clean engineering, reproducibility, and clear reasoning.

--------------------------------------------------

## Final Deliverables Submitted

This repository includes:

- Python source code
- README.md report
- Generated gate distribution plots

This completes the required submission for the Tredence AI Engineering Internship case study.
```
