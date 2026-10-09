Upload this folder to Overleaf as overleaf_plots.

reconstruction_loss_comparison.png: Reconstruction losses across layers; independent 10/20-epoch runs, different y-axis scales.
recognition_accuracy_comparison.png: Test accuracy over 30 recognition Gibbs steps; 10,000 test images, one stochastic evaluation per model.
recognition_correct_label_convergence.png: Correct prediction example from the final 20-epoch model.
recognition_incorrect_label_convergence.png: Incorrect prediction example from the final 20-epoch model.
generation_10_vs_20_epochs.png: Final video frames at Gibbs step 200; requested labels above, 10 epochs top and 20 epochs bottom; independent samples. Original blue-white-red probability colormap retained.
original_mnist_reference.png: One original test image per class; visual reference, not images used to initialize generation. Grayscale differs from video colormap.
generation_label_influence_optional.png: Optional diagnostic: three chains per label after 200 Gibbs steps, identical starting states and random draws across labels within each row; final 20-epoch model, grayscale pixel probabilities.

Example: \includegraphics[width=\textwidth]{overleaf_plots/recognition_accuracy_comparison.png}
