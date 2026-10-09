from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / 'report_images' / 'dbn_epoch_comparison'
out.mkdir(parents=True, exist_ok=True)
layers = [('vis_hid', 'Bottom RBM: 784–500'), ('hid_pen', 'Middle RBM: 500–500'), ('pen_lbl_top', 'Top RBM: (500+10)–2000')]
rows = []
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (layer, title) in zip(axes, layers):
    result = []
    for epochs, suffix, color in [(10, '', '#0072B2'), (20, '_20epochs', '#D55E00')]:
        folder = root / 'report_images' / ('dbn_pretraining_weight_decay' + suffix) / layer
        history = np.genfromtxt(folder / 'epoch_metrics.csv', delimiter=',', names=True)
        assert len(history) == epochs
        summary = dict(line.split('=', 1) for line in (folder / 'training_summary.txt').read_text().splitlines())
        loss = float(history['training_reconstruction_mse'][-1])
        seconds = float(summary['training_seconds'])
        result.extend([loss, seconds])
        ax.plot(history['epoch'], history['training_reconstruction_mse'], label=f'{epochs} epochs', color=color, marker='o', markersize=3, linewidth=1.5)
    rows.append([layer, *result])
    ax.set(title=title, xlabel='Epoch', ylabel='Training reconstruction MSE', xlim=(1,20))
    ax.set_xticks([1,5,10,15,20])
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / 'reconstruction_loss_comparison.png', dpi=300)
fig.savefig(out / 'reconstruction_loss_comparison.pdf')
plt.close(fig)
with (out / 'reconstruction_comparison.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['rbm','mse_10_epochs','seconds_10_epochs','mse_20_epochs','seconds_20_epochs'])
    writer.writerows(rows)
table_rows = '\n'.join(f"        {name} & {r[1]:.5f} & {r[2]:.1f} & {r[3]:.5f} & {r[4]:.1f} \\\\" for name,r in zip(['Bottom','Middle','Top'],rows))
text = r'''\subsection{Reconstruction losses across layers}

\begin{table}[htbp]
    \centering
    \caption{Final epoch training reconstruction MSE and training time
    for the weight-decay experiments.}
    \label{tab:dbn-layer-losses}
    \begin{tabular}{lrrrr}
        \toprule
        & \multicolumn{2}{c}{10 epochs} & \multicolumn{2}{c}{20 epochs} \\
        RBM & MSE & Time (s) & MSE & Time (s) \\
        \midrule
''' + table_rows + r'''
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{dbn_epoch_comparison/reconstruction_loss_comparison.pdf}
    \caption{Training reconstruction MSE per epoch for each RBM,
    comparing separate 10- and 20-epoch training runs. The panels
    use different vertical scales.}
    \label{fig:dbn-layer-losses}
\end{figure}

Table~\ref{tab:dbn-layer-losses} and Figure~\ref{fig:dbn-layer-losses}
show lower final reconstruction MSE for all three RBMs in the
20-epoch experiment, with smaller improvements in later epochs.
The values are averages over minibatches measured during training,
rather than held-out evaluations of the final models.

Losses should be compared within each layer because their inputs
differ: the bottom RBM reconstructs pixels, the middle RBM
reconstructs learned feature probabilities, and the top RBM
reconstructs these higher-level representations together with labels.
The experiments used separate random initializations and stochastic
sampling, so their differences cannot be attributed solely to
the number of epochs. Training times also depend on computer load.
'''
(root / 'report' / 'dbn_reconstruction_losses_overleaf.txt').write_text(text, encoding='utf-8')
print('\n'.join(str(row) for row in rows))
