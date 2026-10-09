from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
root=Path(__file__).resolve().parents[1]
out=root/'report'/'overleaf_plots'
fig,ax=plt.subplots(figsize=(7,4))
for suffix,label in [('', '10 training epochs'),('_20epochs','20 training epochs')]:
    data=np.genfromtxt(root/'report_images'/('dbn_label_convergence_weight_decay'+suffix)/'accuracy_by_step.csv',delimiter=',',names=True)
    ax.plot(data['gibbs_step'],100*data['accuracy_fraction'],label=label)
ax.set(xlabel='Recognition Gibbs step',ylabel='Test accuracy (%)',xlim=(1,30))
ax.grid(alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(out/'recognition_accuracy_comparison.png',dpi=300)
plt.close(fig)
fig,axes=plt.subplots(2,10,figsize=(14,3.5))
for digit in range(10):
    old=Image.open(root/'tmp'/'video_review_weight_decay'/f'digit{digit}.png')
    # Old review contains step 50 on the left and step 200 on the right.
    old=old.crop((old.width//2,0,old.width,old.height))
    new=Image.open(root/'tmp'/'video_review_20epochs'/f'digit{digit}.png')
    for row,im in enumerate([old,new]):
        axes[row,digit].imshow(im)
        axes[row,digit].set_xticks([])
        axes[row,digit].set_yticks([])
        if row==0: axes[row,digit].set_title(f'Label {digit}',fontsize=10)
axes[0,0].set_ylabel('10 epochs')
axes[1,0].set_ylabel('20 epochs')
fig.tight_layout()
fig.savefig(out/'generation_10_vs_20_epochs.png',dpi=250)
plt.close(fig)
# Original test images provide a reference, not inputs to generation.
from util import load_idxfile
images=load_idxfile(str(root/'t10k-images-idx3-ubyte'))
labels=load_idxfile(str(root/'t10k-labels-idx1-ubyte'))
fig,axes=plt.subplots(1,10,figsize=(14,2))
for digit,ax in enumerate(axes):
    ax.imshow(images[np.flatnonzero(labels==digit)[0]],cmap='gray',vmin=0,vmax=255)
    ax.set_title(f'Label {digit}',fontsize=10)
    ax.axis('off')
fig.tight_layout()
fig.savefig(out/'original_mnist_reference.png',dpi=250)
plt.close(fig)
(out/'README.txt').write_text('Upload this folder to Overleaf as overleaf_plots.\n\n'
'reconstruction_loss_comparison.png: Reconstruction losses across layers; independent 10/20-epoch runs, different y-axis scales.\n'
'recognition_accuracy_comparison.png: Test accuracy over 30 recognition Gibbs steps; 10,000 test images, one stochastic evaluation per model.\n'
'recognition_correct_label_convergence.png: Correct prediction example from the final 20-epoch model.\n'
'recognition_incorrect_label_convergence.png: Incorrect prediction example from the final 20-epoch model.\n'
'generation_10_vs_20_epochs.png: Final video frames at Gibbs step 200; requested labels above, 10 epochs top and 20 epochs bottom; independent samples. Original blue-white-red probability colormap retained.\n'
'original_mnist_reference.png: One original test image per class; visual reference, not images used to initialize generation. Grayscale differs from video colormap.\n'
'generation_label_influence_optional.png: Optional diagnostic: three chains per label after 200 Gibbs steps, identical starting states and random draws across labels within each row; final 20-epoch model, grayscale pixel probabilities.\n\n'
'Example: \\includegraphics[width=\\textwidth]{overleaf_plots/recognition_accuracy_comparison.png}\n',encoding='utf-8')
