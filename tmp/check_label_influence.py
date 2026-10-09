import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
def load(name, param):
    return np.load(root / 'trained_rbm' / f'rbm.{name}.{param}.npy')
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))
w = load('pen+lbl--top', 'weight_vh')
bv = load('pen+lbl--top', 'bias_v')
bh = load('pen+lbl--top', 'bias_h')
wm = load('hid--pen', 'weight_vh')
bm = load('hid--pen', 'bias_v')
wb = load('vis--hid', 'weight_vh')
bb = load('vis--hid', 'bias_v')
for name, block in [('representation', w[:500]), ('labels', w[500:])]:
    print(name, 'range', block.min(), block.max(), 'std', block.std(), 'RMS', np.sqrt(np.mean(block**2)))
print('label biases', bv[500:])
print('pairwise label weight RMS difference', np.sqrt(np.mean((w[500:,None]-w[None,500:])**2)))
# Each label receives identical starting states and random draws for three chains.
# Differences therefore reflect label conditioning rather than unrelated noise.
rng = np.random.default_rng(123)
pen = np.tile((rng.random((3,500)) < .5).astype(float), (10,1))
labels = np.repeat(np.eye(10), 3, axis=0)
for step in range(200):
    ph = sigmoid(pen @ w[:500] + labels @ w[500:] + bh)
    h = (ph > np.tile(rng.random((3,2000)), (10,1))).astype(float)
    pv = sigmoid(h @ w[:500].T + bv[:500])
    pen = (pv > np.tile(rng.random((3,500)), (10,1))).astype(float)
    if step in (0,49,199):
        print('step', step+1, 'pen activity', pen.mean(), 'label variance of pen probabilities', pv.reshape(10,3,500).var(axis=0).mean())
hid_probs = sigmoid(pen @ wm.T + bm)
hid = (hid_probs > np.tile(rng.random((3,500)), (10,1))).astype(float)
pixels = sigmoid(hid @ wb.T + bb)
print('final pixel means per label', pixels.reshape(10,3,784).mean(axis=(1,2)))
print('pixel RMS change across label pairs', np.sqrt(np.mean((pixels.reshape(10,3,784)[:,None]-pixels.reshape(10,3,784)[None,:])**2)))
out = root / 'tmp' / 'label_influence'
out.mkdir(exist_ok=True)
canvas = Image.new('RGB', (10*112, 3*112+20), 'white')
draw = ImageDraw.Draw(canvas)
for digit in range(10):
    draw.text((digit*112+5,3),str(digit),fill='black')
    for chain in range(3):
        im=Image.fromarray((pixels[digit*3+chain].reshape(28,28)*255).astype('uint8')).resize((112,112))
        canvas.paste(im,(digit*112,20+chain*112))
canvas.save(out / 'paired_labels.png')
