"""Reference outputs from Noise2Mic (the BoxNet corpus snapshot) for Noise2MicParityCudaTests.

Usage, with the noise2mic sources on PYTHONPATH (their src/ package):
  python parity.py network OUT                      # seeded R2UNet: weights, input, output
  python parity.py check-saved MODEL INPUT OUT      # run a checkpoint saved by Warp on INPUT
  python parity.py preprocess ODD EVEN XML OUT      # MicrographPreprocessor on one corpus pair
  python parity.py infer MODEL ODD EVEN XML OUT     # paired inference exactly as noise2mic's infer loop
  python parity.py train LIST EPOCHS OUT [SEED]     # noise2mic's training recipe on a pair list (odd even xml per line)
  python parity.py compare ODD EVEN XML MODEL...    # pairwise correlations of the models' paired outputs
  python parity.py export-weights CHECKPOINT OUT    # a Noise2Mic checkpoint as weights Warp loads (plain dict + recipe)
Arrays are raw little-endian float32 files with a .json giving their shape.
"""
import json, math, random, sys
import numpy as np, torch, mrcfile
from src.model import R2UNet
from src.utils.preprocessing import MicrographPreprocessor
from src.utils.cryo_utils import parse_warp_xml_for_ctf

PATCH, FEATURES, DEPTH, T = 512, 32, 4, 3
STRENGTH, FALLOFF, HIGHPASS = 0.9, 0.1, 600.0


def save(path, array):
    array = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    array.tofile(path + '.bin')
    json.dump({'shape': list(array.shape)}, open(path + '.json', 'w'))


def load(path):
    shape = json.load(open(path + '.json'))['shape']
    return np.fromfile(path + '.bin', dtype=np.float32).reshape(shape)


def model(device):
    return R2UNet(patch_size=PATCH, in_channels=1, out_channels=1, features_base=FEATURES,
                  num_res_blocks_per_stage=DEPTH, t=T).to(device).eval()


def strict():
    # Compare against full fp32 on both sides.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def read(path):
    with mrcfile.open(path, permissive=True) as f:
        return f.data.astype(np.float32), float(f.voxel_size.x)


def preprocess(odd_path, even_path, xml):
    odd, pixel = read(odd_path)
    even, _ = read(even_path)
    ctf = parse_warp_xml_for_ctf(xml)
    pre = MicrographPreprocessor(patch_size=PATCH, deconv_strength=STRENGTH, deconv_falloff=FALLOFF, deconv_highpass_a=HIGHPASS)
    odd_n, even_n, dims = pre.preprocess_micrograph_pair(odd, even, ctf, pixel)
    return odd_n.squeeze(0).numpy(), even_n.squeeze(0).numpy(), dims


def tiled(net, image, device, overlap=64, batch=16):
    """noise2mic/src/infer.py's loop: Hann-blended 512 tiles."""
    img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    _, _, h, w = img.shape
    stride = PATCH - overlap
    nh = math.ceil(max(1, (h - PATCH)) / stride) + 1
    nw = math.ceil(max(1, (w - PATCH)) / stride) + 1
    hann = (torch.hann_window(PATCH, periodic=False, device=device).unsqueeze(1) *
            torch.hann_window(PATCH, periodic=False, device=device).unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    out, count, patches, coords = torch.zeros_like(img), torch.zeros_like(img), [], []
    for i in range(nh):
        for j in range(nw):
            y, x = min(i * stride, h - PATCH), min(j * stride, w - PATCH)
            patches.append(img[:, :, y:y + PATCH, x:x + PATCH]); coords.append((y, x))
    with torch.no_grad():
        for i in range(0, len(patches), batch):
            result = net(torch.cat(patches[i:i + batch], 0))
            for k, (y, x) in enumerate(coords[i:i + batch]):
                out[:, :, y:y + PATCH, x:x + PATCH] += result[k:k + 1] * hann
                count[:, :, y:y + PATCH, x:x + PATCH] += hann
    return (out / torch.clamp(count, min=1e-6)).squeeze().cpu().numpy()


def crop(image, h, w):
    y, x = (image.shape[0] - h) // 2, (image.shape[1] - w) // 2
    return image[y:y + h, x:x + w]


def main():
    device = torch.device('cuda')
    command = sys.argv[1]
    if command == 'network':
        strict()
        torch.manual_seed(1234)
        net = model(device)
        x = torch.randn(2, 1, PATCH, PATCH, device=device)
        with torch.no_grad():
            y = net(x)
        out = sys.argv[2]
        torch.save({'model_state_dict': dict(net.state_dict()), 'patch_size': PATCH}, out + '-model.pt')
        save(out + '-input', x.cpu().numpy()); save(out + '-output', y.cpu().numpy())
    elif command == 'check-saved':
        strict()
        # libtorch tags typed dicts with torch.jit._pickle.restore_type_tag, which weights_only loading refuses.
        checkpoint = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
        net = model(device)
        net.load_state_dict(checkpoint['model_state_dict'])
        with torch.no_grad():
            y = net(torch.from_numpy(load(sys.argv[3])).to(device))
        save(sys.argv[4], y.cpu().numpy())
        print(json.dumps({k: v for k, v in checkpoint.items() if k != 'model_state_dict'}))
    elif command == 'preprocess':
        odd, even, dims = preprocess(*sys.argv[2:5])
        out = sys.argv[5]
        save(out + '-odd', odd); save(out + '-even', even)
        json.dump({'height': dims[0], 'width': dims[1]}, open(out + '-dims.json', 'w'))
    elif command == 'infer':
        checkpoint = torch.load(sys.argv[2], map_location='cpu')
        net = model(device)
        net.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state_dict'].items()})
        odd, even, dims = preprocess(*sys.argv[3:6])
        save(sys.argv[6], crop(tiled(net, (odd + even) / 2, device), *dims))
    elif command == 'train':
        train(sys.argv[2], int(sys.argv[3]), sys.argv[4], device, seed=int(sys.argv[5]) if len(sys.argv) > 5 else 7)
    elif command == 'export-weights':
        # libtorch's unpickler cannot read the OrderedDict (with _metadata) that state_dict() returns.
        checkpoint = torch.load(sys.argv[2], map_location='cpu')
        state = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state_dict'].items()}
        keep = ['patch_size', 'model_features', 'model_depth', 't', 'deconv_strength', 'deconv_falloff', 'deconv_highpass_a']
        torch.save({'model_state_dict': state, **{k: checkpoint[k] for k in keep if k in checkpoint}}, sys.argv[3])
    elif command == 'compare':
        odd, even, dims = preprocess(*sys.argv[2:5])
        outputs = []
        for path in sys.argv[5:]:
            net = model(device)
            net.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'].items()})
            outputs.append(crop(tiled(net, (odd + even) / 2, device), *dims).ravel())
        noisy = crop((odd + even) / 2, *dims).ravel()
        names = [p.split('/')[-1] for p in sys.argv[5:]]
        for i, a in enumerate(outputs):
            row = ' '.join(f'{np.corrcoef(a, b)[0, 1]:.4f}' for b in outputs)
            print(f'{names[i]:>28}: {row}   vs noisy input {np.corrcoef(a, noisy)[0, 1]:.4f}')


def train(listing, epochs, out, device, batch_size=12, per_image=4, warmup=50, lr_start=1e-4, lr_finish=1e-6, seed=7, micro=4):
    """noise2mic/src/train.py and dataset.py: random augmented patches, odd/even swap, Adam + warm-up + ExponentialLR, AMP."""
    random.seed(seed); torch.manual_seed(seed)
    pairs = [line.split() for line in open(listing) if line.strip()]
    prepared = [preprocess(*p)[:2] for p in pairs]
    net = model(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_start)
    gamma = (lr_finish / lr_start) ** (1 / (epochs - 1)) if epochs > 1 else 1.0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        order = list(range(len(prepared) * per_image)); random.shuffle(order)
        for i in range(0, len(order), batch_size):
            if epoch == 0 and i // batch_size < warmup:
                for group in optimizer.param_groups:
                    group['lr'] = lr_start * (i // batch_size + 1) / warmup
            odds, evens = [], []
            for index in order[i:i + batch_size]:
                odd, even = prepared[index // per_image]
                y0, x0 = random.randint(0, odd.shape[0] - PATCH), random.randint(0, odd.shape[1] - PATCH)
                pair = torch.from_numpy(np.stack([odd[y0:y0 + PATCH, x0:x0 + PATCH], even[y0:y0 + PATCH, x0:x0 + PATCH]]))
                pair = torch.rot90(pair, random.choice([0, 1, 2, 3]), dims=(-2, -1))
                if random.random() < 0.5: pair = torch.flip(pair, dims=[-1])
                if random.random() < 0.5: pair = torch.flip(pair, dims=[-2])
                odds.append(pair[0:1]); evens.append(pair[1:2])
            source, target = torch.stack(odds).to(device), torch.stack(evens).to(device)
            if random.random() < 0.5: source, target = target, source
            optimizer.zero_grad()
            # Batch 12 does not fit a 44 GB GPU; GroupNorm is per sample, so accumulating micro-batches is the same step.
            total = 0.0
            for m in range(0, len(source), micro):
                with torch.autocast('cuda', dtype=torch.float16):
                    output = net(source[m:m + micro].half())
                loss = criterion(output.float(), target[m:m + micro]) * (len(source[m:m + micro]) / len(source))
                scaler.scale(loss).backward(); total += loss.item()
            scaler.step(optimizer); scaler.update()
            losses.append(total)
        scheduler.step()
        print(f'epoch {epoch + 1}: mean loss {np.mean(losses[-math.ceil(len(order) / batch_size):]):.5f}', flush=True)
    torch.save({'model_state_dict': dict(net.state_dict()), 'patch_size': PATCH, 'model_features': FEATURES, 'model_depth': DEPTH, 't': T,
                'deconv_strength': STRENGTH, 'deconv_falloff': FALLOFF, 'deconv_highpass_a': HIGHPASS}, out)
    json.dump(losses, open(out + '.losses.json', 'w'))


if __name__ == '__main__':
    main()
