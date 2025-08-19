import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from itertools import product

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generate_data_A(n_samples):
    X = np.zeros((n_samples, 25, 25), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        coords = np.random.choice(25*25, 2, replace=False)
        (x1, y1), (x2, y2) = divmod(coords[0], 25), divmod(coords[1], 25)
        X[i, x1, y1] = 1.0
        X[i, x2, y2] = 1.0
        y[i] = np.hypot(x1 - x2, y1 - y2)
    return X.reshape(n_samples, 625), y

def generate_data_B(n_samples):
    X = np.zeros((n_samples, 25, 25), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        N = np.random.randint(3, 11)
        coords = np.random.choice(25*25, N, replace=False)
        X[i].flat[coords] = 1.0
        dmin = np.inf
        for a in range(N):
            x1, y1 = divmod(coords[a], 25)
            for b in range(a+1, N):
                x2, y2 = divmod(coords[b], 25)
                dmin = min(dmin, np.hypot(x1 - x2, y1 - y2))
        y[i] = dmin
    return X.reshape(n_samples, 625), y

def generate_data_C(n_samples):
    X = np.zeros((n_samples, 25, 25), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        N = np.random.randint(3, 11)
        coords = np.random.choice(25*25, N, replace=False)
        X[i].flat[coords] = 1.0
        dmax = 0.0
        for a in range(N):
            x1, y1 = divmod(coords[a], 25)
            for b in range(a+1, N):
                x2, y2 = divmod(coords[b], 25)
                dmax = max(dmax, np.hypot(x1 - x2, y1 - y2))
        y[i] = dmax
    return X.reshape(n_samples, 625), y

def generate_data_D(n_samples):
    X = np.zeros((n_samples, 25, 25), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        N = np.random.randint(1, 11)
        coords = np.random.choice(25*25, N, replace=False)
        X[i].flat[coords] = 1.0
        y[i] = float(N)
    return X.reshape(n_samples, 625), y

def generate_data_E(n_samples):
    X = np.zeros((n_samples, 25, 25), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        N = np.random.randint(1, 11)
        mat = np.zeros((25, 25), dtype=np.float32)
        for _ in range(N):
            s = np.random.randint(1, 7)
            x0 = np.random.randint(0, 26 - s)
            y0 = np.random.randint(0, 26 - s)
            mat[x0:x0 + s, y0:y0 + s] = 1.0
        X[i] = mat
        y[i] = float(N)
    return X.reshape(n_samples, 625), y

class SimpleMLP(nn.Module):
    def __init__(self, input_size=625, hidden_size=128, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def sanity_check():
    print("\n=== Sanity Checks ===")
    funcs = [
        (generate_data_A, 'A', 2),
        (generate_data_B, 'B', None),
        (generate_data_C, 'C', None),
        (generate_data_D, 'D', None),
        (generate_data_E, 'E', None)
    ]
    for fn, name, expected in funcs:
        X, y = fn(5)
        counts = [(X[i] == 1.0).sum() for i in range(5)]
        print(f"Problem{name}: counts={counts}, y={y}, Xminmax=({X.min():.1f},{X.max():.1f})")



def save_datasets(tasks):
    for fn, name in tasks:
        X_tr, y_tr = fn(800)
        X_te, y_te = fn(200)
        np.savetxt(f'{name}_train.txt', np.hstack([X_tr, y_tr.reshape(-1,1)]), fmt='%.4f', delimiter=',')
        np.savetxt(f'{name}_test.txt',  np.hstack([X_te, y_te.reshape(-1,1)]),  fmt='%.4f', delimiter=',')
        print(f"Saved {name}_train.txt and {name}_test.txt")



def hyperparameter_search(fn, name, param_grid, epochs=20):
    X_full, y_full = fn(800)
    perm = np.random.permutation(800)
    X_full, y_full = X_full[perm], y_full[perm]
    split = int(0.8 * len(X_full))
    X_tr, y_tr = X_full[:split], y_full[:split]
    X_val, y_val = X_full[split:], y_full[split:]
    best_mse = float('inf')
    best_params = {}
    for lr, hs, bs in product(param_grid['lr'], param_grid['hidden_size'], param_grid['batch_size']):
        tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1))
        tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=bs)
        model = SimpleMLP(hidden_size=hs).to('cpu')
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.MSELoss()
        for _ in range(epochs):
            model.train()
            for xb, yb in tr_ld:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        mses = []
        with torch.no_grad():
            for xb, yb in val_ld:
                mses.append(crit(model(xb), yb).item())
        mse = np.mean(mses)
        if mse < best_mse:
            best_mse = mse
            best_params = {'lr': lr, 'hidden_size': hs, 'batch_size': bs}
    print(f"[HP] Best for {name}: {best_params}, MSE={best_mse:.4f}")
    return best_params



def train_and_evaluate(fn, name,
                       epochs=100,
                       lr=1e-2,
                       hidden_size=128,
                       batch_size=32,
                       patience=5):
    print(f"\n===== Problem{name} =====")
    X_train, y_train = fn(800)
    X_test, y_test   = fn(200)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = {'25%': 200, '50%': 400, '100%': 800}
    criterion = nn.MSELoss()
    final_mae = {}
    final_mse = {}

    for split_name, n in splits.items():
        tr_ds = TensorDataset(torch.from_numpy(X_train[:n]), torch.from_numpy(y_train[:n]).unsqueeze(1))
        te_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).unsqueeze(1))
        tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=batch_size)

        model = SimpleMLP(hidden_size=hidden_size).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)

        best_mse = float('inf')
        wait = 0
        hist = []
        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in tr_ld:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                errs = []
                for xb, yb in te_ld:
                    preds = model(xb.to(device))
                    errs.append(((preds.cpu().numpy().flatten() - yb.numpy().flatten()) ** 2))
                epoch_mse = np.mean(np.concatenate(errs))
                hist.append(epoch_mse)

            if epoch_mse < best_mse:
                best_mse = epoch_mse
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch} for split {split_name}")
                    break


        plt.figure()
        plt.plot(hist, marker='o')
        plt.title(f"{name} [{split_name}] Test MSE per Epoch")
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.savefig(f'{name}_{split_name}_epoch_mse.png')
        plt.show()
        plt.close()


        with torch.no_grad():
            abs_e, sq_e = [], []
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb)
                abs_e.append(torch.abs(p - yb).cpu().numpy())
                sq_e.append(((p - yb) ** 2).cpu().numpy())
            final_mae[split_name] = float(np.mean(np.concatenate(abs_e)))
            final_mse[split_name] = float(np.mean(np.concatenate(sq_e)))

        print(f"{name} [{split_name}] MAE={final_mae[split_name]:.4f}, MSE={final_mse[split_name]:.4f}")


    keys = list(final_mae.keys())
    maes = [final_mae[k] for k in keys]
    mses = [final_mse[k] for k in keys]

    plt.figure()
    plt.plot(keys, maes, marker='o', label='MAE')
    plt.plot(keys, mses, marker='s', label='MSE')
    plt.title(f"{name} Test Error vs Training Set Size")
    plt.xlabel('Training Set Size')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{name}_error_vs_train.png')
    plt.show()
    plt.close()


    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            p = model(xb.to(device)).cpu().numpy().flatten()
            all_p.extend(p)
            all_t.extend(yb.numpy().flatten())
    all_p, all_t = np.array(all_p), np.array(all_t)
    errs = np.abs(all_p - all_t)
    print("\nSample\tTrue\tPred\tAbsErr\tRelErr")
    for i in range(10):
        print(f"{i}\t{all_t[i]:.2f}\t{all_p[i]:.2f}\t{errs[i]:.2f}\t{errs[i]/(all_t[i]+1e-8):.2f}")



if __name__ == '__main__':
    tasks = [
        (generate_data_A, 'A'),
        (generate_data_B, 'B'),
        (generate_data_C, 'C'),
        (generate_data_D, 'D'),
        (generate_data_E, 'E'),
    ]
    sanity_check()
    save_datasets(tasks)
    grid = {'lr': [1e-3, 1e-2], 'hidden_size': [64, 128], 'batch_size': [32, 64]}
    for fn, name in tasks:
        best = hyperparameter_search(fn, name, grid, epochs=20)
        train_and_evaluate(fn, name,
                           epochs=100,
                           lr=best['lr'],
                           hidden_size=best['hidden_size'],
                           batch_size=best['batch_size'],
                           patience=5)
