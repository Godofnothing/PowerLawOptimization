import torch


def load_kernel_and_err(data_path: str, normalize_ntk: bool = True):
    data = torch.load(data_path)
    # save kernel
    K = data['NTK_train_train']
    # save err vec
    d_f = data['target_train_train']
    if normalize_ntk:
        # get kernel size
        N = K.shape[0]
        # renormalize NTK
        mult_factor = N / torch.linalg.eigvalsh(K).max()
        K *= mult_factor
    # normalize err
    d_f = (d_f - d_f.mean()) / d_f.std()
    return K, d_f 
