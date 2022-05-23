import torch


@torch.no_grad()
def train_spectral_diagonal(
    n_steps, 
    state, 
    lambda_f, 
    alpha_fn, 
    beta_fn, 
    batch_fn, 
    tau: float = 1.0,
    track_diag_err = False,
    track_freq = 1
):
    N = len(lambda_f)
    loss_curve = torch.zeros(n_steps + 1)
    C, J, P = state['C'], state['J'], state['P'] 
    # compute loss before training
    loss_curve[0] = C.mean().cpu() / 2
    if track_diag_err:
        Cs = torch.zeros((n_steps // track_freq, N))
    else:
        Cs = None
    
    for step in range(1, n_steps + 1):
        alpha, beta, batch_size = alpha_fn(step), beta_fn(step), batch_fn(step)
        # get gamma
        gamma = (N - batch_size) / (N - 1) / batch_size
        interaction_term = gamma * (alpha * lambda_f) ** 2 * (C.sum() - tau * C)
        # get common term
        common_term = (alpha * lambda_f) ** 2 * C - 2 * beta * alpha * lambda_f * J  \
            + beta ** 2 * P + interaction_term
        # update elements
        C_new = C - 2 * alpha * lambda_f * C + 2 * beta * J + common_term
        J_new = beta * J - alpha * lambda_f * C + common_term
        P_new = common_term
        C, J, P = C_new, J_new, P_new
        # update loss
        loss_curve[step] = C.mean().cpu() / 2
        if track_diag_err and step % track_freq == 0:
            Cs[step // track_freq] = torch.flip(C, dims=(0,)).cpu()
        
    state['C'], state['J'], state['P'] = C, J, P
    return state, loss_curve, Cs
