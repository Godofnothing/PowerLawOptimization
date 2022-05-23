import torch


@torch.no_grad()
def train_full(n_steps, state, K, alpha_fn, beta_fn, batch_fn):
    N = len(K)
    loss_curve = torch.zeros(n_steps + 1)
    C, J, P = state['C'], state['J'], state['P'] 

    # add loss on 0th iteration
    loss = torch.trace(C).cpu() / (2 * N)
    loss_curve[0] = loss.item()
    
    for step in range(1, n_steps + 1):
        alpha, beta, batch_size = alpha_fn(step), beta_fn(step), batch_fn(step)
        # get gamma
        gamma = (N - batch_size) / (N - 1) / batch_size
        # get interaction term
        interaction_term = (gamma * alpha ** 2) * N * (K @ (torch.diag(C)[:, None] * K))
        # aux computations
        KC, KJ, KCK = K @ C, K @ J, K @ C @ K
        common_term = interaction_term - alpha * beta * (KJ + KJ.T) + \
            alpha ** 2 * (1 - gamma) * KCK + beta ** 2 * P
        # update elements
        C_new = C - alpha * (KC + KC.T) + beta * (J + J.T) + common_term
        J_new = beta * J - alpha * KC.T + common_term
        P_new = common_term
        C, J, P = C_new, J_new, P_new
        # update loss
        loss_curve[step] = torch.trace(C).cpu() / (2 * N)
        
    state['C'], state['J'], state['P'] = C, J, P
            
    return state, loss_curve
    