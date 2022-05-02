import torch


@torch.no_grad()
def train_mean_field_SGD(n_steps, state, lambda_f, alpha_fn, beta_fn, batch_fn):
    N = len(lambda_f)
    loss_curve = torch.zeros(n_steps)
    C, J, P = state['C'], state['J'], state['P'] 
    
    for step in range(n_steps):
        alpha, beta, batch_size = alpha_fn(step), beta_fn(step), batch_fn(step)
        # get gamma
        gamma = (N - batch_size) / (N - 1) / batch_size
        # get interaction term
        interaction_term = gamma * (alpha * lambda_f) ** 2 * (C.sum() - C)
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
        
    state['C'], state['J'], state['P'] = C, J, P
            
    return state, loss_curve
