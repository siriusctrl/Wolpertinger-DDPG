def soft_update(target, source, tau_update):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau_update) + param.data * tau_update
        )