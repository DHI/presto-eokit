# Maintainer: Walid Ghariani
# Description: layerwise learning rate decay for better Fine-tuning


def get_layerwise_lr_decay(model, base_lr=1e-4, decay_factor=0.9, weight_decay=1e-4):
    param_groups = []
    encoder_layers = list(model.encoder.children())
    num_layers = len(encoder_layers)

    for i, layer in enumerate(encoder_layers):
        lr = base_lr * (decay_factor ** (num_layers - i - 1))
        param_groups.append(
            {"params": layer.parameters(), "lr": lr, "weight_decay": weight_decay}
        )

    param_groups.append(
        {"params": model.head.parameters(), "lr": base_lr, "weight_decay": weight_decay}
    )

    return param_groups
