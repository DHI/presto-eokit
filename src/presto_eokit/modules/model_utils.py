# Maintainer: Walid Ghariani (wagh@dhigroup.com)
# Description: get_model_params function to get model parameters numbers(trainable and non-trainable) for each component of the model


def get_model_params(model, name="Model"):
    def format_count(n):
        return f"{n:,}"

    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    components = {
        name: model,
        f"{name}.encoder": getattr(model, "encoder", None),
        f"{name}.decoder": getattr(model, "decoder", None),
    }

    for comp_name, comp in components.items():
        if comp is not None:
            total, trainable = count_params(comp)
            print(
                f"{comp_name}: total={format_count(total)}, trainable={format_count(trainable)}"
            )
