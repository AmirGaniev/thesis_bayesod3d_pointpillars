import torch


def sample_linear_params(mean, var, num_samples, eps=1e-4):
    """
    Sample linear parameters of boxes
    Args:
        mean [torch.Tensor(..., D)]: Bounding box parameter means
        var [torch.Tensor(..., D)]: Bounding box parameter variances
        num_samples [int]: Number of samples (M)
    Returns:
        samples [torch.Tensor(M, ..., D)]: Bounding box samples
    """
    std_dev = torch.sqrt(var + eps)
    dists = torch.distributions.normal.Normal(loc=mean, scale=std_dev)
    samples = dists.rsample((num_samples,))
    return samples


def sample_angular_params(mean, var, num_samples, eps=1e-4):
    """
    Sample angular parameters of boxes
    Args:
        mean [torch.Tensor(...)]: Bounding box angular means
        var [torch.Tensor(...)]: Bounding box angular variances
        num_samples [int]: Number of samples (M)
    Returns:
        samples [torch.Tensor(M, ...)]: Bounding box angle samples
    """
    conc = 1 / (var + eps)
    dists = torch.distributions.von_mises.VonMises(loc=mean, concentration=conc)
    samples = dists.sample((num_samples,))
    return samples


def sample_boxes(mean, var, num_samples, eps=1e-4, angular_dim=6):
    """
    Sample boxes from distributions
    Args:
        mean [torch.Tensor(..., D)]: Bounding box parameter means
        var [torch.Tensor(..., D)]: Bounding box parameter variances
        num_samples [int]: Number of samples (M)
    Returns:
        samples [torch.Tensor(M, ..., D)]: Bounding box samples
    """
    # Linear
    D = mean.shape[-1]
    linear_dims = torch.arange(D)
    linear_dims = linear_dims[linear_dims != angular_dim]
    linear_samples = sample_linear_params(mean=mean[..., linear_dims],
                                          var=var[..., linear_dims],
                                          num_samples=num_samples,
                                          eps=eps)
    # Angular
    angular_samples = sample_angular_params(mean=mean[..., angular_dim],
                                            var=var[..., angular_dim],
                                            num_samples=num_samples,
                                            eps=eps)

    # Combine samples
    samples_1 = linear_samples[..., :angular_dim]
    samples_2 = linear_samples[..., angular_dim:]
    angular_samples = angular_samples.unsqueeze(dim=-1)
    samples = torch.cat((samples_1, angular_samples, samples_2),
                        dim=-1)

    return samples

