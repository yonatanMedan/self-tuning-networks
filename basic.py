import torch


# ### First Step Mapping Hyper Params to Unconstrained floats in R

# we want to map from parameters which are constrained to parameters with which can be any number from -inf to +inf, an example could be dropout, which is constrained from 0.0 to 1.0. we can map it with s_logit as described below

def logit(x):
    return torch.log(x) - torch.log(1-x)


def s_logit(x, min=0, max=1):
    """Stretched logit function: Maps x lying in (min, max) to R"""
    return logit((x - min)/(max-min))


from matplotlib import pyplot as plt

# in the plot we can see that values from 0 to 1 (which could be dropout values) are mapped to any value from 0.0 to 1.0

plt.plot(torch.arange(0,1,0.0001).numpy(),s_logit(torch.arange(0,1,0.0001)).numpy())


def get_unconstrained_R(hyper_paramater, min=0.,max=1):
    if min is not None and max is not None:
        return s_logit(torch.tensor(hyper_paramater), min=min, max=max)
    else:
        return logit(torch.tensor(hyper_paramater))


# ## Hypter parameter scale 

# ### might be the $\epsilon$ which we add to hyper parameters ???

def inv_softplus(x):
    """ Inverse softplus function: Maps x lying in (0, infty) to R"""
    return torch.log(torch.exp(x) - 1)


def get_scale(deviation):
    return inv_softplus(torch.tensor(deviation))


def map_hyper_params(param,scale,_min=0.,_max=1.):
    return get_unconstrained_R(param,_min,_max),get_scale(scale)
    


map_hyper_params(0.8,0.1,0.,0.9)

plt.plot(torch.arange(0,3,0.01).numpy(),get_scale(torch.arange(0,3,0.01)).numpy())


get_unconstrained_R(1.)

values which are un constrained could be 
