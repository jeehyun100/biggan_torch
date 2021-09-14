
import numpy as np
import torch

from scipy.stats import truncnorm
from scipy.stats import norm
import matplotlib.pyplot as plt

class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
            print("done")
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


G_batch_size = 1
dim_z = 120


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda',
                fp16=False, z_var=1.0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

    if fp16:
        z_ = z_.half()

    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses)
    y_ = y_.to(device, torch.int64)
    return z_, y_


# z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
# z_var = 1.0
# z_.init_distribution('normal', mean=0, var=z_var)
# start, step, end = 0.05,0.05,1.0
#
# z_, y_ = prepare_z_y(G_batch_size, dim_z, 1000 )
# z_.sample_()
#
# list_arrage = np.arange(start, end + step, step)
# print(list_arrage)
# print('Getting truncation values for variance in range (%3.3f:%3.3f:%3.3f)...' % (start, step, end))
# for var in np.arange(start, end + step, step):
#     print(var)
#     z_.var = var
#     print("done")
#     # Optionally comment this out if you want to run with standing stats
#     # accumulated at one z variance setting


# user input
myclip_a = -0.5
myclip_b = 0.5
my_mean = 0.0
my_std = 0.3

a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
x_range = np.linspace(-2,2,1000)
plt.plot(x_range, truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std))
plt.plot(x_range, norm.pdf(x_range, my_mean, my_std))
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()