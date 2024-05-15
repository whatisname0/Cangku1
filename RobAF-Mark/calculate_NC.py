import torch
import numpy as np

def computeNC(image1,image2):
    if isinstance(image1, torch.Tensor) and isinstance(image2, torch.Tensor):
        nc1 = (torch.sum((image1*image2)**2))/torch.sqrt(torch.sum(image2**2))/torch.sqrt(torch.sum(image1**2))
        image1 = 1-image1
        image2 = 1-image2
        nc2 = (torch.sum((image1*image2)**2))/torch.sqrt(torch.sum(image2**2))/torch.sqrt(torch.sum(image1**2))
        return (nc1+nc2)/2
    elif isinstance(image1, np.ndarray) and isinstance(image2, np.ndarray):
        nc1 = (np.sum((image1*image2)**2))/np.sqrt(np.sum(image2**2))/np.sqrt(np.sum(image1**2))
        image1 = 1-image1
        image2 = 1-image2
        nc2 = (np.sum((image1*image2)**2))/np.sqrt(np.sum(image2**2))/np.sqrt(np.sum(image1**2))
        return (nc1+nc2)/2
    else:
        raise TypeError("数组类型不是torch的tensor或numpy的array")
