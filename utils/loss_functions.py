import torch


def soft_dice_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))


def soft_dice_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def soft_dice_loss_multi_class_debug(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    loss_components = 1 - 2 * intersection/denom
    return loss, loss_components


def generalized_soft_dice_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-12

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width
    ysum = y.sum(dim=sum_dims)
    wc = 1 / (ysum ** 2 + eps)
    intersection = ((y * p).sum(dim=sum_dims) * wc).sum()
    denom =  ((ysum + p.sum(dim=sum_dims)) * wc).sum()

    loss = 1 - (2. * intersection / denom)
    return loss


def jaccard_like_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y ** 2 + p ** 2).sum(dim=sum_dims) + (y*p).sum(dim=sum_dims) + eps

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def jaccard_like_loss(input:torch.Tensor, target:torch.Tensor, device: str = 'cuda'):

    sm = torch.nn.Softmax(dim=1).to(device)
    input_prob = sm(input)
    input_prob = input_prob[:, 1, ]
    # input_pred = torch.argmax(input_prob, dim=1, keepdim=False)

    eps = 1e-6

    iflat = input_prob.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps

    return 1 - ((2. * intersection) / denom)


def jaccard_like_balanced_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps
    piccard = (2. * intersection)/denom

    n_iflat = 1-iflat
    n_tflat = 1-tflat
    neg_intersection = (n_iflat * n_tflat).sum()
    neg_denom = (n_iflat**2 + n_tflat**2).sum() - (n_iflat * n_tflat).sum()
    n_piccard = (2. * neg_intersection)/neg_denom

    return 1 - piccard - n_piccard


def soft_dice_loss_balanced(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    dice_pos = ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

    negatiev_intersection = ((1-iflat) * (1 - tflat)).sum()
    dice_neg =  (2 * negatiev_intersection) / ((1-iflat).sum() + (1-tflat).sum() + eps)

    return 1 - dice_pos - dice_negimport
    torch


# define "soft" cross-entropy with pytorch tensor operations
def soft_cross_entropy_loss_1d(logits: torch.Tensor, target: torch.Tensor):
    # logits shape B, C, H, W
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    multiplication = target * log_probs
    sum_ = multiplication.sum()
    n_samples = logits.shape[0]
    output = -sum_ / n_samples
    # return -(target * log_probs).sum() / input.shape[0]
    return output


# define "soft" cross-entropy with pytorch tensor operations
def soft_cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor):
    # logits shape B, C, H, W
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    multiplication = target * log_probs
    sum_ = multiplication.sum()
    n_samples = logits.shape[-2] * logits.shape[-1] * logits.shape[0]
    output = -sum_ / n_samples
    return output


# define rmse loss with pytorch tensor operations
def root_mean_square_error_loss(logits: torch.Tensor, target: torch.Tensor):
    # logits shape B, C, H, W
    probs = torch.nn.functional.softmax(logits, dim=1)
    nominator = torch.sum(torch.pow(torch.sub(probs, target), 2))
    denominator = target.numel()
    output = torch.sqrt(nominator / denominator)
    return output

if __name__ == '__main__':

    torch.manual_seed(2020)

    C_dim = 1

    # input values are logits
    input = torch.autograd.Variable(torch.randn((2, 3)))
    # target values are "soft" probabilities that sum to one (for each sample in batch)
    target = torch.nn.functional.softmax(torch.autograd.Variable(torch.randn((2, 3))), dim=C_dim)

    loss = soft_cross_entropy_loss_1d(input, target)

    # make "hard" categorical target
    target_cat = target.argmax(dim=C_dim)
    # make "hard" one-hot target
    target_onehot = torch.zeros_like(target).scatter(1, target_cat.unsqueeze(1), 1)


    # check that soft_cross_entorpy agrees with pytorch's cross_entropy for "hard" case
    reference_loss = torch.nn.functional.cross_entropy(input, target_cat)
    print(reference_loss)
    soft_loss = soft_cross_entropy_loss_1d(input, target_onehot)
    print(soft_loss)



