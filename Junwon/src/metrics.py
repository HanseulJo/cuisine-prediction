import torch

def top_k_accuracy(y_true, y_score, k=10):
    if not y_true.shape[0] == y_score.shape[0]:
        raise ValueError(f'y_true has shape {y_true.shape} and y_score has shape {y_score.shape}.')

    _, indices = torch.topk(y_score, k, dim=1)
    indices = indices.t()
    corrects = indices.eq(y_true.view(1, -1).expand_as(indices)).float().sum(0, keepdim=True)
    acc = corrects.mean().item()
    print('acc', acc)

    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    # res.append(correct_k.mul_(100.0 / batch_size))
    # https://forums.fast.ai/t/return-top-k-accuracy/27658/2
    return acc