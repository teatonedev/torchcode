import torch

from torch import Tensor



def crossentropyloss(logits: Tensor, targets: Tensor) -> Tensor:
    batchSize = logits.shape[0]
    m = torch.max(logits, dim=1, keepdim=True).values

    lse = m + torch.log(torch.sum(torch.exp(logits - m), dim=1, keepdim=True))

    targetLogits = logits[range(batchSize), targets].unsqueeze(1)

    lossPerSample = lse - targetLogits

    return torch.mean(lossPerSample)



def main():
    logits = torch.tensor([[1.0, 2.0, 5.0], [0.5, 0.2, 0.1]])
    targets = torch.tensor([2,0])

    myLoss = crossentropyloss(logits,targets)
    referenceTorchLoss = torch.nn.functional.cross_entropy(logits,targets)

    print("Handmade implementation:", myLoss)
    print("Pytorch Reference:", referenceTorchLoss)


if __name__=="__main__":
    main()
