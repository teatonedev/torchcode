import torch


def mySoftmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    maxVal = torch.max(x, dim=dim, keepdim=True).values
    exponents = torch.exp(x - maxVal)

    sumOfExponents = torch.sum(exponents, dim=dim, keepdim=True)

    return (exponents / sumOfExponents)




def main():
    x = torch.tensor([1.0, 2.0, 3.0, 42.0])
    print("Output:", mySoftmax(x, dim=-1))
    print("Summation:", mySoftmax(x, dim=-1).sum())
    print("Reference:", torch.softmax(x, dim=-1))
    





if __name__=="__main__":
    main()
