import torch



def relu(x : torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[out < 0] = 0
    return out




def main():
    x = torch.tensor([-2., -1., 0., -8., -17., 1., 2.])
    print(f"Input:{x}")
    print(f"Output:{relu(x)}")
    print(f"shape after relu: {relu(x).shape}")


if __name__ == "__main__":
    main()

