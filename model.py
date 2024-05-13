import torch
import torch.nn as nn
import torch.nn.functional as nnF

class Net(nn.Module):
    def __init__(self) :
        super(Net,self).__init__()

        # Enc Modules
        self.enc1 = nn.Conv2d(in_channels=2, out_channels=4,kernel_size=1,groups=1)
        self.norm1 = nn.BatchNorm2d(4)
        self.act1 = nn.PReLU()
        self.enc2 = nn.Conv2d(in_channels=4, out_channels=4,kernel_size=(5,1),stride=(2,1),groups=4)
        self.norm2 = nn.BatchNorm2d(4)

        # Bottleneck Modules
        self.bn1 = nn.GRU(32, 32, batch_first=True)
        self.bn2 = nn.MultiheadAttention(32, 32, batch_first = True)

        # Dec Modules
        self.dec1 = nn.ConvTranspose2d(in_channels=4,out_channels=4,kernel_size=(5,1),stride=(2,1),groups=4)
        self.norm3 = nn.BatchNorm2d(4) 
        self.dec2 = nn.ConvTranspose2d(in_channels=4,out_channels=1,kernel_size=1,groups=1)

        # Activation Modules
        kernel = torch.eye(2)
        kernel = kernel.reshape(2,1, 2, 1)
        self.register_buffer('kernel', kernel)

    def forward(self,x,h=None) :
        B,C,F,T = x.shape
        z = self.enc1(x)
        z = self.norm1(z)
        z = self.act1(z)
        z = self.enc2(z)
        z = self.norm2(z)

        z = torch.permute(z,(0,3,1,2))
        z = torch.cat((z,z),dim=2)
        z = torch.stack((z,z),dim=2)
        z = z.reshape(B, T, 32)

        z,h1 = self.bn1(z,h)
        z,h2 = self.bn2(z,z,z)

        z = z.reshape(B, T, 4, 8)
        z = torch.permute(z,(0,2,3,1))

        z = self.dec1(z)
        z = self.norm3(z)
        z = self.dec2(z)

        z = nnF.conv2d(z,self.kernel)
        z = z.relu()

        return z, h1

if __name__ == "__main__" : 
    B = 1
    C = 2
    F = 7
    F2 = 18
    T = 13

    m = Net()
    m.load_state_dict(torch.load('dummy.pt'))
    x = torch.rand(B,C,F,T)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(m.parameters())

    z, h = m(x,None)
    z, h = m(x,h)

    """
    for i in range(10) :
        x = torch.rand(B,C,F,T)
        gt = torch.rand(B,C,F2,T)
        y = m(x)[0]
        loss = criterion(y,gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """

    m.eval()
    torch.save(m.state_dict(), 'dummy.pt')

    torch.onnx.export(
            m,
            (torch.rand(1,C,F,T), torch.rand(1,1,32)),
            "dummy.onnx",
            verbose=False,
            opset_version=16,
            input_names=["inputs", "state_in"],
            output_names=["outputs", "state_out"],
    )