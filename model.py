import torch
import torch.nn as nn
import torch.nn.functional as nnF

"""
Dummy Model
- Generate complex mask
"""
class Net(nn.Module):
    def __init__(self) :
        super(Net,self).__init__()

        # Enc Modules
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4,kernel_size=(7,1),stride=(7,1)),
            nn.BatchNorm2d(4),
            nn.PReLU()
        )

        # Bottleneck Modules
        self.bn1 = nn.GRU(144, 144, batch_first=True)
        self.bn2 = nn.MultiheadAttention(144, 144, batch_first = True)

        # Dec Modules
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4,out_channels=2,kernel_size=(7,1),stride=(7,1),output_padding=(5,0)),
            nn.BatchNorm2d(2),
            nn.Tanh()
        )

    def forward(self,x,h=None) :
        B,C,F,T = x.shape
        z = self.enc(x)

        # z [B,C,F,T] -> [B,T,F*C]
        z = torch.permute(z,(0,3,1,2))
        z = torch.reshape(z,(B,T,-1))

        z,h1 = self.bn1(z,h)
        z,h2 = self.bn2(z,z,z)

        z = z.reshape(B, T, 4, 36)
        z = torch.permute(z,(0,2,3,1))

        z = self.dec(z)
        z = z * x
        return z, h1
"""
Wrapper Model for Dummy Model
"""
class NetHelper(nn.Module):
    def __init__(self):
        super(NetHelper,self).__init__()

        self.model = Net()

        self.window = torch.hann_window(512)

    """
        x[B, L] : wav input 
            - B : batch
            - L : length
    """
    def forward(self,x,state=None):
        """
        X [B, F, T, 2] : STFT
        """
        X = torch.stft(x, n_fft=512, hop_length=128, win_length=512, window=self.window.to(x.device),return_complex=False)

        # X [B,F,T,2] -> [B,2,F,T]
        X = torch.permute(X,(0,3,1,2))

        Y,h = self.model(X,state)
        Y = torch.permute(Y,(0,2,3,1))

        # torch.istft requires complex input
        y = torch.istft(Y[...,0] + Y[...,1]*1j, n_fft=512, hop_length=128, win_length=512, window=self.window.to(x.device))
        return y,h


if __name__ == "__main__" : 
    B = 64
    L = 16000

    # device option
    device = "cpu"
    device = "cuda:0"

    m = NetHelper().to(device)
    #m.load_state_dict(torch.load('dummy.pt'))

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(m.parameters(),lr=1e-2)

    for i in range(100) :
        x = torch.rand(B,L).to(device)
        y = m(x)[0]
        loss = criterion(y,x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

    m.eval()
    # Save Net not wrapper model
    torch.save(m.model.state_dict(), 'dummy.pt')
    m.to("cpu")

    torch.onnx.export(
            m.model,
            (torch.rand(1,2,257,125), torch.rand(1,1,144)),
            #(torch.rand(1,16000), torch.rand(1,1,144)),
            "dummy.onnx",
            verbose=False,
            opset_version=16,
            input_names=["inputs", "state_in"],
            output_names=["outputs", "state_out"],
    )