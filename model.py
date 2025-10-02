import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# UNet Backbone
# -----------------------------
class UNetBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU()
        )
        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.LeakyReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.LeakyReLU()
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.LeakyReLU()
        )

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU()
        )

        # Output feature map
        self.out_conv = nn.Conv2d(base_channels, base_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)
        out = torch.cat([x, out], dim=1)  # C = 3 + C'
        return out


# -----------------------------
# FilterParamPredictor
# -----------------------------
class FilterParamPredictor(nn.Module):
    def __init__(self, in_channels, out_params, dropout=0.5):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels*4, out_params),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


# -----------------------------
# Cubic Polynomial Filter
# -----------------------------
class CubicPolynomialFilter(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.num_channels = num_channels
        self.num_params = 10  # cubic-10
        self.params = nn.Parameter(torch.zeros(num_channels, self.num_params))

        with torch.no_grad():
            self.params[:, -1] = 1.0
            self.params[:, :-1] = torch.randn(num_channels, self.num_params-1) * 0.1

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        y_coords = y_coords.float().unsqueeze(0).unsqueeze(0)
        x_coords = x_coords.float().unsqueeze(0).unsqueeze(0)

        out = torch.zeros_like(x)
        for ch in range(C):
            p = self.params[ch]
            i = x[:, ch:ch+1, :, :]
            A, Bc, C_, D, E, F, G, H_, I_, J = p
            poly = i * (A*x_coords**3 + Bc*x_coords**2*y_coords + C_*x_coords**2 + D +
                        E*x_coords*y_coords + F*x_coords + G*y_coords**3 + H_*y_coords**2 +
                        I_*y_coords + J)
            out[:, ch:ch+1, :, :] = poly
        return out


# -----------------------------
# Elliptical Filter
# -----------------------------
class EllipticalFilter(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        self.num_channels = num_channels
        self.params = None

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        if self.params is None or self.num_channels != C:
            self.num_channels = C
            self.params = nn.Parameter(torch.zeros(C, 6, device=device))
            with torch.no_grad():
                self.params[:, 0] = H // 2
                self.params[:, 1] = W // 2
                self.params[:, 2] = H // 4
                self.params[:, 3] = W // 4
                self.params[:, 4] = 0.0
                self.params[:, 5] = 1.0

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        y_coords = y_coords.float().unsqueeze(0).unsqueeze(0)
        x_coords = x_coords.float().unsqueeze(0).unsqueeze(0)

        out = torch.zeros_like(x)
        for ch in range(C):
            h, k, a, b, theta, se = self.params[ch]
            a = torch.clamp(torch.abs(a), min=1.0)
            b = torch.clamp(torch.abs(b), min=1.0)
            se = torch.clamp(se, 0.0, 2.0)

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            x_rot = (x_coords - h) * cos_theta + (y_coords - k) * sin_theta
            y_rot = (x_coords - h) * sin_theta - (y_coords - k) * cos_theta
            dist = (x_rot**2 / (a**2)) + (y_rot**2 / (b**2))
            scale = se * torch.clamp(1 - dist, min=0.0, max=1.0)

            out[:, ch:ch+1, :, :] = x[:, ch:ch+1, :, :] * (scale + 0.1)
        return torch.clamp(out, 0.0, 1.0)


# -----------------------------
# Graduated Filter
# -----------------------------
class GraduatedFilter(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        self.num_channels = num_channels
        self.params = None

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        if self.params is None or self.num_channels != C:
            self.num_channels = C
            self.params = nn.Parameter(torch.zeros(C, 6, device=device))
            with torch.no_grad():
                self.params[:, 0] = 1.0
                self.params[:, 1] = H // 2
                self.params[:, 2] = H // 4
                self.params[:, 3] = 3*H//4
                self.params[:, 4] = 1.0
                self.params[:, 5] = 0.0

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        y_coords = y_coords.float().unsqueeze(0).unsqueeze(0)
        x_coords = x_coords.float().unsqueeze(0).unsqueeze(0)

        out = torch.zeros_like(x)
        for ch in range(C):
            m, c, o1, o2, sg, ginv_real = self.params[ch]
            m = torch.clamp(m, -10, 10)
            o1 = torch.clamp(torch.abs(o1), min=1.0)
            o2 = torch.clamp(torch.abs(o2), min=1.0)
            sg = torch.clamp(sg, 0.0, 2.0)

            ginv = torch.sigmoid(ginv_real)
            alpha = torch.atan(m)
            l = y_coords - (m * x_coords + c)
            d1 = o1 * torch.abs(torch.cos(alpha))
            d2 = o2 * torch.abs(torch.cos(alpha))

            a = sg * torch.clamp(0.5 * (1 + l/d2), 0.0, 1.0)
            b = sg * torch.clamp(0.5 * (1 + l/d1), 0.0, 1.0)
            s = torch.where(l >= 0, ginv*a + (1-ginv)*b, (1-ginv)*a + ginv*b)

            s = torch.clamp(s + 0.1, 0.0, 1.0)
            out[:, ch:ch+1, :, :] = x[:, ch:ch+1, :, :] * s
        return torch.clamp(out, 0.0, 1.0)


# -----------------------------
# DeepLPF Main Model
# -----------------------------
class DeepLPF(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.backbone = UNetBackbone(in_channels=in_channels, base_channels=base_channels)
        self.polynomial_filter = CubicPolynomialFilter(num_channels=in_channels)
        self.graduated_filter = GraduatedFilter()
        self.elliptical_filter = EllipticalFilter()

    def forward(self, x):
        feature_map = self.backbone(x)
        Y1 = feature_map[:, :3, :, :]
        Y2 = self.polynomial_filter(Y1)

        C_prime_features = feature_map[:, 3:, :, :]
        two_stream_input = torch.cat([Y2, C_prime_features], dim=1) if C_prime_features.shape[1] > 0 else Y2

        S_elliptical = self.elliptical_filter(two_stream_input)
        S_graduated = self.graduated_filter(two_stream_input)
        S = S_elliptical + S_graduated

        Y3 = Y2 * S[:, :3, :, :]
        Y_hat = Y3 + Y1
        return Y_hat
