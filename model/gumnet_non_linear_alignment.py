import torch
import torch.nn as nn
import torch.nn.functional as F


class GumNetNonLinearAlignment(nn.Module):
    """
    Non-Linear Transformation Regression and Spatial Warping Module for the GumNet architecture.

    The module predicts a sparse set of control-point displacements from the bidirectional
    correlation vectors, builds an image-conditioned adaptive regularization map (`l_map`),
    and solves a Thin-Plate-Spline (TPS) system to warp the source image. The TPS
    coefficients live in normalized `[-1, 1]^2` coordinates and can be re-applied to a
    source image at any resolution via `warp(...)`.

    Args:
        input_dim (int, optional): Total number of features after concatenating both
            bidirectional correlation vectors. Defaults to 8192.
        grid_size (int, optional): Side length of the control-point grid. Defaults to 4
            (16 control points arranged in a 4x4 grid inside `[-0.8, 0.8]^2`).
        lambda_scale (float, optional): Scalar multiplier applied to the normalized adaptive
            lambda map. Defaults to 0.1, matching the notebook formulation.

    Shape:
        - c_ab: `(B, D)` where `D` is the dimension of a single correlation vector (4096).
        - c_ba: `(B, D)` matching `c_ab`.
        - source_image: `(B, C, H, W)`.
        - warped_image: `(B, C, H, W)`.
        - target_pts: `(B, N, 2)` where `N = grid_size**2`.
        - l_map: `(B, 1, H, W)`.
        - displacement_field: `(B, 2, H, W)`.

    Examples:
        >>> module = GumNetNonLinearAlignment(input_dim=8192, grid_size=4)
        >>> c_ab = torch.randn(2, 4096)
        >>> c_ba = torch.randn(2, 4096)
        >>> img = torch.randn(2, 1, 192, 192)
        >>> warped, target_pts, l_map, disp = module(c_ab, c_ba, img)
        >>> warped.shape, target_pts.shape, l_map.shape, disp.shape
        (torch.Size([2, 1, 192, 192]), torch.Size([2, 16, 2]), torch.Size([2, 1, 192, 192]), torch.Size([2, 2, 192, 192]))
    """

    def __init__(self, input_dim=8192, grid_size=4, lambda_scale=0.1):
        super(GumNetNonLinearAlignment, self).__init__()

        self.grid_size = grid_size
        self.num_control_points = grid_size * grid_size
        self.lambda_scale = lambda_scale

        self.fc1 = nn.Linear(input_dim, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc_out = nn.Linear(2000, self.num_control_points * 2)

        # Fixed source control-point grid in normalized coords, kept inside [-0.8, 0.8]
        # to leave margin from the image boundary (TPS is ill-conditioned at corners).
        coords = torch.linspace(-0.8, 0.8, grid_size)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        source_pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (N, 2) as (x, y)
        self.register_buffer('source_pts', source_pts)

        self._initialize_identity()

    def _initialize_identity(self):
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _create_base_grid(self, B, H, W, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing='ij',
        )
        base_grid = torch.stack([x, y], dim=-1)              # (H, W, 2)
        return base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    def _compute_lambda_map(self, image):
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, :, :, 1:] - image[:, :, :, :-1]
        omega = torch.sqrt(
            F.pad(dx ** 2, (0, 0, 0, 1)) + F.pad(dy ** 2, (0, 1, 0, 0))
        ) + 1e-6
        # Collapse channels by mean so multi-channel inputs reduce to a single field.
        omega = omega.mean(dim=1, keepdim=True)
        l_map = 1.0 / (omega.pow(4) + 1e-7)
        mean = l_map.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return (l_map / mean) * self.lambda_scale

    @staticmethod
    def _tps_kernel(pts_a, pts_b):
        """U(r) = r^2 * log(r^2 + eps). pts_a: (..., M, 2), pts_b: (..., N, 2)."""
        diff = pts_a.unsqueeze(-2) - pts_b.unsqueeze(-3)
        r2 = (diff ** 2).sum(dim=-1)
        return r2 * torch.log(r2 + 1e-6)

    def _solve_tps(self, target_pts, l_map):
        """Solve the regularized TPS linear system. Returns coefficients of shape (B, N+3, 2)."""
        B, N, _ = target_pts.shape
        device, dtype = target_pts.device, target_pts.dtype
        source_pts = self.source_pts.to(device=device, dtype=dtype)

        K = self._tps_kernel(source_pts, source_pts)              # (N, N)
        K = K.unsqueeze(0).expand(B, -1, -1)                       # (B, N, N)
        P = torch.cat([torch.ones(N, 1, device=device, dtype=dtype), source_pts], dim=1)  # (N, 3)
        P = P.unsqueeze(0).expand(B, -1, -1)                       # (B, N, 3)

        # Sample λ at each source control point from the adaptive map.
        sample_grid = source_pts.view(1, 1, N, 2).expand(B, -1, -1, -1)
        lam = F.grid_sample(
            l_map, sample_grid, mode='bilinear', padding_mode='border', align_corners=True
        ).view(B, N)

        K_reg = K + torch.diag_embed(lam)                          # (B, N, N)

        zero_33 = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        top = torch.cat([K_reg, P], dim=2)                         # (B, N, N+3)
        bot = torch.cat([P.transpose(1, 2), zero_33], dim=2)       # (B, 3, N+3)
        L = torch.cat([top, bot], dim=1)                           # (B, N+3, N+3)

        zero_32 = torch.zeros(B, 3, 2, device=device, dtype=dtype)
        Y = torch.cat([target_pts, zero_32], dim=1)                # (B, N+3, 2)

        return torch.linalg.solve(L, Y)                            # (B, N+3, 2)

    def _evaluate_tps(self, W, H, W_size, device, dtype):
        """Evaluate the TPS function over a dense H x W_size grid. Returns (B, H, W_size, 2)."""
        B = W.shape[0]
        N = self.source_pts.shape[0]
        source_pts = self.source_pts.to(device=device, dtype=dtype)

        base_grid = self._create_base_grid(B, H, W_size, device, dtype)   # (B, H, W, 2)
        grid_flat = base_grid.reshape(B, H * W_size, 2)                   # (B, HW, 2)

        U = self._tps_kernel(grid_flat, source_pts.unsqueeze(0).expand(B, -1, -1))  # (B, HW, N)
        P_grid = torch.cat(
            [torch.ones(B, H * W_size, 1, device=device, dtype=dtype), grid_flat], dim=2
        )                                                                 # (B, HW, 3)

        warped = U @ W[:, :N] + P_grid @ W[:, N:]                         # (B, HW, 2)
        return warped.view(B, H, W_size, 2)

    def warp(self, image, target_pts, l_map):
        """Apply the TPS deformation defined by (target_pts, l_map) to an image at any resolution.

        The solve uses `l_map` at its native resolution (typically the network's input
        resolution); the warp is then evaluated densely at `image`'s spatial size. This
        keeps the deformation deterministic across resolutions — `image` can be the
        original 192x192 or the full-resolution source.
        """
        W = self._solve_tps(target_pts, l_map)
        _, _, H, Wd = image.shape
        warped_coords = self._evaluate_tps(W, H, Wd, image.device, image.dtype)
        return F.grid_sample(
            image, warped_coords, mode='bilinear', padding_mode='border', align_corners=True
        )

    def forward(self, c_ab, c_ba, source_image):
        """
        Predict an adaptive TPS deformation from the correlation vectors and warp the source image.

        Args:
            c_ab: Correlation features A→B, shape (B, 4096).
            c_ba: Correlation features B→A, shape (B, 4096).
            source_image: Moving image to be warped, shape (B, C, H, W).

        Returns:
            warped_image (B, C, H, W): Source image warped by the predicted TPS.
            target_pts (B, N, 2): Learned target control points in normalized coords.
            l_map (B, 1, H, W): Adaptive regularization field used by the solver.
            displacement_field (B, 2, H, W): Dense displacement (warped_coords - base_grid).
        """
        B, _, H, W = source_image.shape
        device, dtype = source_image.device, source_image.dtype

        c = torch.cat([c_ab, c_ba], dim=1)
        c = F.relu(self.fc1(c))
        c = F.relu(self.fc2(c))
        raw_shifts = self.fc_out(c)                                       # (B, N*2)

        delta_pts = raw_shifts.view(B, self.num_control_points, 2)        # (B, N, 2)
        target_pts = self.source_pts.to(device=device, dtype=dtype).unsqueeze(0) + delta_pts

        l_map = self._compute_lambda_map(source_image)                    # (B, 1, H, W)

        coeffs = self._solve_tps(target_pts, l_map)                       # (B, N+3, 2)
        warped_coords = self._evaluate_tps(coeffs, H, W, device, dtype)   # (B, H, W, 2)

        base_grid = self._create_base_grid(B, H, W, device, dtype)
        displacement_field = (warped_coords - base_grid).permute(0, 3, 1, 2).contiguous()

        warped_image = F.grid_sample(
            source_image, warped_coords,
            mode='bilinear', padding_mode='border', align_corners=True,
        )

        return warped_image, target_pts, l_map, displacement_field
