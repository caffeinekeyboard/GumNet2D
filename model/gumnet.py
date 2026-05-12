import torch.nn as nn
from .gumnet_feature_extraction import GumNetFeatureExtraction
from .gumnet_siamese_matching import GumNetSiameseMatching
from .gumnet_non_linear_alignment import GumNetNonLinearAlignment


class GumNet(nn.Module):
    """
    Complete GumNet architecture for fingerprint matching with non-linear spatial alignment.

    The spatial-alignment stage solves an adaptive Thin-Plate-Spline (TPS) system whose
    regularization is conditioned on a per-image lambda map (`l_map`). The module exposes
    everything downstream code needs to compute the registration loss: the warped image,
    the learned target control points, the lambda map, and the dense displacement field.

    Args:
        in_channels (int, optional): Number of input channels for images. Defaults to 1.
        grid_size (int, optional): Side length of the TPS control-point grid. Defaults to 4.

    Shape:
        - template:   `(B, in_channels, 192, 192)`
        - impression: `(B, in_channels, 192, 192)`

    Returns:
        warped_impression: `(B, in_channels, 192, 192)`
        target_pts: `(B, grid_size**2, 2)` — learned control points in normalized coords.
        l_map: `(B, 1, 192, 192)` — adaptive regularization field used by the solver.
        displacement_field: `(B, 2, 192, 192)` — dense displacement (warped_coords - base_grid).

    Examples:
        >>> model = GumNet()
        >>> template = torch.randn(4, 1, 192, 192)
        >>> impression = torch.randn(4, 1, 192, 192)
        >>> warped, target_pts, l_map, disp = model(template, impression)
        >>> warped.shape, target_pts.shape, l_map.shape, disp.shape
        (torch.Size([4, 1, 192, 192]), torch.Size([4, 16, 2]), torch.Size([4, 1, 192, 192]), torch.Size([4, 2, 192, 192]))
    """

    def __init__(self, in_channels=1, grid_size=4):
        super(GumNet, self).__init__()

        self.feature_extractor = GumNetFeatureExtraction(in_channels=in_channels)
        self.siamese_matcher = GumNetSiameseMatching()
        self.spatial_aligner = GumNetNonLinearAlignment(grid_size=grid_size)

    def forward(self, template, impression):
        """
        Forward pass through the complete GumNet.

        Args:
            template (torch.Tensor): Template fingerprint image, shape (B, C, 192, 192).
            impression (torch.Tensor): Impression fingerprint to be aligned, shape (B, C, 192, 192).

        Returns:
            warped_impression, target_pts, l_map, displacement_field.
            See class docstring for shapes.
        """
        template_features = self.feature_extractor(template, branch='Sa')      # [B, 512, 14, 14]
        impression_features = self.feature_extractor(impression, branch='Sb')  # [B, 512, 14, 14]

        corr_ab, corr_ba = self.siamese_matcher(template_features, impression_features)

        warped_impression, target_pts, l_map, displacement_field = self.spatial_aligner(
            corr_ab, corr_ba, impression
        )

        return warped_impression, target_pts, l_map, displacement_field
