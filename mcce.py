class ConditionalBatchNorm3d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1,
            affine=False, track_running_stats=True)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        # Initialise bias at 0
        self.embed.weight.data[:, num_features:].zero_()
    def forward(self, x, class_id):
        out = self.bn(x)
        gamma, beta = self.embed(class_id).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1, 1) * out
            + beta.view(-1, self.num_features,1,1,1)
