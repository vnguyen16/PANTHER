# patch-level classification
*look for red circle comments*
- modified PANTHER class in model_PANTHER.py
- modified trainer.py --> def validate_classification, def train_loop_classification
- modified model_linear.py (linear classifier)

# normalization
*look for blue downwards arrow comments*
- modified proto_utils.py --> def cluster (to normalize prototype centroids)
- layers.py --> class PANTHERBase(nn.Module) --> def forward (to normalize patch features)
- networks.py --> (to normalize patch features) def mog_eval 