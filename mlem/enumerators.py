from enum import Enum


class BlackBoxType(Enum):
    """Kind of black box to use."""

    NN = "nn"
    RF = "rf"


class ExplainerType(Enum):
    """Type of local explainer."""

    # LIME
    LIME = "lime"
    # LORE DECISION TREES
    LORE_DTS = "loredts"


class SamplingTechnique(Enum):
    """Random sampling for the explainer."""

    # Gaussian random (LIME)
    GAUSS = "gaussian"
    # Latin Hypercube Sampling (LIME)
    LHS = "lhs"
    # Use the same dataset as the explainer (only for neighborhood) (LIME)
    SAME = "same"
