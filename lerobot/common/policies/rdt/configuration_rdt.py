from dataclasses import dataclass, field

@dataclass
class RDTConfig:
    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.image": [3, 96, 96],
            "observation.state": [2],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [2],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.image": "mean_std",
            "observation.state": "min_max",
        }
    )
    output_normalization_modes: dict[str, str] = field(default_factory=lambda: {"action": "min_max"})

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    # Unet.
    condition_embed_dim :int = 256
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        image_keys = {k for k in self.input_shapes if k.startswith("observation.image")}

        if len(image_keys) == 0 and "observation.environment_state" not in self.input_shapes:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if len(image_keys) > 0:
            if self.crop_shape is not None:
                for image_key in image_keys:
                    if (
                        self.crop_shape[0] > self.input_shapes[image_key][1]
                        or self.crop_shape[1] > self.input_shapes[image_key][2]
                    ):
                        raise ValueError(
                            f"`crop_shape` should fit within `input_shapes[{image_key}]`. Got {self.crop_shape} "
                            f"for `crop_shape` and {self.input_shapes[image_key]} for "
                            "`input_shapes[{image_key}]`."
                        )
            # Check that all input images have the same shape.
            first_image_key = next(iter(image_keys))
            for image_key in image_keys:
                if self.input_shapes[image_key] != self.input_shapes[first_image_key]:
                    raise ValueError(
                        f"`input_shapes[{image_key}]` does not match `input_shapes[{first_image_key}]`, but we "
                        "expect all image shapes to match."
                    )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )