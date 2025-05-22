from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from typing import List, Tuple, Optional
class VanillaSGMRequestTrain(BaseModel):
    model_name: str
    dataset: str = Field("mnist", description="Name of the dataset")
    batch_size: int = Field(128, description="Size of the batch")
    n_epochs: Optional[int] = Field(100, description="Number of epochs")
    num_samples: Optional[int] = Field(16, description="Number of samples to generate")
    num_steps: Optional[int] = Field(500, description="Number of steps")
    num_workers: int = Field(0, description="Number of workers for the dataloader")
    sigma: float = Field(25.0, description="Sigma value (noise level)")
    lr: float = Field(5e-4, description="Learning rate")
    model_channels: List[int] = Field([32, 64, 128, 256], description="Channels of the model")
    embed_dim: int = Field(256, description="Embedding dimension size")
    sample_and_save_freq: int = Field(10, description="Frequency for sampling and saving")
    sampler_type: str = Field("PC", description="Type of the sampler", choices=["EM", "PC", "ODE"])
    snr: float = Field(0.16, description="Signal-to-noise ratio")
    atol: float = Field(1e-6, description="Absolute tolerance")
    rtol: float = Field(1e-6, description="Relative tolerance")
    eps: float = Field(1e-3, description="Smallest timestep for numerical stability")


class NCSNv2RequestTrain(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    dataset: str = Field("mnist", description="Name of the dataset")
    batch_size: int = Field(128, description="Batch size")
    num_workers: int = Field(0, description="Number of workers for the dataloader")
    centered: Optional[bool] = Field(False, description="Indicates whether the data is centered")
    normalization: Literal['InstanceNorm', 'GroupNorm', 'VarianceNorm', 'InstanceNorm++'] = Field(
        "InstanceNorm++",
        description="Type of normalization to apply"
    )
    nf: Optional[int] = Field(128, description="Base number of filters")
    act: str = Field("elu", description="Activation function name")
    sigma_min: Optional[float] = Field(0.01, description="Minimum sigma value")
    sigma_max: Optional[float] = Field(50, description="Maximum sigma value")
    num_scales: Optional[float] = Field(232, description="Number of noise scales")
    ema_decay: Optional[float] = Field(0.999, description="EMA decay rate")
    lr: Optional[float] = Field(5e-4, description="Learning rate")
    n_epochs: Optional[int] = Field(100, description="Number of training epochs")
    beta1: Optional[float] = Field(0.9, description="Beta1 value for the Adam optimizer")
    beta2: Optional[float] = Field(0.999, description="Beta2 value for the Adam optimizer")
    weight_decay: Optional[float] = Field(0.0, description="Weight decay coefficient")
    warmup: Optional[int] = Field(0, description="Number of warmup steps")
    grad_clip: Optional[float] = Field(-1.0, description="Gradient clipping value")
    sample_and_save_freq: Optional[int] = Field(5, description="Frequency of sampling and saving")
    sampler: Optional[Literal['pc', 'ode']] = Field('pc', description="Type of sampler to use")
    predictor: Optional[Literal['none', 'em', 'rd', 'as']] = Field('none', description="Type of predictor used during sampling")
    corrector: Optional[Literal['none', 'l', 'ald']] = Field('ald', description="Type of corrector used during sampling")
    snr: Optional[float] = Field(0.176, description="Signal-to-noise ratio")
    n_steps: Optional[int] = Field(5, description="Number of sampling steps")
    probability_flow: Optional[bool] = Field(False, description="Enable probability flow during sampling")
    noise_removal: Optional[bool] = Field(False, description="Enable noise removal")
    continuous: Optional[bool] = Field(False, description="Use continuous mode")
    reduce_mean: Optional[bool] = Field(False, description="Use mean reduction")
    likelihood_weighting: Optional[bool] = Field(False, description="Enable likelihood weighting")


class VanillaSGMRequestSample(BaseModel):
    model_name: str  # Name of the model to be used for sampling

    num_samples: Optional[int] = Field(16, description="Number of samples to generate")
    dataset: str = Field("mnist", description="Name of the dataset used for model context")
    batch_size: int = Field(128, description="Batch size for the sampling process")

    snr: float = Field(0.16, description="Signal-to-noise ratio used during sampling")
    eps: float = Field(1e-3, description="Smallest time step to ensure numerical stability")
    
    sampler_type: str = Field(
        "PC", 
        description="Type of sampler to use during sample generation",
        choices=["EM", "PC", "ODE"]
    )
    num_steps: Optional[int] = Field(500, description="Number of denoising steps to apply")
    
    atol: float = Field(1e-6, description="Absolute tolerance for the ODE/PC solver")
    rtol: float = Field(1e-6, description="Relative tolerance for the ODE/PC solver")
    
    sigma: float = Field(25.0, description="Noise level for the diffusion process")

    n_epochs: Optional[int] = Field(100, description="(Optional) Number of epochs if relevant to the sampler")
    lr: float = Field(5e-4, description="Learning rate associated with training or fine-tuning before sampling")
    
    model_channels: List[int] = Field(
        [32, 64, 128, 256], 
        description="List defining the number of channels at each layer of the model"
    )
    embed_dim: int = Field(256, description="Dimension of the embedding vectors used in the model")

    sample_and_save_freq: int = Field(
        10, 
        description="Frequency (in steps) at which samples are generated and saved"
    )


class NCSNv2RequestSample(BaseModel):
    model_name: str = Field(..., description="Name of the model to use for sampling")
    dataset: str = Field("mnist", description="Name of the dataset")
    batch_size: int = Field(128, description="Batch size used during sampling")
    num_workers: int = Field(0, description="Number of workers used by the DataLoader")

    centered: Optional[bool] = Field(False, description="Indicates whether the input data is centered")
    
    normalization: Literal['InstanceNorm', 'GroupNorm', 'VarianceNorm', 'InstanceNorm++'] = Field(
        "InstanceNorm++",
        description="Type of normalization applied to the data"
    )

    nf: Optional[int] = Field(128, description="Base number of filters for the model")
    act: str = Field("elu", description="Activation function used in the network")
    
    sigma_min: Optional[float] = Field(0.01, description="Minimum sigma value for diffusion process")
    sigma_max: Optional[float] = Field(50, description="Maximum sigma value for diffusion process")
    num_scales: Optional[float] = Field(232, description="Number of scales used for noise levels")
    
    ema_decay: Optional[float] = Field(0.999, description="Exponential Moving Average decay rate")
    lr: Optional[float] = Field(5e-4, description="Learning rate")
    n_epochs: Optional[int] = Field(100, description="Number of epochs (if applicable)")
    
    beta1: Optional[float] = Field(0.9, description="Beta1 parameter for the Adam optimizer")
    beta2: Optional[float] = Field(0.999, description="Beta2 parameter for the Adam optimizer")
    weight_decay: Optional[float] = Field(0.0, description="Weight decay (L2 regularization)")
    warmup: Optional[int] = Field(0, description="Number of warm-up steps for the optimizer")
    
    grad_clip: Optional[float] = Field(-1.0, description="Value for gradient clipping to prevent exploding gradients")

    sample_and_save_freq: Optional[int] = Field(5, description="How often to sample and save images (in steps)")
    
    sampler: Optional[Literal['pc', 'ode']] = Field('pc', description="Sampling algorithm type (predictor-corrector or ODE-based)")
    predictor: Optional[Literal['none', 'em', 'rd', 'as']] = Field('none', description="Predictor type used during sampling")
    corrector: Optional[Literal['none', 'l', 'ald']] = Field('ald', description="Corrector type used during sampling")
    
    snr: Optional[float] = Field(0.176, description="Signal-to-noise ratio for the sampling process")
    n_steps: Optional[int] = Field(5, description="Number of sampling steps")
    
    probability_flow: Optional[bool] = Field(False, description="Whether to use probability flow-based sampling")
    noise_removal: Optional[bool] = Field(False, description="Whether to perform noise removal in final outputs")
    
    continuous: Optional[bool] = Field(False, description="Enable continuous mode for sampling")
    reduce_mean: Optional[bool] = Field(False, description="Whether to apply reduce mean operation on loss/outputs")
    likelihood_weighting: Optional[bool] = Field(False, description="Enable likelihood weighting in training/sampling")

    num_samples: Optional[int] = Field(16, description="Number of samples to generate")

class VanillaVAERequestTrain(BaseModel):
    model_name: str
    train: bool = Field(default=False, description="Train model")
    sample: bool = Field(default=False, description="Sample model")
    dataset: str = Field(default="mnist", description="Dataset name", choices=["mnist", "cifar10", "cifar100", "places365", "dtd", "fashionmnist", "chestmnist", "octmnist", "tissuemnist", "pneumoniamnist", "svhn", "tinyimagenet", "imagenet"])
    out_dataset: str = Field(default="fashionmnist", description="Outlier dataset name", choices=["mnist", "cifar10", "cifar100", "places365", "dtd", "fashionmnist", "chestmnist", "octmnist", "tissuemnist", "pneumoniamnist", "svhn", "tinyimagenet", "imagenet"])
    batch_size: int = Field(default=128, description="Batch size")
    n_epochs: int = Field(default=100, description="Number of epochs")
    lr: float = Field(default=0.0002, description="Learning rate")
    latent_dim: int = Field(default=128, description="Latent dimension")
    hidden_dims: Optional[List[int]] = Field(default=None, description="Hidden dimensions")
    checkpoint: Optional[str] = Field(default=None, description="Checkpoint path")
    num_samples: int = Field(default=16, description="Number of samples")
    sample_and_save_freq: int = Field(default=5, description="Sample and save frequency")
    loss_type: str = Field(default="mse", description="Loss type", choices=["mse", "ssim"])
    kld_weight: float = Field(default=1e-4, description="KL weight")
    num_workers: int = Field(default=0, description="Number of workers for dataloader")

class HierarchicalVAERequestTrain(BaseModel):
    model_name: str
    train: bool = Field(False, description="Train the model")
    sample: bool = Field(False, description="Sample the model")
    dataset: str = Field(
        'mnist', 
        description="Dataset name",
        choices=['mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 
                 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'tinyimagenet', 'imagenet']
    )
    batch_size: int = Field(256, description="Batch size")
    n_epochs: int = Field(100, description="Number of epochs")
    lr: float = Field(0.01, description="Learning rate")
    latent_dim: int = Field(512, description="Latent dimension")
    checkpoint: Optional[str] = Field(None, description="Checkpoint path")
    sample_and_save_freq: int = Field(5, description="Sample and save frequency")
    num_workers: int = Field(0, description="Number of workers for dataloader")

class ConditionalVAERequestTrain(BaseModel):
    model_name: str
    train: bool = Field(False, description="Train the model")
    sample: bool = Field(False, description="Sample the model")
    dataset: str = Field(
        'mnist', 
        description="Dataset name",
        choices=['mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 
                 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'tinyimagenet', 'imagenet']
    )
    batch_size: int = Field(128, description="Batch size")
    n_epochs: int = Field(100, description="Number of epochs")
    lr: float = Field(0.0002, description="Learning rate")
    latent_dim: int = Field(128, description="Latent dimension")
    hidden_dims: Optional[List[int]] = Field(None, description="Hidden dimensions")
    checkpoint: Optional[str] = Field(None, description="Checkpoint path")
    num_samples: int = Field(16, description="Number of samples")
    num_classes: int = Field(10, description="Number of classes")
    kld_weight: float = Field(1e-4, description="KL weight")
    loss_type: str = Field(
        'mse', 
        description="Loss type",
        choices=['mse', 'ssim']
    )
    sample_and_save_freq: int = Field(5, description="Sample and save frequency")
    num_workers: int = Field(0, description="Number of workers for dataloader")

class VanillaVAERequestSample(BaseModel):
    model_name: str
    sample: bool = Field(default=False, description="Sample model")
    dataset: str = Field(
        default="mnist", 
        description="Dataset name", 
        choices=["mnist", "cifar10", "cifar100", "places365", "dtd", "fashionmnist", "chestmnist", "octmnist", "tissuemnist", "pneumoniamnist", "svhn", "tinyimagenet", "imagenet"]
    )
    out_dataset: str = Field(
        default="fashionmnist", 
        description="Outlier dataset name", 
        choices=["mnist", "cifar10", "cifar100", "places365", "dtd", "fashionmnist", "chestmnist", "octmnist", "tissuemnist", "pneumoniamnist", "svhn", "tinyimagenet", "imagenet"]
    )
    batch_size: int = Field(default=128, description="Batch size")
    n_epochs: int = Field(default=100, description="Number of epochs")
    lr: float = Field(default=0.0002, description="Learning rate")
    latent_dim: int = Field(default=128, description="Latent dimension")
    hidden_dims: Optional[List[int]] = Field(default=None, description="Hidden dimensions")
    checkpoint: Optional[str] = Field(default=None, description="Checkpoint path")
    num_samples: int = Field(default=16, description="Number of samples")
    sample_and_save_freq: int = Field(default=5, description="Sample and save frequency")
    loss_type: str = Field(
        default="mse", 
        description="Loss type", 
        choices=["mse", "ssim"]
    )
    kld_weight: float = Field(default=1e-4, description="KL weight")
    num_workers: int = Field(default=0, description="Number of workers for dataloader")
class CondDDPMRequestTrain(BaseModel):
    train: bool = Field(False, description="Train model")
    sample: bool = Field(False, description="Sample model")
    num_samples: int = Field(default=16, description="Number of samples")
    outlier_detection: bool = Field(False, description="Outlier detection")
    batch_size: int = Field(128, description="Batch size")
    n_epochs: int = Field(100, description="Number of epochs")
    lr: float = Field(1e-3, description="Learning rate")
    timesteps: int = Field(500, description="Number of timesteps")
    beta_start: float = Field(0.0001, description="Beta start")
    beta_end: float = Field(0.02, description="Beta end")
    dataset: str = Field("mnist", description="Dataset name", choices=['mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'tinyimagenet', 'imagenet'])
    ddpm: float = Field(1.0, description="DDIM sampling is 0.0, pure DDPM is 1.0")
    checkpoint: Optional[str] = Field(None, description="Checkpoint path")
    out_dataset: str = Field("fashionmnist", description="Outlier dataset name", choices=['mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'tinyimagenet', 'imagenet'])
    sample_timesteps: int = Field(500, description="Number of timesteps for sampling")
    n_features: int = Field(128, description="Number of features")
    n_classes: int = Field(10, description="Number of classes")
    sample_and_save_freq: int = Field(10, description="Sample and save frequency")
    drop_prob: float = Field(0.1, description="Dropout probability")
    guide_w: float = Field(0.5, description="Guide weight")
    ws_test: List[float] = Field([0.0, 0.5, 2.0], description="Guidance weights for test")
    num_workers: int = Field(0, description="Number of workers for dataloader")

class DiffAERequestTrain(BaseModel):
    train: bool = Field(False, description="Train model")
    manipulate: bool = Field(False, description="Manipulate latents")
    batch_size: int = Field(16, description="Batch size")
    n_epochs: int = Field(100, description="Number of epochs")
    lr: float = Field(1e-3, description="Learning rate")
    timesteps: int = Field(1000, description="Number of timesteps")
    sample_timesteps: int = Field(100, description="Number of timesteps for sampling")
    dataset: str = Field("mnist", description="Dataset name", choices=['mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 'tinyimagenet', 'imagenet', 'celeba'])
    checkpoint: Optional[str] = Field(None, description="Checkpoint path")
    embedding_dim: int = Field(512, description="Embedding dimension")
    embedding_dimension: Optional[int] = Field(None, description="Alternative embedding dimension")
    model_channels: Tuple[int, ...] = Field((64, 128, 256), description="Model channels")
    attention_levels: Tuple[bool, ...] = Field((False, True, True), description="Attention levels (must match length of model_channels)")
    num_res_blocks: int = Field(1, description="Number of residual blocks")
    sample_and_save_freq: int = Field(10, description="Sample and save frequency")
    num_workers: int = Field(0, description="Number of workers for dataloader")

class VanDDPMTrain(BaseModel):
    train: bool = False
    sample: bool = False
    outlier_detection: bool = False
    batch_size: int = 128
    n_epochs: int = 100
    lr: float = 1e-3
    timesteps: int = 300
    n_features: int = 64
    init_channels: int = 32
    channel_scale_factors: Tuple[int, ...] = Field((1, 2, 2), description="Model channels")
    resnet_block_groups: int = 8
    use_convnext: bool = True
    convnext_scale_factor: int = 2
    beta_start: float = 0.0001
    beta_end: float = 0.02
    sample_and_save_freq: int = 5
    dataset: str = Field("mnist", choices=["mnist", "cifar10", "cifar100", "places365", "dtd", "fashionmnist", "chestmnist", "octmnist", "tissuemnist", "pneumoniamnist", "svhn", "tinyimagenet","imagenet"])
    ddpm: float = 1.0
    checkpoint: str = None
    num_samples: int = 16
    out_dataset: str = Field("fashionmnist", choices=["mnist", "cifar10", "cifar100", "places365", "dtd", "fashionmnist", "chestmnist", "octmnist", "tissuemnist", "pneumoniamnist", "svhn", "tinyimagenet","imagenet"])
    loss_type: str = Field("huber", choices=["huber","l2", "l1"])
    sample_timesteps: int = 300
    num_workers: int = 0

class PixelCNNTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample from model")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    batch_size: int = Field(default=128, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=1e-3, description="learning rate")
    gamma: float = Field(default=0.99, description="gamma for the lr scheduler")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    hidden_channels: int = Field(default=64, description="number of channels for the convolutional layers")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")


class VQVAETransformerTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample model")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    batch_size: int = Field(default=128, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs for VQVAE")
    lr: float = Field(default=0.0002, description="learning rate VQVAE")
    lr_t: float = Field(default=0.0002, description="learning rate transformer")
    n_epochs_t: int = Field(default=100, description="number of epochs transformer")
    num_res_layers: int = Field(default=2, description="number of residual layers")
    downsample_parameters: Tuple[Tuple[int, ...], Tuple[int, ...]] = Field(
        default=((2, 4, 1, 1), (2, 4, 1, 1)),
        description="downsample parameters (duplicated as tuple tuple)"
    )
    upsample_parameters: Tuple[Tuple[int, ...], Tuple[int, ...]] = Field(
        default=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        description="upsample parameters (duplicated as tuple tuple)"
    )
    num_channels: Tuple[int, ...] = Field(default=(256, 256), description="number of channels")
    num_res_channels: Tuple[int, ...] = Field(default=(256, 256), description="number of res channels")
    num_embeddings: int = Field(default=256, description="number of embeddings")
    embedding_dim: int = Field(default=32, description="embedding dimension")
    attn_layers_dim: int = Field(default=96, description="attn layers dim")
    attn_layers_depth: int = Field(default=12, description="attn layers depth")
    attn_layers_heads: int = Field(default=8, description="attn layers heads")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path to VQVAE")
    checkpoint_t: Optional[str] = Field(default=None, description="checkpoint path to Transformer")
    num_samples: int = Field(default=16, description="number of samples")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")

class VQGANTransformerTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample model")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    batch_size: int = Field(default=128, description="batch size")
    n_epochs: int = Field(default=1, description="number of epochs for VQVAE")
    lr: float = Field(default=0.0002, description="learning rate VQVAE")
    lr_d: float = Field(default=0.0005, description="learning rate discriminator")
    adv_weight: float = Field(default=0.01, description="adversarial weight")
    perceptual_weight: float = Field(default=0.001, description="perceptual weight")
    lr_t: float = Field(default=0.0005, description="learning rate transformer")
    n_epochs_t: int = Field(default=100, description="number of epochs transformer")
    num_res_layers: int = Field(default=2, description="number of residual layers")
    downsample_parameters: Tuple[Tuple[int, ...], Tuple[int, ...]] = Field(
        default=((2, 4, 1, 1), (2, 4, 1, 1)),
        description="downsample parameters (duplicated as tuple tuple)"
    )
    upsample_parameters: Tuple[Tuple[int, ...], Tuple[int, ...]] = Field(
        default=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        description="upsample parameters (duplicated as tuple tuple)"
    )
    num_channels: Tuple[int, ...] = Field(default=(256, 256), description="number of channels")
    num_res_channels: Tuple[int, ...] = Field(default=(256, 256), description="number of res channels")
    num_embeddings: int = Field(default=256, description="number of embeddings")
    embedding_dim: int = Field(default=32, description="embedding dimension")
    attn_layers_dim: int = Field(default=96, description="attn layers dim")
    attn_layers_depth: int = Field(default=12, description="attn layers depth")
    attn_layers_heads: int = Field(default=8, description="attn layers heads")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path to VQVAE")
    checkpoint_t: Optional[str] = Field(default=None, description="checkpoint path to Transformer")
    num_samples: int = Field(default=16, description="number of samples")
    num_layers_d: int = Field(default=3, description="number of layers in discriminator")
    num_channels_d: int = Field(default=64, description="number of channels in discriminator")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")

class VanillaGANTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample from model")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    batch_size: int = Field(default=128, description="batch size")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    n_epochs: int = Field(default=100, description="number of epochs")
    lrg: float = Field(default=0.0002, description="learning rate generator")
    lrd: float = Field(default=0.0002, description="learning rate discriminator")
    beta1: float = Field(default=0.5, description="beta1")
    beta2: float = Field(default=0.999, description="beta2")
    latent_dim: int = Field(default=100, description="latent dimension")
    img_size: int = Field(default=32, description="image size")
    channels: int = Field(default=1, description="channels")
    sample_and_save_freq: int = Field(default=5, description="sample interval")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    discriminator_checkpoint: Optional[str] = Field(default=None, description="discriminator checkpoint path")
    n_samples: int = Field(default=9, description="number of samples")
    d: int = Field(default=128, description="d")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")

class AdversarialVAETrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    test: bool = Field(default=False, description="test model")
    sample: bool = Field(default=False, description="sample model")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    batch_size: int = Field(default=128, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=0.0002, description="learning rate")
    latent_dim: int = Field(default=128, description="latent dimension")
    hidden_dims: Optional[List[int]] = Field(default=None, description="hidden dimensions")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    num_samples: int = Field(default=16, description="number of samples")
    gen_weight: float = Field(default=0.002, description="generator weight")
    recon_weight: float = Field(default=0.002, description="reconstruction weight")
    sample_and_save_frequency: int = Field(default=5, description="sample and save frequency")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    discriminator_checkpoint: Optional[str] = Field(default=None, description="discriminator checkpoint path")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    kld_weight: float = Field(default=1e-4, description="kl weight")
    loss_type: Literal['mse', 'ssim'] = Field(default='mse', description="loss type")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")
    size: Optional[int] = Field(default=None, description="size of image (leave None for default for each dataset)")

class CondGANTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample from model")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    batch_size: int = Field(default=128, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=0.0002, description="learning rate")
    beta1: float = Field(default=0.5, description="beta1")
    beta2: float = Field(default=0.999, description="beta2")
    latent_dim: int = Field(default=100, description="latent dimension")
    n_classes: int = Field(default=10, description="number of classes")
    img_size: int = Field(default=32, description="image size")
    channels: int = Field(default=1, description="channels")
    sample_and_save_freq: int = Field(default=5, description="sample interval")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    num_samples: int = Field(default=9, description="number of samples")
    d: int = Field(default=128, description="d")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")

class CycleGANTrain(BaseModel):
    train: bool = Field(default=True, description="train model")
    test: bool = Field(default=False, description="test model")
    batch_size: int = Field(default=1, description="batch size")
    n_epochs: int = Field(default=200, description="number of epochs")
    lr: float = Field(default=0.0002, description="learning rate")
    decay: float = Field(default=100, description="epoch to start linearly decaying the learning rate to 0")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    dataset: Literal['horse2zebra'] = Field(default='horse2zebra', description="dataset name")
    checkpoint_A: Optional[str] = Field(default=None, description="checkpoint A path")
    checkpoint_B: Optional[str] = Field(default=None, description="checkpoint B path")
    input_size: int = Field(default=128, description="input size")
    in_channels: int = Field(default=3, description="in channels")
    out_channels: int = Field(default=3, description="out channels")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")

class PresGANTrain(BaseModel):
    # Genel ayarlar
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample from model")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    
    # Model argümanları
    nz: int = Field(default=100, description="size of the latent z vector")
    ngf: int = Field(default=64, description="number of generator filters")
    ndf: int = Field(default=64, description="number of discriminator filters")
    
    # Optimizasyon argümanları
    batch_size: int = Field(default=64, description="input batch size")
    n_epochs: int = Field(default=100, description="number of epochs to train for")
    lrD: float = Field(default=0.0002, description="learning rate for discriminator")
    lrG: float = Field(default=0.0002, description="learning rate for generator")
    lrE: float = Field(default=0.0002, description="learning rate for encoder")
    beta1: float = Field(default=0.5, description="beta1 for adam")
    
    # Checkpointing ve Logging argümanları
    checkpoint: Optional[str] = Field(default=None, description="a given checkpoint file for generator")
    discriminator_checkpoint: Optional[str] = Field(default=None, description="a given checkpoint file for discriminator")
    sigma_checkpoint: Optional[str] = Field(default=None, description="a given file for logsigma for the generator")
    num_gen_images: int = Field(default=16, description="number of images to generate for inspection")
    
    # PresGAN spesifik argümanları
    sigma_lr: float = Field(default=0.0002, description="generator variance learning rate")
    lambda_: float = Field(default=0.01, description="entropy coefficient")
    sigma_min: float = Field(default=0.01, description="min value for sigma")
    sigma_max: float = Field(default=0.3, description="max value for sigma")
    logsigma_init: float = Field(default=-1.0, description="initial value for log_sigma_sian")
    num_samples_posterior: int = Field(default=2, description="number of samples from posterior")
    burn_in: int = Field(default=2, description="hmc burn in")
    leapfrog_steps: int = Field(default=5, description="number of leap frog steps for hmc")
    flag_adapt: int = Field(default=1, description="adapt flag (0 or 1)")
    delta: float = Field(default=1.0, description="delta for hmc")
    hmc_learning_rate: float = Field(default=0.02, description="learning rate for hmc")
    hmc_opt_accept: float = Field(default=0.67, description="hmc optimal acceptance rate")
    stepsize_num: float = Field(default=1.0, description="initial value for hmc stepsize")
    restrict_sigma: int = Field(default=0, description="whether to restrict sigma or not")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")

class WassersteinGANTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample from model")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    batch_size: int = Field(default=256, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    latent_dim: int = Field(default=100, description="latent dimension")
    d: int = Field(default=64, description="d")
    lrg: float = Field(default=0.0002, description="learning rate generator")
    lrd: float = Field(default=0.0002, description="learning rate discriminator")
    beta1: float = Field(default=0.5, description="beta1")
    beta2: float = Field(default=0.999, description="beta2")
    sample_and_save_freq: int = Field(default=5, description="sample interval")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    discriminator_checkpoint: Optional[str] = Field(default=None, description="discriminator checkpoint path")
    gp_weight: float = Field(default=10.0, description="gradient penalty weight")
    n_critic: int = Field(default=5, description="number of critic updates per generator update")
    n_samples: int = Field(default=9, description="number of samples")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")

class RealNVPTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample model")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    batch_size: int = Field(default=128, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=1e-3, description="learning rate")
    weight_decay: float = Field(default=1e-5, description="weight decay")
    max_grad_norm: float = Field(default=100.0, description="max grad norm")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    num_scales: int = Field(default=2, description="number of scales")
    mid_channels: int = Field(default=64, description="mid channels")
    num_blocks: int = Field(default=8, description="number of blocks")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")

class VanillaFlowTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample model")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    batch_size: int = Field(default=128, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=1e-3, description="learning rate")
    c_hidden: int = Field(default=16, description="Hidden units in the first coupling layer")
    multi_scale: bool = Field(default=False, description="use multi scale")
    vardeq: bool = Field(default=False, description="use variational dequantization")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    n_layers: int = Field(default=8, description="number of layers")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")
    num_samples: int = Field(default=16, description="number of samples")
class GlowTrain(BaseModel):
    num_samples: int = Field(default=16, description="number of samples")
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample from model")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    batch_size: int = Field(default=128, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=0.0002, description="learning rate")
    hidden_channels: int = Field(default=64, description="hidden channels")
    K: int = Field(default=8, description="Number of layers per block")
    L: int = Field(default=3, description="number of blocks")
    actnorm_scale: float = Field(default=1.0, description="act norm scale")
    flow_permutation: Literal['invconv', 'shuffle', 'reverse'] = Field(
        default='invconv', description="flow permutation"
    )
    flow_coupling: Literal['additive', 'affine'] = Field(
        default='affine', description="flow coupling, affine"
    )
    LU_decomposed: bool = Field(default=False, description="Train with LU decomposed 1x1 convs")
    learn_top: bool = Field(default=False, description="learn top layer (prior)")
    y_condition: bool = Field(default=False, description="Class Conditioned Glow")
    y_weight: float = Field(default=0.01, description="weight of class condition")
    num_classes: int = Field(default=10, description="number of classes")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    n_bits: int = Field(default=8, description="number of bits")
    max_grad_clip: float = Field(default=0.0, description="max grad clip")
    max_grad_norm: float = Field(default=0.0, description="max grad norm")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")
    warmup: int = Field(default=10, description="warmup epochs")
    decay: float = Field(default=1e-5, description="decay rate")

class FlowPPTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample from model")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    batch_size: int = Field(default=8, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=1e-3, description="learning rate")
    warm_up: int = Field(default=200, description="warm up")
    grad_clip: float = Field(default=1.0, description="gradient clip")
    drop_prob: float = Field(default=0.2, description="dropout probability")
    num_blocks: int = Field(default=10, description="number of blocks")
    num_components: int = Field(default=32, description="Number of components in the mixture")
    num_dequant_blocks: int = Field(default=2, description="Number of blocks in dequantization")
    num_channels: int = Field(default=96, description="Number of channels in Flow++")
    use_attn: bool = Field(default=False, description="use attention")
    sample_and_save_freq: int = Field(default=5, description="sample interval")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")

class FlowMatchingTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample model")
    batch_size: int = Field(default=256, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=1e-3, description="learning rate")
    model_channels: int = Field(default=64, description="number of features")
    num_res_blocks: int = Field(default=2, description="number of residual blocks per downsample")
    attention_resolutions: Tuple[int, ...] = Field(default=(4,), description="downsample rates at which attention will take place")
    dropout: float = Field(default=0.0, description="dropout probability")
    channel_mult: Tuple[int, ...] = Field(default=(1, 2, 2), description="channel multiplier for each level of the UNet")
    conv_resample: bool = Field(default=True, description="use learned convolutions for upsampling and downsampling")
    dims: int = Field(default=2, description="determines if the signal is 1D, 2D, or 3D")
    num_heads: int = Field(default=4, description="number of attention heads in each attention layer")
    num_head_channels: int = Field(default=32, description="use a fixed channel width per attention head")
    use_scale_shift_norm: bool = Field(default=False, description="use a FiLM-like conditioning mechanism")
    resblock_updown: bool = Field(default=False, description="use residual blocks for up/downsampling")
    use_new_attention_order: bool = Field(default=False, description="use a different attention pattern for potentially increased efficiency")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    num_samples: int = Field(default=16, description="number of samples")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist', 
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn', 
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    interpolation: bool = Field(default=False, description="interpolation")
    solver_lib: Literal['torchdiffeq', 'zuko', 'none'] = Field(default='none', description="solver library")
    step_size: float = Field(default=0.1, description="step size for ODE solver")
    solver: Literal[
        'dopri5', 'rk4', 'dopri8', 'euler', 'bosh3', 'adaptive_heun', 'midpoint', 'explicit_adams', 'implicit_adams'
    ] = Field(default='dopri5', description="solver for ODE")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")
    warmup: int = Field(default=10, description="warmup epochs")
    decay: float = Field(default=1e-5, description="decay rate")

class CondFlowMatchingTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample model")
    batch_size: int = Field(default=256, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=1e-3, description="learning rate")
    n_features: int = Field(default=64, description="number of features")
    init_channels: int = Field(default=32, description="initial channels")
    channel_scale_factors: Tuple[int, ...] = Field(default=(1, 2, 2), description="channel scale factors")
    resnet_block_groups: int = Field(default=8, description="resnet block groups")
    use_convnext: bool = Field(default=True, description="use convnext (default: True)")
    convnext_scale_factor: int = Field(default=2, description="convnext scale factor (default: 2)")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    num_samples: int = Field(default=16, description="number of samples")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    interpolation: bool = Field(default=False, description="interpolation")
    solver_lib: Literal['torchdiffeq', 'zuko', 'none'] = Field(default='none', description="solver library")
    step_size: float = Field(default=0.1, description="step size for ODE solver")
    solver: Literal[
        'dopri5', 'rk4', 'dopri8', 'euler', 'bosh3', 'adaptive_heun',
        'midpoint', 'explicit_adams', 'implicit_adams'
    ] = Field(default='dopri5', description="solver for ODE")
    num_classes: int = Field(default=10, description="number of classes")
    prob: float = Field(default=0.5, description="probability of conditioning during training")
    guidance_scale: float = Field(default=2.0, description="guidance scale")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")
    warmup: int = Field(default=10, description="warmup epochs")
    decay: float = Field(default=1e-5, description="decay rate")

class RectifiedFlowsTrain(BaseModel):
    train: bool = Field(default=False, description="train model")
    sample: bool = Field(default=False, description="sample model")
    outlier_detection: bool = Field(default=False, description="outlier detection")
    dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='mnist', description="dataset name")
    out_dataset: Literal[
        'mnist', 'cifar10', 'cifar100', 'places365', 'dtd', 'fashionmnist',
        'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn',
        'tinyimagenet', 'imagenet'
    ] = Field(default='fashionmnist', description="outlier dataset name")
    batch_size: int = Field(default=128, description="batch size")
    n_epochs: int = Field(default=100, description="number of epochs")
    lr: float = Field(default=5e-4, description="learning rate")
    patch_size: int = Field(default=2, description="patch size")
    dim: int = Field(default=64, description="dimension")
    n_layers: int = Field(default=6, description="number of layers")
    n_heads: int = Field(default=4, description="number of heads")
    multiple_of: int = Field(default=256, description="multiple of")
    ffn_dim_multiplier: Optional[int] = Field(default=None, description="ffn dim multiplier")
    norm_eps: float = Field(default=1e-5, description="norm eps")
    class_dropout_prob: float = Field(default=0.1, description="class dropout probability")
    sample_and_save_freq: int = Field(default=5, description="sample and save frequency")
    num_classes: int = Field(default=10, description="number of classes")
    checkpoint: Optional[str] = Field(default=None, description="checkpoint path")
    cfg: float = Field(default=1.0, description="label guidance")
    sample_steps: int = Field(default=50, description="number of steps for sampling")
    no_wandb: bool = Field(default=False, description="disable wandb logging")
    num_workers: int = Field(default=0, description="number of workers for dataloader")