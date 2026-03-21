import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # random seed for reproducibility.
    config.seed = 42
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples. total samples = num_epochs * num_batches_per_epoch * sample.batch_size. for jpeg reward ~1.5k–3k
    # samples to see effect; for harder rewards (aesthetic, llava) often 5k–20k+. with 16 samples/epoch: 100–200 ok
    # for quick runs, 300–500+ for stronger learning.
    config.num_epochs = 100
    # number of epochs between saving model checkpoints.
    config.save_freq = 10
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 5
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    # Use "fp16" on Apple Silicon (MPS) to save memory and speed up; "bf16" is for Ampere+ CUDA.
    config.mixed_precision = "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
    # attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
    # about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
    # files will also be large.
    config.use_lora = True

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    # revision of the model to load.
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 50
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    # batch size (per GPU!) to use for sampling. On M4 48GB, 2–4 is safe; increase if memory allows.
    sample.batch_size = 4
    # number of batches to sample per epoch. samples per epoch = num_batches_per_epoch * batch_size * num_gpus.
    # more batches = more data per epoch but longer epoch; aim for at least hundreds of samples total over training.
    sample.num_batches_per_epoch = 10

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training. Must divide sample.batch_size.
    train.batch_size = 4
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 1
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = True
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the PPO clip range.
    train.clip_range = 1e-4
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 1.0

    ###### Prompt Function ######
    # prompt function to use. see `prompts.py` for available prompt functions.
    config.prompt_fn = "imagenet_animals"
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = {}

    ###### Reward Function ######
    # reward function to use. see `rewards.py` for available reward functions.
    config.reward_fn = "jpeg_aesthetic_combined"
    # optional kwargs for the reward factory (e.g. jpeg_aesthetic_combined weights). empty = call factory with no args.
    config.reward_fn_kwargs = ml_collections.ConfigDict()

    ###### Per-Prompt Stat Tracking ######
    # when enabled, the model will track the mean and std of reward on a per-prompt basis and use that to compute
    # advantages. set `config.per_prompt_stat_tracking` to None to disable per-prompt stat tracking, in which case
    # advantages will be calculated using the mean and std of the entire batch.
    config.per_prompt_stat_tracking = ml_collections.ConfigDict()
    # number of reward values to store in the buffer for each prompt. the buffer persists across epochs.
    config.per_prompt_stat_tracking.buffer_size = 16
    # the minimum number of reward values to store in the buffer before using the per-prompt mean and std. if the buffer
    # contains fewer than `min_count` values, the mean and std of the entire batch will be used instead.
    config.per_prompt_stat_tracking.min_count = 16

    return config
