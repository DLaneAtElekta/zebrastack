# InceptionV1 Training Guide - Quick Reference

This guide provides a quick reference for training InceptionV1 from scratch on ImageNet.

## Prerequisites

### 1. ImageNet Dataset Setup

**Download from**: https://www.image-net.org/download.php
- **Approval time**: ~5 days after registration
- **Total size**: ~150 GB

**Required files**:
```
ILSVRC2012_img_train.tar       # 138 GB - 1.3M training images
ILSVRC2012_img_val.tar         # 6.3 GB - 50K validation images
ILSVRC2012_devkit_t12.tar.gz   # Metadata and class labels
```

**Directory structure**:
```
/path/to/imagenet/
├── ILSVRC2012_img_train.tar
├── ILSVRC2012_img_val.tar
├── ILSVRC2012_devkit_t12.tar.gz
└── (will be extracted automatically by PyTorch)
```

### 2. Hardware Requirements

**Minimum**:
- 1× GPU with 16GB+ VRAM (e.g., V100, RTX 3090)
- 64GB+ system RAM
- 500GB+ free disk space

**Recommended**:
- 4-8× GPUs (V100, A100, or equivalent)
- 128GB+ system RAM
- 1TB+ NVMe SSD for dataset

**Training time estimates**:
- 1× V100: ~7-10 days
- 4× V100: ~2-3 days
- 8× A100: ~1-2 days

### 3. Software Dependencies

```bash
# Core dependencies
pip install torch>=2.0 torchvision>=0.15
pip install tensorboard

# Optional but recommended
pip install pytorch-lightning  # Cleaner training code
pip install timm              # Advanced augmentations
pip install wandb             # Experiment tracking

# For visualization integration
pip install lucent pandas pillow
```

## Training Configuration

### Basic Training Setup

```python
# config/inception_training.yaml
model:
  name: inception_v1
  num_classes: 1000
  auxiliary_classifiers: true
  dropout: 0.4

dataset:
  root: /path/to/imagenet
  train_split: train
  val_split: val
  num_workers: 8
  pin_memory: true

training:
  batch_size: 256        # Total across all GPUs
  epochs: 90
  optimizer: sgd
  learning_rate: 0.1
  momentum: 0.9
  weight_decay: 1e-4
  scheduler: step_decay  # or cosine_annealing
  warmup_epochs: 5
  gradient_clip: 1.0

augmentation:
  train_size: 224
  val_size: 224
  resize_size: 256
  random_flip: true
  color_jitter: true
  auto_augment: false    # Enable for better accuracy
  mixup_alpha: 0.0       # Set to 0.2 to enable mixup
```

### Command Line Training

```bash
# Single GPU training
python train_inception.py \
    --config config/inception_training.yaml \
    --data-path /path/to/imagenet \
    --output-dir checkpoints/inception_v1

# Multi-GPU training (4 GPUs)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_inception.py \
    --config config/inception_training.yaml \
    --data-path /path/to/imagenet \
    --output-dir checkpoints/inception_v1 \
    --distributed

# Resume from checkpoint
python train_inception.py \
    --config config/inception_training.yaml \
    --resume checkpoints/inception_v1/checkpoint_epoch_30.pth
```

## Data Preprocessing & Augmentation

### Training Transforms

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### Validation Transforms

```python
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

## Training Best Practices

### 1. Learning Rate Scheduling

**Step Decay** (Original paper):
```python
# Multiply LR by 0.1 at specific epochs
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],
    gamma=0.1
)
```

**Cosine Annealing** (Modern practice):
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=90,  # Total epochs
    eta_min=1e-6
)
```

**Warmup** (First 5 epochs):
```python
def warmup_lr(epoch, warmup_epochs=5, base_lr=0.1):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr
```

### 2. Auxiliary Classifiers

InceptionV1 has two auxiliary classifiers that help with gradient flow:

```python
# During training
main_loss = criterion(main_output, targets)
aux1_loss = criterion(aux1_output, targets)
aux2_loss = criterion(aux2_output, targets)

total_loss = main_loss + 0.3 * aux1_loss + 0.3 * aux2_loss

# During validation/inference
# Only use main_output, ignore auxiliary outputs
```

### 3. Mixed Precision Training

Speed up training with FP16:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4. Distributed Training

For multi-GPU training:

```python
# Initialize distributed training
torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()

# Wrap model
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    find_unused_parameters=True  # For auxiliary classifiers
)

# Use DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    shuffle=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size_per_gpu,
    sampler=train_sampler,
    num_workers=8,
    pin_memory=True
)
```

## Monitoring Training

### Key Metrics to Track

1. **Training metrics**:
   - Loss (main + auxiliary)
   - Top-1 accuracy
   - Top-5 accuracy
   - Learning rate

2. **Validation metrics** (every epoch):
   - Validation loss
   - Top-1 accuracy (target: ~70%)
   - Top-5 accuracy (target: ~90%)

3. **System metrics**:
   - GPU utilization
   - GPU memory usage
   - Data loading time
   - Training throughput (images/sec)

### TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/inception_v1')

# Log training metrics
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Accuracy/train_top1', train_acc1, epoch)
writer.add_scalar('Accuracy/train_top5', train_acc5, epoch)

# Log validation metrics
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/val_top1', val_acc1, epoch)
writer.add_scalar('Accuracy/val_top5', val_acc5, epoch)

# Log learning rate
writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

writer.close()
```

### Checkpointing

```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_acc1': best_acc1,
    'config': config,
}
torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pth')

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_30.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Integration with Visualization System

### Visualize During Training

Generate feature visualizations at key checkpoints:

```python
from enhanced_visualization_generator import EnhancedVisualizationGenerator

# Every 10 epochs
if epoch % 10 == 0:
    # Load checkpoint into visualization system
    model_for_viz = load_model_for_lucent(checkpoint_path)

    generator = EnhancedVisualizationGenerator(
        model_name=f"inception_v1_epoch_{epoch}",
        image_size=384,
        csv_filename=f"visualizations_epoch_{epoch}.csv"
    )

    # Generate sample visualizations
    results = generator.generate_batch(
        num_visualizations=50,
        objective_types=["channel", "gabor"],
        enable_mutation=False
    )
```

### Compare Training Stages

```python
# Compare features at different training stages
compare_checkpoints(
    checkpoint_paths=[
        'checkpoints/checkpoint_epoch_0.pth',   # Random init
        'checkpoints/checkpoint_epoch_30.pth',  # Early training
        'checkpoints/checkpoint_epoch_60.pth',  # Mid training
        'checkpoints/checkpoint_epoch_90.pth',  # Fully trained
    ],
    layers=['mixed3a', 'mixed4a', 'mixed5a'],
    objective_type='gabor'
)
```

## Expected Results

### Target Metrics (90 epochs)

- **Top-1 Accuracy**: ~69-70%
- **Top-5 Accuracy**: ~89-90%
- **Top-5 Error**: ~6.7% (original paper)

### Convergence Timeline

| Epoch | Top-1 Acc | Top-5 Acc | Notes |
|-------|-----------|-----------|-------|
| 0     | ~0.1%     | ~0.5%     | Random initialization |
| 10    | ~40-50%   | ~65-75%   | Basic features learned |
| 30    | ~60-65%   | ~82-87%   | Before first LR decay |
| 60    | ~68-69%   | ~88-89%   | After second LR decay |
| 90    | ~69-70%   | ~89-90%   | Final convergence |

## Troubleshooting

### Common Issues

**1. Out of memory errors**:
```bash
# Reduce batch size
--batch-size 128  # Instead of 256

# Enable gradient accumulation
--accumulation-steps 2

# Use mixed precision
--fp16
```

**2. Slow data loading**:
```bash
# Increase workers
--num-workers 16

# Use faster storage (NVMe SSD)
# Pre-extract ImageNet to local disk
```

**3. Training instability**:
```bash
# Enable gradient clipping
--gradient-clip 1.0

# Reduce learning rate
--learning-rate 0.05

# Add warmup
--warmup-epochs 5
```

**4. Low GPU utilization**:
- Check data loading is not bottleneck
- Increase batch size if memory allows
- Reduce number of workers if CPU bound

## Advanced Topics

### Custom Objectives from Visualizations

Train model to produce better visualizations:

```python
# Add visualization quality loss
viz_loss = compute_visualization_quality(
    model,
    layer='mixed4a',
    metric='depth_complexity'
)

total_loss = classification_loss + 0.1 * viz_loss
```

### Transfer Learning

Fine-tune trained model on custom dataset:

```python
# Load ImageNet-trained checkpoint
model = load_checkpoint('checkpoints/best_model.pth')

# Replace final classifier
model.fc = nn.Linear(1024, num_custom_classes)

# Fine-tune with lower learning rate
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,  # 100x lower than initial training
    momentum=0.9
)
```

## References

- Original Paper: "Going Deeper with Convolutions" (Szegedy et al., 2014)
- ImageNet Dataset: https://www.image-net.org/
- PyTorch ImageNet Training: https://github.com/pytorch/examples/tree/main/imagenet
- Modern Training Techniques: https://pytorch.org/tutorials/

## See Also

- `FEATURE_ROADMAP.md` - Complete feature roadmap including training
- `CLAUDE.md` - General codebase documentation
- `comprehensive_demo.py` - Example visualization workflows
