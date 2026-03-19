import torch

# Kiểm tra các checkpoint
checkpoints = [
    'best_model_50.pth',
    'best_model_full.pth',
    'checkpoints_full/checkpoint_epoch_100.pth'
]

for ckpt_path in checkpoints:
    try:
        print(f"\n{'='*60}")
        print(f"Checking: {ckpt_path}")
        print('='*60)

        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Thông tin cơ bản
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'val_mre' in checkpoint:
            print(f"Val MRE: {checkpoint['val_mre']:.2f}mm")

        # Kiểm tra shape của conv_stem để detect backbone
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            conv_stem_shape = state_dict['backbone.conv_stem.weight'].shape
            print(f"Conv stem shape: {conv_stem_shape}")

            # Detect backbone dựa trên shape
            if conv_stem_shape[0] == 32:
                print("=> Backbone: efficientnet_b0")
            elif conv_stem_shape[0] == 40:
                print("=> Backbone: efficientnet_b3")
            else:
                print(f"=> Unknown backbone (first conv channels: {conv_stem_shape[0]})")

    except FileNotFoundError:
        print(f"File not found: {ckpt_path}")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*60)
