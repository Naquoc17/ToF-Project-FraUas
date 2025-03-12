naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-detection-Yolo11$ python3 train.py 
Phase 1 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11l.pt, data=data.yaml, epochs=20, time=None, patience=100, batch=8, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train4, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train4
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  2    173824  ultralytics.nn.modules.block.C3k2            [128, 256, 2, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  2    691712  ultralytics.nn.modules.block.C3k2            [256, 512, 2, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  2   1455616  ultralytics.nn.modules.block.C2PSA           [512, 512, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2    756736  ultralytics.nn.modules.block.C3k2            [1024, 256, 2, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   2365440  ultralytics.nn.modules.block.C3k2            [768, 512, 2, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, [256, 512, 512]]          
YOLO11l summary: 631 layers, 25,312,793 parameters, 25,312,777 gradients, 87.3 GFLOPs

Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/labels.cache... 12819 images, 1832 backgrounds, 0 corrupt: 100%|██████████| 12819/12819 [00:00<?, ?it/s]
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0002_jpg.rf.573a42f8536acbbec0af38bfe5765558.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0002_jpg.rf.62a16c681fbe140845363a73b8e533ab.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0002_jpg.rf.df975580c65051afaf8b500231ba2c53.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0008_jpg.rf.5fd743739143b274c1e1956c6b60a2ba.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0008_jpg.rf.7993d302e9b33a034d4d40b39a08de8b.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0008_jpg.rf.f2307ebe8d702f1f7883911116a76793.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0020_jpg.rf.52095f5084038bd9ab6221e38b3beaf5.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0020_jpg.rf.8fa7be8ff0fecffb2bb080cf812fc803.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0020_jpg.rf.af904754c4a13eeec209b56a76dfd850.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0022_jpg.rf.8fd311f2649e576153b3843c234ae5d2.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0022_jpg.rf.d8efff0597f0648c603106c996360d2c.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0030_jpg.rf.3bf786992a099b6e2963194a85dd9975.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0030_jpg.rf.d8dc9d29ce2ff304fb86f9fdc19c6599.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0030_jpg.rf.e9043fa6c8aaaca65b3bcddf790ac8d1.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0033_jpg.rf.1864330f9bca38879e7c23db8a98b43b.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0033_jpg.rf.9e861bd96ac733d7425ac6c80e193996.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0033_jpg.rf.c24b49bae3e3c7e5e59bbae973e2717d.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0037_jpg.rf.3e82d3b34e5a4b3228080a81a408cd77.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0037_jpg.rf.9e320be7fd15adf12e8a814af3301b21.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0037_jpg.rf.d3d1246f7d26ad658406a1df2bd9a076.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0046_jpg.rf.02ab5a76ec535bc4327d13a9aa2424f0.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0046_jpg.rf.4609c34b088e01e2a4043b53b244cfe7.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0046_jpg.rf.9af37f6b8a5577ae282dbd3b1f768ef5.jpg: 1 duplicate labels removed
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 8578, len(boxes) = 13216. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
val: Scanning /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/valid/labels.cache... 535 images, 0 backgrounds, 0 corrupt: 100%|██████████| 535/535 [00:00<?, ?it/s]
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 271, len(boxes) = 601. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
Plotting labels to runs/detect/train4/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train4
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/20      5.66G     0.7283      1.357      1.213          6        640: 100%|██████████| 1603/1603 [03:45<00:00,  7.12it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.01it/s]
                   all        535        601      0.665      0.686      0.668      0.513

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/20      5.81G     0.6444      1.138      1.135          5        640: 100%|██████████| 1603/1603 [03:30<00:00,  7.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.49it/s]
                   all        535        601      0.654      0.742      0.721      0.597

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/20      5.87G     0.6079      1.062      1.101          9        640: 100%|██████████| 1603/1603 [03:25<00:00,  7.79it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.48it/s]
                   all        535        601      0.656      0.633      0.683      0.559

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/20      5.86G     0.5695          1      1.079         11        640: 100%|██████████| 1603/1603 [03:26<00:00,  7.76it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.54it/s]
                   all        535        601      0.691      0.852      0.764      0.639

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/20      5.81G     0.5417     0.9551      1.058          9        640: 100%|██████████| 1603/1603 [03:27<00:00,  7.74it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.48it/s]
                   all        535        601      0.696      0.842      0.756      0.633

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/20      5.86G     0.5239     0.9084      1.043          8        640: 100%|██████████| 1603/1603 [03:25<00:00,  7.80it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.46it/s]
                   all        535        601      0.722      0.798      0.767      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/20      5.86G     0.5009     0.8793      1.031          9        640: 100%|██████████| 1603/1603 [03:25<00:00,  7.82it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.48it/s]
                   all        535        601       0.77      0.872      0.836      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/20      5.87G        0.5      0.857       1.03          5        640: 100%|██████████| 1603/1603 [03:25<00:00,  7.79it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.54it/s]
                   all        535        601      0.797      0.872      0.868      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/20      5.86G     0.4926     0.8429      1.026         11        640: 100%|██████████| 1603/1603 [03:24<00:00,  7.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.59it/s]
                   all        535        601      0.755      0.926      0.851      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/20      5.87G     0.4773     0.8282      1.018          8        640: 100%|██████████| 1603/1603 [03:23<00:00,  7.86it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.60it/s]
                   all        535        601      0.787      0.887      0.869      0.698
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/20      5.86G     0.5867     0.7649      1.087          2        640: 100%|██████████| 1603/1603 [03:22<00:00,  7.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.68it/s]
                   all        535        601      0.785      0.857      0.847      0.689

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/20      5.85G     0.5654     0.7309      1.075          3        640: 100%|██████████| 1603/1603 [03:23<00:00,  7.86it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.67it/s]
                   all        535        601       0.77      0.865       0.85        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/20      5.82G      0.554     0.7124      1.065          2        640: 100%|██████████| 1603/1603 [03:23<00:00,  7.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.49it/s]
                   all        535        601      0.784      0.833      0.837      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/20      5.87G     0.5491     0.6905      1.063          3        640: 100%|██████████| 1603/1603 [03:23<00:00,  7.88it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.70it/s]
                   all        535        601      0.774      0.855      0.814      0.693

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/20      5.86G     0.5335     0.6782      1.048          3        640: 100%|██████████| 1603/1603 [03:24<00:00,  7.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.72it/s]
                   all        535        601      0.824      0.817      0.839      0.711

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/20      5.86G     0.5219     0.6552      1.042          1        640: 100%|██████████| 1603/1603 [03:24<00:00,  7.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.70it/s]
                   all        535        601      0.765       0.89      0.833       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/20      5.86G     0.5126     0.6407      1.038          3        640: 100%|██████████| 1603/1603 [03:22<00:00,  7.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.64it/s]
                   all        535        601      0.808      0.869      0.857      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/20      5.86G     0.5029     0.6266      1.034          2        640: 100%|██████████| 1603/1603 [03:23<00:00,  7.87it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.56it/s]
                   all        535        601      0.821      0.829      0.849      0.714

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/20      5.86G     0.4895      0.612      1.027          5        640: 100%|██████████| 1603/1603 [03:23<00:00,  7.86it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.74it/s]
                   all        535        601      0.805      0.851       0.84      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/20      5.86G     0.4791     0.6012      1.023          3        640: 100%|██████████| 1603/1603 [03:22<00:00,  7.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.56it/s]
                   all        535        601      0.803      0.878      0.844      0.719

20 epochs completed in 1.168 hours.
Optimizer stripped from runs/detect/train4/weights/last.pt, 51.2MB
Optimizer stripped from runs/detect/train4/weights/best.pt, 51.2MB

Validating runs/detect/train4/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11l summary (fused): 464 layers, 25,281,625 parameters, 0 gradients, 86.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:03<00:00, 10.55it/s]
                   all        535        601      0.806      0.879      0.844      0.718
                 human        142        185       0.86      0.962      0.951      0.819
             no-object        327        327      0.994       0.98      0.994      0.994
      undefined-object         66         89      0.564      0.697      0.587      0.343
Speed: 0.1ms preprocess, 3.9ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs/detect/train4
Phase 2 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11l.pt, data=data.yaml, epochs=30, time=None, patience=100, batch=25, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train42, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0005, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train42

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  2    173824  ultralytics.nn.modules.block.C3k2            [128, 256, 2, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  2    691712  ultralytics.nn.modules.block.C3k2            [256, 512, 2, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  2   1455616  ultralytics.nn.modules.block.C2PSA           [512, 512, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2    756736  ultralytics.nn.modules.block.C3k2            [1024, 256, 2, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   2365440  ultralytics.nn.modules.block.C3k2            [768, 512, 2, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, [256, 512, 512]]          
YOLO11l summary: 631 layers, 25,312,793 parameters, 25,312,777 gradients, 87.3 GFLOPs

Transferred 1015/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/labels.cache... 12819 images, 1832 backgrounds, 0 corrupt: 100%|██████████| 12819/12819 [00:00<?, ?it/s]
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0002_jpg.rf.573a42f8536acbbec0af38bfe5765558.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0002_jpg.rf.62a16c681fbe140845363a73b8e533ab.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0002_jpg.rf.df975580c65051afaf8b500231ba2c53.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0008_jpg.rf.5fd743739143b274c1e1956c6b60a2ba.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0008_jpg.rf.7993d302e9b33a034d4d40b39a08de8b.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0008_jpg.rf.f2307ebe8d702f1f7883911116a76793.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0020_jpg.rf.52095f5084038bd9ab6221e38b3beaf5.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0020_jpg.rf.8fa7be8ff0fecffb2bb080cf812fc803.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0020_jpg.rf.af904754c4a13eeec209b56a76dfd850.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0022_jpg.rf.8fd311f2649e576153b3843c234ae5d2.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0022_jpg.rf.d8efff0597f0648c603106c996360d2c.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0030_jpg.rf.3bf786992a099b6e2963194a85dd9975.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0030_jpg.rf.d8dc9d29ce2ff304fb86f9fdc19c6599.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0030_jpg.rf.e9043fa6c8aaaca65b3bcddf790ac8d1.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0033_jpg.rf.1864330f9bca38879e7c23db8a98b43b.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0033_jpg.rf.9e861bd96ac733d7425ac6c80e193996.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0033_jpg.rf.c24b49bae3e3c7e5e59bbae973e2717d.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0037_jpg.rf.3e82d3b34e5a4b3228080a81a408cd77.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0037_jpg.rf.9e320be7fd15adf12e8a814af3301b21.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0037_jpg.rf.d3d1246f7d26ad658406a1df2bd9a076.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0046_jpg.rf.02ab5a76ec535bc4327d13a9aa2424f0.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0046_jpg.rf.4609c34b088e01e2a4043b53b244cfe7.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0046_jpg.rf.9af37f6b8a5577ae282dbd3b1f768ef5.jpg: 1 duplicate labels removed
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 8578, len(boxes) = 13216. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
val: Scanning /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/valid/labels.cache... 535 images, 0 backgrounds, 0 corrupt: 100%|██████████| 535/535 [00:00<?, ?it/s]
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 271, len(boxes) = 601. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
Plotting labels to runs/detect/train42/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0005' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005859375000000001), 173 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train42
Starting training for 30 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/30      16.4G     0.4584     0.8128      1.008         44        640: 100%|██████████| 513/513 [02:59<00:00,  2.86it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.71it/s]
                   all        535        601      0.812      0.847      0.852       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/30      16.6G     0.4501      0.755      1.001         37        640: 100%|██████████| 513/513 [02:52<00:00,  2.97it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.72it/s]
                   all        535        601      0.761      0.891      0.833      0.698

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/30      16.5G     0.4827     0.7965      1.013         43        640: 100%|██████████| 513/513 [02:52<00:00,  2.98it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.71it/s]
                   all        535        601      0.753      0.878      0.822      0.687

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/30      16.6G     0.4671     0.7767      1.008         40        640: 100%|██████████| 513/513 [02:50<00:00,  3.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.72it/s]
                   all        535        601        0.8      0.869      0.858       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/30      16.6G     0.4583     0.7666      1.001         43        640: 100%|██████████| 513/513 [02:50<00:00,  3.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.70it/s]
                   all        535        601      0.805      0.852       0.85      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/30      16.6G     0.4517     0.7507     0.9983         43        640: 100%|██████████| 513/513 [02:50<00:00,  3.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.71it/s]
                   all        535        601      0.777      0.861      0.826      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/30      16.6G      0.448     0.7486      1.001         45        640: 100%|██████████| 513/513 [02:50<00:00,  3.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.71it/s]
                   all        535        601      0.743      0.816      0.785      0.665

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/30      16.6G     0.4418     0.7419     0.9964         42        640: 100%|██████████| 513/513 [02:50<00:00,  3.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.68it/s]
                   all        535        601      0.791      0.854      0.833      0.693

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/30      16.6G     0.4481      0.736     0.9992         60        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.73it/s]
                   all        535        601      0.759       0.85       0.81      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/30      16.6G     0.4375     0.7252     0.9923         49        640: 100%|██████████| 513/513 [02:49<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.74it/s]
                   all        535        601        0.8      0.829      0.837      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/30      16.6G     0.4325     0.7136     0.9892         52        640: 100%|██████████| 513/513 [02:50<00:00,  3.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.73it/s]
                   all        535        601      0.769      0.873      0.852       0.69

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/30      16.6G     0.4225     0.7028     0.9865         38        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.75it/s]
                   all        535        601      0.797      0.849      0.832      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/30      16.6G     0.4211     0.7031     0.9872         41        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.74it/s]
                   all        535        601      0.811      0.842       0.84      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/30      16.6G     0.4199     0.7086     0.9855         37        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.72it/s]
                   all        535        601      0.828      0.846      0.834      0.698

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/30      16.6G     0.4205     0.6949     0.9861         48        640: 100%|██████████| 513/513 [02:49<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.75it/s]
                   all        535        601       0.79      0.821      0.812      0.708

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/30      16.6G     0.4079     0.6797     0.9832         39        640: 100%|██████████| 513/513 [02:49<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.76it/s]
                   all        535        601      0.791      0.865      0.839      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/30      16.6G     0.4086      0.683     0.9816         46        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.74it/s]
                   all        535        601      0.769      0.869      0.839       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/30      16.6G     0.4067      0.678      0.982         25        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.74it/s]
                   all        535        601      0.773      0.864      0.855      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/30      16.6G     0.4033     0.6653     0.9789         37        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.75it/s]
                   all        535        601      0.794      0.888      0.849      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/30      16.6G     0.3954     0.6668     0.9774         33        640: 100%|██████████| 513/513 [02:49<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.76it/s]
                   all        535        601      0.805      0.868      0.849      0.712
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/30      16.6G     0.5015     0.6074      1.026         20        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.75it/s]
                   all        535        601      0.781      0.864      0.829      0.711

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/30      16.6G      0.493     0.5959      1.025         23        640: 100%|██████████| 513/513 [02:49<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.75it/s]
                   all        535        601      0.783      0.876      0.842      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/30      16.6G     0.4836     0.5857       1.02         17        640: 100%|██████████| 513/513 [02:48<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.75it/s]
                   all        535        601        0.8      0.869       0.84       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/30      16.6G     0.4752     0.5774      1.016         16        640: 100%|██████████| 513/513 [02:48<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.75it/s]
                   all        535        601      0.801       0.85      0.841      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/30      16.6G     0.4674      0.565      1.015         18        640: 100%|██████████| 513/513 [02:48<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.76it/s]
                   all        535        601      0.816      0.848      0.846      0.719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/30      16.6G     0.4676     0.5568      1.008         26        640: 100%|██████████| 513/513 [02:48<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.76it/s]
                   all        535        601        0.8      0.845      0.836      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/30      16.6G      0.461     0.5563      1.008         17        640: 100%|██████████| 513/513 [02:48<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.76it/s]
                   all        535        601      0.804       0.87      0.851      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/30      16.6G     0.4503     0.5429      1.001         19        640: 100%|██████████| 513/513 [02:48<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.76it/s]
                   all        535        601      0.822      0.848      0.842      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/30      16.6G     0.4467     0.5362      1.001         18        640: 100%|██████████| 513/513 [02:48<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.76it/s]
                   all        535        601      0.804      0.864      0.845      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/30      16.6G     0.4446     0.5329          1         13        640: 100%|██████████| 513/513 [02:49<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.76it/s]
                   all        535        601      0.799      0.866      0.843      0.722

30 epochs completed in 1.452 hours.
Optimizer stripped from runs/detect/train42/weights/last.pt, 51.2MB
Optimizer stripped from runs/detect/train42/weights/best.pt, 51.2MB

Validating runs/detect/train42/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11l summary (fused): 464 layers, 25,281,625 parameters, 0 gradients, 86.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:03<00:00,  3.52it/s]
                   all        535        601      0.804       0.87      0.851      0.724
                 human        142        185      0.839      0.957      0.946      0.821
             no-object        327        327      0.991       0.98      0.994      0.994
      undefined-object         66         89      0.582      0.674      0.613      0.355
Speed: 0.1ms preprocess, 3.8ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs/detect/train42
Phase 3 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11l.pt, data=data.yaml, epochs=10, time=None, patience=5, batch=25, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train422, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=True, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train422

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  2    173824  ultralytics.nn.modules.block.C3k2            [128, 256, 2, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  2    691712  ultralytics.nn.modules.block.C3k2            [256, 512, 2, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  2   2234368  ultralytics.nn.modules.block.C3k2            [512, 512, 2, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  2   1455616  ultralytics.nn.modules.block.C2PSA           [512, 512, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2    756736  ultralytics.nn.modules.block.C3k2            [1024, 256, 2, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   2365440  ultralytics.nn.modules.block.C3k2            [768, 512, 2, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   2496512  ultralytics.nn.modules.block.C3k2            [1024, 512, 2, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, [256, 512, 512]]          
YOLO11l summary: 631 layers, 25,312,793 parameters, 25,312,777 gradients, 87.3 GFLOPs

Transferred 1015/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/labels.cache... 12819 images, 1832 backgrounds, 0 corrupt: 100%|██████████| 12819/12819 [00:00<?, ?it/s]
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0002_jpg.rf.573a42f8536acbbec0af38bfe5765558.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0002_jpg.rf.62a16c681fbe140845363a73b8e533ab.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0002_jpg.rf.df975580c65051afaf8b500231ba2c53.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0008_jpg.rf.5fd743739143b274c1e1956c6b60a2ba.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0008_jpg.rf.7993d302e9b33a034d4d40b39a08de8b.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0008_jpg.rf.f2307ebe8d702f1f7883911116a76793.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0020_jpg.rf.52095f5084038bd9ab6221e38b3beaf5.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0020_jpg.rf.8fa7be8ff0fecffb2bb080cf812fc803.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0020_jpg.rf.af904754c4a13eeec209b56a76dfd850.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0022_jpg.rf.8fd311f2649e576153b3843c234ae5d2.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0022_jpg.rf.d8efff0597f0648c603106c996360d2c.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0030_jpg.rf.3bf786992a099b6e2963194a85dd9975.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0030_jpg.rf.d8dc9d29ce2ff304fb86f9fdc19c6599.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0030_jpg.rf.e9043fa6c8aaaca65b3bcddf790ac8d1.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0033_jpg.rf.1864330f9bca38879e7c23db8a98b43b.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0033_jpg.rf.9e861bd96ac733d7425ac6c80e193996.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0033_jpg.rf.c24b49bae3e3c7e5e59bbae973e2717d.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0037_jpg.rf.3e82d3b34e5a4b3228080a81a408cd77.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0037_jpg.rf.9e320be7fd15adf12e8a814af3301b21.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0037_jpg.rf.d3d1246f7d26ad658406a1df2bd9a076.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0046_jpg.rf.02ab5a76ec535bc4327d13a9aa2424f0.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0046_jpg.rf.4609c34b088e01e2a4043b53b244cfe7.jpg: 1 duplicate labels removed
train: WARNING ⚠️ /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train/images/video_001_mp4-0046_jpg.rf.9af37f6b8a5577ae282dbd3b1f768ef5.jpg: 1 duplicate labels removed
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 8578, len(boxes) = 13216. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
val: Scanning /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/valid/labels.cache... 535 images, 0 backgrounds, 0 corrupt: 100%|██████████| 535/535 [00:00<?, ?it/s]
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 271, len(boxes) = 601. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
Plotting labels to runs/detect/train422/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005859375000000001), 173 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train422
Starting training for 10 epochs...
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/10      16.4G     0.4667      0.562      1.012         19        640: 100%|██████████| 513/513 [02:58<00:00,  2.87it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.73it/s]
                   all        535        601       0.84      0.806      0.844      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/10      16.6G      0.496     0.5951      1.024         15        640: 100%|██████████| 513/513 [02:52<00:00,  2.98it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.72it/s]
                   all        535        601      0.809      0.857      0.844      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/10      16.5G     0.5165     0.6295      1.036         20        640: 100%|██████████| 513/513 [02:51<00:00,  2.98it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.72it/s]
                   all        535        601      0.798      0.852      0.869      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/10      16.6G     0.5231     0.6337      1.038         19        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.71it/s]
                   all        535        601      0.838      0.808      0.826      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/10      16.6G     0.5063     0.6108      1.032         15        640: 100%|██████████| 513/513 [02:49<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.73it/s]
                   all        535        601      0.801      0.841      0.818      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/10      16.6G      0.492     0.5922      1.022         22        640: 100%|██████████| 513/513 [02:49<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.73it/s]
                   all        535        601      0.761      0.882      0.811      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/10      16.6G     0.4769     0.5735      1.014         19        640: 100%|██████████| 513/513 [02:49<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.74it/s]
                   all        535        601      0.772      0.823      0.801      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/10      16.6G     0.4595     0.5547      1.004         20        640: 100%|██████████| 513/513 [02:49<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  3.74it/s]
                   all        535        601      0.805      0.831      0.812      0.693
EarlyStopping: Training stopped early as no improvement observed in last 5 epochs. Best results observed at epoch 3, best model saved as best.pt.
To update EarlyStopping(patience=5) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

8 epochs completed in 0.390 hours.
Optimizer stripped from runs/detect/train422/weights/last.pt, 51.2MB
Optimizer stripped from runs/detect/train422/weights/best.pt, 51.2MB

Validating runs/detect/train422/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11l summary (fused): 464 layers, 25,281,625 parameters, 0 gradients, 86.6 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:03<00:00,  3.51it/s]
                   all        535        601      0.798      0.852      0.869      0.725
                 human        142        185      0.844      0.946      0.956      0.821
             no-object        327        327      0.984      0.969      0.991      0.991
      undefined-object         66         89      0.565       0.64      0.659      0.362
Speed: 0.1ms preprocess, 3.8ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to runs/detect/train422
done
naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-detection-Yolo11$ 
