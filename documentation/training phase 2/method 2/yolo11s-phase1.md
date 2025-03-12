naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-detection-Yolo11$ python3 train.py 
Phase 1 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11s.pt, data=data.yaml, epochs=20, time=None, patience=100, batch=8, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train5, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train5
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]    
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    443776  ultralytics.nn.modules.block.C3k2            [768, 256, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    127680  ultralytics.nn.modules.block.C3k2            [512, 128, 1, False]          
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1    345472  ultralytics.nn.modules.block.C3k2            [384, 256, 1, False]          
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 23        [16, 19, 22]  1    820569  ultralytics.nn.modules.head.Detect           [3, [128, 256, 512]]          
YOLO11s summary: 319 layers, 9,428,953 parameters, 9,428,937 gradients, 21.6 GFLOPs

Transferred 493/499 items from pretrained weights
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
Plotting labels to runs/detect/train5/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train5
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/20      2.25G     0.6529      1.309      1.125          6        640: 100%|██████████| 1603/1603 [01:48<00:00, 14.81it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 17.94it/s]
                   all        535        601      0.713      0.803      0.814      0.642

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/20      2.33G     0.6099      1.054       1.09          5        640: 100%|██████████| 1603/1603 [01:41<00:00, 15.74it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 18.55it/s]
                   all        535        601      0.762      0.853       0.83      0.648

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/20      2.34G     0.5853     0.9994      1.066          9        640: 100%|██████████| 1603/1603 [01:38<00:00, 16.22it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 18.91it/s]
                   all        535        601        0.7      0.784      0.783      0.644

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/20      2.34G     0.5514     0.9493       1.05         11        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 18.82it/s]
                   all        535        601      0.773      0.875      0.844      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/20      2.33G     0.5269     0.9069      1.034          9        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.12it/s]
                   all        535        601      0.791      0.844      0.865      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/20      2.34G     0.5134     0.8689      1.024          8        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.11it/s]
                   all        535        601      0.848       0.85      0.879      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/20      2.34G     0.4962      0.846      1.016          9        640: 100%|██████████| 1603/1603 [01:35<00:00, 16.73it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.37it/s]
                   all        535        601      0.805      0.877      0.854      0.714

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/20      2.35G     0.4866     0.8189      1.011          5        640: 100%|██████████| 1603/1603 [01:35<00:00, 16.71it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.34it/s]
                   all        535        601      0.785        0.9      0.872      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/20      2.35G     0.4846     0.8106      1.009         11        640: 100%|██████████| 1603/1603 [01:35<00:00, 16.74it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.06it/s]
                   all        535        601      0.809       0.85      0.848      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/20      2.34G     0.4705     0.7954      1.004          8        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.29it/s]
                   all        535        601      0.839      0.859        0.9      0.731
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/20      2.34G     0.5713     0.7325      1.067          2        640: 100%|██████████| 1603/1603 [01:37<00:00, 16.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.59it/s]
                   all        535        601      0.815      0.847      0.855       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/20      2.34G     0.5551      0.707      1.061          3        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.59it/s]
                   all        535        601      0.779      0.894       0.87      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/20      2.33G     0.5459     0.6838      1.051          2        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.60it/s]
                   all        535        601      0.796      0.874      0.859      0.711

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/20      2.35G     0.5365     0.6698      1.047          3        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.82it/s]
                   all        535        601       0.79      0.863       0.83      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/20      2.34G     0.5227     0.6589      1.034          3        640: 100%|██████████| 1603/1603 [01:35<00:00, 16.76it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.75it/s]
                   all        535        601      0.791       0.88      0.855      0.731

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/20      2.35G       0.51     0.6306      1.026          1        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.76it/s]
                   all        535        601      0.822      0.857      0.861      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/20      2.34G     0.5004     0.6195      1.021          3        640: 100%|██████████| 1603/1603 [01:35<00:00, 16.71it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.73it/s]
                   all        535        601      0.784      0.891      0.871       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/20      2.34G     0.4932     0.6112      1.021          2        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.73it/s]
                   all        535        601      0.839      0.845      0.863      0.734

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/20      2.34G     0.4814     0.5946      1.014          5        640: 100%|██████████| 1603/1603 [01:35<00:00, 16.74it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.57it/s]
                   all        535        601      0.797      0.884       0.85      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/20      2.34G     0.4741     0.5874      1.012          3        640: 100%|██████████| 1603/1603 [01:36<00:00, 16.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 19.23it/s]
                   all        535        601      0.797      0.899      0.851      0.728

20 epochs completed in 0.553 hours.
Optimizer stripped from runs/detect/train5/weights/last.pt, 19.2MB
Optimizer stripped from runs/detect/train5/weights/best.pt, 19.2MB

Validating runs/detect/train5/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11s summary (fused): 238 layers, 9,413,961 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:01<00:00, 18.22it/s]
                   all        535        601      0.848      0.848      0.879      0.735
                 human        142        185       0.89      0.916      0.958      0.792
             no-object        327        327      0.965      0.982      0.992      0.992
      undefined-object         66         89      0.689      0.647      0.686      0.419
Speed: 0.1ms preprocess, 1.2ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to runs/detect/train5
Phase 2 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11s.pt, data=data.yaml, epochs=30, time=None, patience=100, batch=25, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train52, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0005, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train52

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]    
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    443776  ultralytics.nn.modules.block.C3k2            [768, 256, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    127680  ultralytics.nn.modules.block.C3k2            [512, 128, 1, False]          
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1    345472  ultralytics.nn.modules.block.C3k2            [384, 256, 1, False]          
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 23        [16, 19, 22]  1    820569  ultralytics.nn.modules.head.Detect           [3, [128, 256, 512]]          
YOLO11s summary: 319 layers, 9,428,953 parameters, 9,428,937 gradients, 21.6 GFLOPs

Transferred 499/499 items from pretrained weights
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
Plotting labels to runs/detect/train52/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0005' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005859375000000001), 87 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train52
Starting training for 30 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/30      6.47G     0.4609     0.7862      1.001         44        640: 100%|██████████| 513/513 [01:18<00:00,  6.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.84it/s]
                   all        535        601       0.83       0.88       0.87      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/30      6.53G     0.4712     0.7857      1.006         37        640: 100%|██████████| 513/513 [01:18<00:00,  6.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.96it/s]
                   all        535        601      0.767      0.835      0.825      0.684

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/30      6.51G     0.5013      0.823      1.016         43        640: 100%|██████████| 513/513 [01:18<00:00,  6.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.96it/s]
                   all        535        601      0.688      0.871      0.795      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/30      6.53G     0.4806     0.7975      1.008         40        640: 100%|██████████| 513/513 [01:18<00:00,  6.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.00it/s]
                   all        535        601      0.732      0.869      0.794      0.674

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/30      6.53G     0.4691     0.7782          1         43        640: 100%|██████████| 513/513 [01:18<00:00,  6.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.04it/s]
                   all        535        601       0.82      0.857      0.849      0.708

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/30      6.53G     0.4593     0.7577      0.995         43        640: 100%|██████████| 513/513 [01:18<00:00,  6.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.07it/s]
                   all        535        601      0.817      0.841      0.849      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/30      6.52G     0.4557     0.7515     0.9964         45        640: 100%|██████████| 513/513 [01:19<00:00,  6.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.98it/s]
                   all        535        601      0.821       0.88       0.86      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/30      6.52G     0.4485     0.7445     0.9912         42        640: 100%|██████████| 513/513 [01:17<00:00,  6.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.05it/s]
                   all        535        601      0.806      0.843       0.85      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/30      6.53G     0.4538     0.7406     0.9938         60        640: 100%|██████████| 513/513 [01:17<00:00,  6.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.11it/s]
                   all        535        601      0.805      0.824       0.83      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/30      6.52G     0.4492     0.7385     0.9917         49        640: 100%|██████████| 513/513 [01:17<00:00,  6.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.16it/s]
                   all        535        601      0.794      0.838      0.849      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/30      6.53G     0.4391     0.7189     0.9856         52        640: 100%|██████████| 513/513 [01:17<00:00,  6.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.05it/s]
                   all        535        601      0.803      0.851      0.836      0.708

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/30      6.53G     0.4289     0.7094     0.9829         38        640: 100%|██████████| 513/513 [01:17<00:00,  6.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.19it/s]
                   all        535        601      0.831      0.818      0.847      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/30      6.53G     0.4242     0.7065     0.9821         41        640: 100%|██████████| 513/513 [01:17<00:00,  6.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.17it/s]
                   all        535        601      0.781      0.889      0.853      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/30      6.52G     0.4256     0.7046     0.9801         37        640: 100%|██████████| 513/513 [01:17<00:00,  6.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.06it/s]
                   all        535        601      0.821       0.83      0.862       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/30      6.52G     0.4217     0.6924     0.9796         48        640: 100%|██████████| 513/513 [01:17<00:00,  6.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.06it/s]
                   all        535        601      0.805      0.815      0.831      0.714

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/30      6.52G      0.409     0.6779     0.9765         39        640: 100%|██████████| 513/513 [01:21<00:00,  6.29it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.19it/s]
                   all        535        601      0.799      0.854      0.848      0.731

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/30      6.53G     0.4131      0.683     0.9756         46        640: 100%|██████████| 513/513 [01:18<00:00,  6.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.12it/s]
                   all        535        601      0.789      0.827      0.826      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/30      6.53G     0.4108     0.6787     0.9766         25        640: 100%|██████████| 513/513 [01:18<00:00,  6.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.09it/s]
                   all        535        601        0.8      0.854      0.857      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/30      6.53G     0.4083     0.6688     0.9749         37        640: 100%|██████████| 513/513 [01:18<00:00,  6.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.10it/s]
                   all        535        601      0.775      0.899      0.842      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/30      6.52G     0.3965     0.6654     0.9717         33        640: 100%|██████████| 513/513 [01:17<00:00,  6.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.11it/s]
                   all        535        601      0.801       0.85      0.847      0.723
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/30      6.53G        0.5     0.6086      1.022         20        640: 100%|██████████| 513/513 [01:10<00:00,  7.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.01it/s]
                   all        535        601      0.812       0.85      0.829      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/30      6.52G     0.4976     0.5986      1.021         23        640: 100%|██████████| 513/513 [01:11<00:00,  7.22it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.07it/s]
                   all        535        601      0.796      0.843      0.837       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/30      6.52G     0.4867     0.5875      1.017         17        640: 100%|██████████| 513/513 [01:10<00:00,  7.32it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.13it/s]
                   all        535        601      0.793      0.855      0.827      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/30      6.52G     0.4798      0.578      1.011         16        640: 100%|██████████| 513/513 [01:10<00:00,  7.23it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.06it/s]
                   all        535        601      0.794      0.868      0.831      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/30      6.52G     0.4733     0.5677       1.01         18        640: 100%|██████████| 513/513 [01:10<00:00,  7.26it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.12it/s]
                   all        535        601      0.769      0.865      0.841       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/30      6.52G     0.4719      0.561      1.003         26        640: 100%|██████████| 513/513 [01:09<00:00,  7.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.06it/s]
                   all        535        601      0.823      0.825       0.84      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/30      6.53G      0.461     0.5552      1.001         17        640: 100%|██████████| 513/513 [01:09<00:00,  7.33it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.99it/s]
                   all        535        601      0.768      0.861       0.83      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/30      6.52G     0.4547     0.5453     0.9969         19        640: 100%|██████████| 513/513 [01:09<00:00,  7.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.05it/s]
                   all        535        601      0.831      0.806      0.832      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/30      6.53G     0.4487     0.5346     0.9963         18        640: 100%|██████████| 513/513 [01:09<00:00,  7.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.09it/s]
                   all        535        601      0.832       0.81      0.831      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/30      6.52G      0.448     0.5355     0.9958         13        640: 100%|██████████| 513/513 [01:10<00:00,  7.31it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  7.07it/s]
                   all        535        601      0.824      0.813      0.833      0.726

30 epochs completed in 0.647 hours.
Optimizer stripped from runs/detect/train52/weights/last.pt, 19.2MB
Optimizer stripped from runs/detect/train52/weights/best.pt, 19.2MB

Validating runs/detect/train52/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11s summary (fused): 238 layers, 9,413,961 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  5.56it/s]
                   all        535        601      0.821       0.83      0.862       0.74
                 human        142        185      0.909      0.946      0.953      0.822
             no-object        327        327      0.817      0.994      0.994      0.994
      undefined-object         66         89      0.736      0.551      0.639      0.404
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs/detect/train52
Phase 3 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11s.pt, data=data.yaml, epochs=10, time=None, patience=5, batch=25, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train522, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=True, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train522

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]    
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    443776  ultralytics.nn.modules.block.C3k2            [768, 256, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    127680  ultralytics.nn.modules.block.C3k2            [512, 128, 1, False]          
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1    345472  ultralytics.nn.modules.block.C3k2            [384, 256, 1, False]          
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 23        [16, 19, 22]  1    820569  ultralytics.nn.modules.head.Detect           [3, [128, 256, 512]]          
YOLO11s summary: 319 layers, 9,428,953 parameters, 9,428,937 gradients, 21.6 GFLOPs

Transferred 499/499 items from pretrained weights
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
Plotting labels to runs/detect/train522/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005859375000000001), 87 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train522
Starting training for 10 epochs...
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/10       6.5G     0.5115     0.6505      1.034         19        640: 100%|██████████| 513/513 [01:12<00:00,  7.07it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.91it/s]
                   all        535        601      0.822      0.871      0.867      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/10      6.52G     0.5272      0.642      1.037         15        640: 100%|██████████| 513/513 [01:11<00:00,  7.17it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.90it/s]
                   all        535        601       0.76       0.88      0.822      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/10      6.51G     0.5418     0.6736      1.043         20        640: 100%|██████████| 513/513 [01:10<00:00,  7.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.93it/s]
                   all        535        601      0.786      0.831      0.827      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/10      6.52G     0.5393     0.6546      1.041         19        640: 100%|██████████| 513/513 [01:11<00:00,  7.16it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.95it/s]
                   all        535        601      0.809       0.83      0.821      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/10      6.53G     0.5253     0.6339      1.037         15        640: 100%|██████████| 513/513 [01:13<00:00,  6.98it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.93it/s]
                   all        535        601      0.795      0.901      0.853      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/10      6.53G     0.5088     0.6158      1.027         22        640: 100%|██████████| 513/513 [01:13<00:00,  7.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.96it/s]
                   all        535        601      0.786      0.841      0.821        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/10      6.53G     0.4934     0.5944      1.018         19        640: 100%|██████████| 513/513 [01:13<00:00,  6.94it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.92it/s]
                   all        535        601      0.801      0.852      0.827      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/10      6.53G     0.4739      0.578      1.006         20        640: 100%|██████████| 513/513 [01:13<00:00,  6.96it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.99it/s]
                   all        535        601      0.808      0.845      0.827      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/10      6.53G     0.4679     0.5561      1.007         16        640: 100%|██████████| 513/513 [01:13<00:00,  6.97it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.98it/s]
                   all        535        601      0.798      0.862      0.838      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/10      6.53G     0.4563     0.5476     0.9986         21        640: 100%|██████████| 513/513 [01:13<00:00,  7.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  6.98it/s]
                   all        535        601      0.803      0.853      0.837      0.727
EarlyStopping: Training stopped early as no improvement observed in last 5 epochs. Best results observed at epoch 5, best model saved as best.pt.
To update EarlyStopping(patience=5) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

10 epochs completed in 0.208 hours.
Optimizer stripped from runs/detect/train522/weights/last.pt, 19.2MB
Optimizer stripped from runs/detect/train522/weights/best.pt, 19.2MB

Validating runs/detect/train522/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11s summary (fused): 238 layers, 9,413,961 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:01<00:00,  5.62it/s]
                   all        535        601      0.795      0.901      0.853      0.726
                 human        142        185      0.839      0.957      0.945      0.816
             no-object        327        327      0.987      0.982      0.993      0.993
      undefined-object         66         89      0.561      0.764       0.62       0.37
Speed: 0.1ms preprocess, 1.3ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to runs/detect/train522
done
naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-detection-Yolo11$ 
