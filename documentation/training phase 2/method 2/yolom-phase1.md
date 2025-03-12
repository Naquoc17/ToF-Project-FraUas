naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-detection-Yolo11$ python3 train.py 
Phase 1 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11m.pt, data=data.yaml, epochs=20, time=None, patience=100, batch=8, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train3, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train3
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, [256, 512, 512]]          
YOLO11m summary: 409 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/train
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
val: Scanning /home/naquoc17/TimeOfFlight-Project/Human-detection-Yolo11/valid/l
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 271, len(boxes) = 601. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
Plotting labels to runs/detect/train3/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train3
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/20      4.45G     0.7065      1.328      1.173          6        640: 100%|██████████| 1603/1603 [02:57<00:00,  9.06it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.04it/s]
                   all        535        601      0.636      0.714      0.659      0.551

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/20      4.59G       0.63      1.095       1.11          5        640: 100%|██████████| 1603/1603 [02:51<00:00,  9.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.40it/s]
                   all        535        601      0.619      0.866      0.733      0.601

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/20      4.64G     0.5938      1.021      1.078          9        640: 100%|██████████| 1603/1603 [02:47<00:00,  9.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.29it/s]
                   all        535        601      0.722      0.783       0.77      0.621

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/20      4.63G      0.563      0.981      1.062         11        640: 100%|██████████| 1603/1603 [02:47<00:00,  9.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.49it/s]
                   all        535        601      0.769      0.825      0.821      0.646

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/20      4.59G     0.5329     0.9337      1.042          9        640: 100%|██████████| 1603/1603 [02:46<00:00,  9.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.49it/s]
                   all        535        601      0.746      0.834      0.832       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/20      4.64G     0.5227     0.8968      1.031          8        640: 100%|██████████| 1603/1603 [02:46<00:00,  9.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.58it/s]
                   all        535        601      0.777      0.813       0.83       0.68

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/20      4.64G     0.4993     0.8669      1.022          9        640: 100%|██████████| 1603/1603 [02:48<00:00,  9.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.54it/s]
                   all        535        601      0.797      0.872      0.854        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/20      4.64G     0.4959     0.8507      1.019          5        640: 100%|██████████| 1603/1603 [02:47<00:00,  9.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.60it/s]
                   all        535        601       0.83      0.886      0.898      0.713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/20      4.64G     0.4889     0.8363      1.015         11        640: 100%|██████████| 1603/1603 [02:46<00:00,  9.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.58it/s]
                   all        535        601      0.788      0.847      0.846      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/20      4.63G     0.4724     0.8155      1.007          8        640: 100%|██████████| 1603/1603 [02:44<00:00,  9.72it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.63it/s]
                   all        535        601      0.783      0.851      0.865      0.713
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/20      4.64G     0.5875     0.7596       1.08          2        640: 100%|██████████| 1603/1603 [02:43<00:00,  9.81it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.65it/s]
                   all        535        601       0.77      0.865      0.825      0.674

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/20      4.64G      0.568     0.7274      1.069          3        640: 100%|██████████| 1603/1603 [02:43<00:00,  9.83it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.71it/s]
                   all        535        601      0.773      0.849      0.842      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/20      4.59G     0.5531     0.7077      1.058          2        640: 100%|██████████| 1603/1603 [02:42<00:00,  9.87it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.78it/s]
                   all        535        601      0.803      0.871      0.869      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/20      4.62G     0.5508     0.6914      1.057          3        640: 100%|██████████| 1603/1603 [02:42<00:00,  9.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.78it/s]
                   all        535        601      0.786      0.878      0.851      0.713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/20      4.63G     0.5352     0.6769      1.043          3        640: 100%|██████████| 1603/1603 [02:42<00:00,  9.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.79it/s]
                   all        535        601      0.752      0.889      0.824      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/20      4.64G     0.5254     0.6555      1.037          1        640: 100%|██████████| 1603/1603 [02:42<00:00,  9.86it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.67it/s]
                   all        535        601      0.796      0.903      0.874      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/20      4.63G     0.5118     0.6347      1.032          3        640: 100%|██████████| 1603/1603 [02:42<00:00,  9.87it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.66it/s]
                   all        535        601      0.834      0.862       0.87      0.719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/20      4.64G     0.5067     0.6237       1.03          2        640: 100%|██████████| 1603/1603 [02:40<00:00,  9.96it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.70it/s]
                   all        535        601      0.828      0.854      0.862      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/20      4.63G      0.494     0.6099      1.023          5        640: 100%|██████████| 1603/1603 [02:43<00:00,  9.80it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.67it/s]
                   all        535        601      0.823      0.884      0.859       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/20      4.64G     0.4879     0.6021       1.02          3        640: 100%|██████████| 1603/1603 [02:43<00:00,  9.81it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.74it/s]
                   all        535        601      0.798      0.889      0.855      0.719

20 epochs completed in 0.940 hours.
Optimizer stripped from runs/detect/train3/weights/last.pt, 40.5MB
Optimizer stripped from runs/detect/train3/weights/best.pt, 40.5MB

Validating runs/detect/train3/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11m summary (fused): 303 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:02<00:00, 12.39it/s]
                   all        535        601      0.796      0.903      0.874      0.725
                 human        142        185      0.812      0.962      0.947      0.812
             no-object        327        327      0.986      0.982      0.994      0.994
      undefined-object         66         89       0.59      0.764       0.68      0.369
Speed: 0.1ms preprocess, 3.0ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs/detect/train3
Phase 2 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11m.pt, data=data.yaml, epochs=30, time=None, patience=100, batch=25, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train32, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0005, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train32

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, [256, 512, 512]]          
YOLO11m summary: 409 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 649/649 items from pretrained weights
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
Plotting labels to runs/detect/train32/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0005' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005859375000000001), 112 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train32
Starting training for 30 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/30        13G     0.4656     0.8175      1.003         44        640: 100%|██████████| 513/513 [02:19<00:00,  3.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.44it/s]
                   all        535        601      0.834      0.849      0.862      0.655

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/30      13.2G     0.4553     0.7638     0.9967         37        640: 100%|██████████| 513/513 [02:14<00:00,  3.82it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.40it/s]
                   all        535        601      0.694      0.815      0.835      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/30      13.1G     0.4887     0.8053      1.007         43        640: 100%|██████████| 513/513 [02:13<00:00,  3.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.44it/s]
                   all        535        601       0.79      0.834      0.831      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/30      13.1G     0.4748     0.7878      1.005         40        640: 100%|██████████| 513/513 [02:12<00:00,  3.87it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.42it/s]
                   all        535        601      0.847      0.813      0.849      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/30      13.1G     0.4651     0.7701     0.9986         43        640: 100%|██████████| 513/513 [02:12<00:00,  3.88it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.42it/s]
                   all        535        601      0.791      0.888       0.85      0.688

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/30      13.1G     0.4542     0.7554     0.9928         43        640: 100%|██████████| 513/513 [02:12<00:00,  3.87it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.42it/s]
                   all        535        601      0.825      0.812      0.846      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/30      13.1G     0.4489     0.7527     0.9947         45        640: 100%|██████████| 513/513 [02:12<00:00,  3.88it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.43it/s]
                   all        535        601      0.824      0.852      0.863      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/30      13.1G     0.4409     0.7419     0.9894         42        640: 100%|██████████| 513/513 [02:12<00:00,  3.88it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.41it/s]
                   all        535        601      0.787      0.856      0.829      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/30      13.1G     0.4488     0.7352      0.992         60        640: 100%|██████████| 513/513 [02:11<00:00,  3.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.44it/s]
                   all        535        601      0.819      0.798      0.837      0.708

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/30      13.1G     0.4412     0.7311     0.9883         49        640: 100%|██████████| 513/513 [02:11<00:00,  3.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.43it/s]
                   all        535        601      0.798      0.863      0.867      0.734

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/30      13.1G     0.4371     0.7176     0.9853         52        640: 100%|██████████| 513/513 [02:11<00:00,  3.90it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.45it/s]
                   all        535        601      0.826      0.823      0.853      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/30      13.1G      0.426     0.7077     0.9831         38        640: 100%|██████████| 513/513 [02:10<00:00,  3.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.48it/s]
                   all        535        601      0.822      0.862      0.855       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/30      13.1G     0.4224     0.7005     0.9816         41        640: 100%|██████████| 513/513 [02:10<00:00,  3.93it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.50it/s]
                   all        535        601      0.837      0.833      0.868       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/30      13.1G     0.4253     0.7069     0.9798         37        640: 100%|██████████| 513/513 [02:11<00:00,  3.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.47it/s]
                   all        535        601      0.855      0.841      0.865       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/30      13.1G     0.4184     0.6966     0.9797         48        640: 100%|██████████| 513/513 [02:11<00:00,  3.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.48it/s]
                   all        535        601      0.803      0.847      0.843      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/30      13.1G     0.4069     0.6811     0.9771         39        640: 100%|██████████| 513/513 [02:11<00:00,  3.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.48it/s]
                   all        535        601      0.799      0.903      0.854      0.722

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/30      13.1G     0.4127     0.6869     0.9758         46        640: 100%|██████████| 513/513 [02:11<00:00,  3.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.48it/s]
                   all        535        601      0.804      0.867      0.844       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/30      13.1G     0.4075     0.6838     0.9761         25        640: 100%|██████████| 513/513 [02:10<00:00,  3.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.49it/s]
                   all        535        601      0.799       0.87      0.847       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/30      13.1G     0.4059     0.6703     0.9746         37        640: 100%|██████████| 513/513 [02:10<00:00,  3.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.48it/s]
                   all        535        601      0.841      0.832      0.854      0.725

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/30      13.1G     0.3939     0.6671     0.9699         33        640: 100%|██████████| 513/513 [02:11<00:00,  3.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.47it/s]
                   all        535        601      0.814      0.845      0.848       0.72
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/30      13.1G     0.4977     0.6083      1.021         20        640: 100%|██████████| 513/513 [02:11<00:00,  3.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.47it/s]
                   all        535        601      0.791      0.851      0.836      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/30      13.1G     0.4974      0.597      1.021         23        640: 100%|██████████| 513/513 [02:11<00:00,  3.90it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.49it/s]
                   all        535        601      0.815      0.827      0.829      0.704

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/30      13.1G     0.4836     0.5859      1.016         17        640: 100%|██████████| 513/513 [02:11<00:00,  3.90it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.48it/s]
                   all        535        601      0.839      0.807      0.831       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/30      13.1G     0.4774     0.5788       1.01         16        640: 100%|██████████| 513/513 [02:10<00:00,  3.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.49it/s]
                   all        535        601       0.77      0.868      0.841      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/30      13.1G     0.4718      0.569      1.011         18        640: 100%|██████████| 513/513 [02:10<00:00,  3.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.49it/s]
                   all        535        601        0.8      0.818      0.839      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/30      13.1G     0.4699     0.5644      1.003         26        640: 100%|██████████| 513/513 [02:10<00:00,  3.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.48it/s]
                   all        535        601      0.821      0.816      0.837      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/30      13.1G     0.4602     0.5576      1.002         17        640: 100%|██████████| 513/513 [02:10<00:00,  3.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.48it/s]
                   all        535        601      0.794      0.858      0.841      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/30      13.1G     0.4557     0.5476     0.9983         19        640: 100%|██████████| 513/513 [02:11<00:00,  3.90it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.49it/s]
                   all        535        601      0.829      0.825      0.839       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/30      13.1G       0.45     0.5392     0.9981         18        640: 100%|██████████| 513/513 [02:11<00:00,  3.90it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.49it/s]
                   all        535        601      0.835      0.818      0.837      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/30      13.1G     0.4493     0.5401      0.998         13        640: 100%|██████████| 513/513 [02:11<00:00,  3.90it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.50it/s]
                   all        535        601      0.839      0.809      0.833      0.727

30 epochs completed in 1.128 hours.
Optimizer stripped from runs/detect/train32/weights/last.pt, 40.5MB
Optimizer stripped from runs/detect/train32/weights/best.pt, 40.5MB

Validating runs/detect/train32/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11m summary (fused): 303 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.05it/s]
                   all        535        601      0.797      0.864      0.867      0.733
                 human        142        185      0.889      0.906      0.956      0.812
             no-object        327        327      0.973      0.966      0.992      0.992
      undefined-object         66         89      0.531      0.719      0.653      0.395
Speed: 0.1ms preprocess, 2.9ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to runs/detect/train32
Phase 3 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11m.pt, data=data.yaml, epochs=10, time=None, patience=5, batch=25, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train322, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=True, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train322

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1413337  ultralytics.nn.modules.head.Detect           [3, [256, 512, 512]]          
YOLO11m summary: 409 layers, 20,055,321 parameters, 20,055,305 gradients, 68.2 GFLOPs

Transferred 649/649 items from pretrained weights
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
Plotting labels to runs/detect/train322/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005859375000000001), 112 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train322
Starting training for 10 epochs...
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/10        13G     0.5168      0.658      1.038         19        640: 100%|██████████| 513/513 [02:19<00:00,  3.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.47it/s]
                   all        535        601      0.802      0.873      0.858      0.711

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/10      13.2G     0.5315     0.6484      1.041         15        640: 100%|██████████| 513/513 [02:13<00:00,  3.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.45it/s]
                   all        535        601      0.789      0.887      0.866      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/10      13.1G     0.5505     0.6792      1.051         20        640: 100%|██████████| 513/513 [02:13<00:00,  3.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.44it/s]
                   all        535        601       0.78      0.825      0.832      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/10      13.1G     0.5433     0.6655      1.047         19        640: 100%|██████████| 513/513 [02:12<00:00,  3.88it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.44it/s]
                   all        535        601       0.79      0.868      0.845      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/10      13.1G     0.5323     0.6446      1.043         15        640: 100%|██████████| 513/513 [02:12<00:00,  3.88it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.44it/s]
                   all        535        601      0.761      0.887      0.823      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/10      13.1G     0.5122     0.6197      1.031         22        640: 100%|██████████| 513/513 [02:12<00:00,  3.88it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.44it/s]
                   all        535        601       0.82      0.849      0.842      0.707
EarlyStopping: Training stopped early as no improvement observed in last 5 epochs. Best results observed at epoch 1, best model saved as best.pt.
To update EarlyStopping(patience=5) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

6 epochs completed in 0.229 hours.
Optimizer stripped from runs/detect/train322/weights/last.pt, 40.5MB
Optimizer stripped from runs/detect/train322/weights/best.pt, 40.5MB

Validating runs/detect/train322/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11m summary (fused): 303 layers, 20,032,345 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:02<00:00,  4.05it/s]
                   all        535        601      0.802      0.872      0.858       0.71
                 human        142        185      0.829      0.969      0.933      0.778
             no-object        327        327      0.976      0.984      0.993      0.993
      undefined-object         66         89      0.602      0.663       0.65      0.361
Speed: 0.1ms preprocess, 3.0ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to runs/detect/train322
done
naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-detection-Yolo11$ 
