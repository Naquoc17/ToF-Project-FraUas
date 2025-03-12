naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-detection-Yolo11$ python3 train.py
Phase 1 started ...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11x.pt, data=data.yaml, epochs=20, time=None, patience=100, batch=8, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train
Overriding model.yaml nc=80 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 23        [16, 19, 22]  1   3149017  ultralytics.nn.modules.head.Detect           [3, [384, 768, 768]]          
YOLO11x summary: 631 layers, 56,877,241 parameters, 56,877,225 gradients, 195.5 GFLOPs

Transferred 1009/1015 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt'...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.35M/5.35M [00:00<00:00, 75.5MB/s]
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
Plotting labels to runs/detect/train/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/20      8.66G     0.7609      1.443      1.272          6        640: 100%|██████████| 1603/1603 [06:09<00:00,  4.33it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.08it/s]
                   all        535        601      0.606       0.63      0.612      0.489

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/20      8.84G     0.6866      1.255      1.186         16        640:         2/20      8.84G     0.6856      1.253      1.185         18        640:         2/20      8.84G     0.6856      1.253      1.185         18        640:         2/20      8.84G     0.6867      1.253      1.186         24        640:         2/20      8.84G     0.6867      1.253      1.186         24        640:         2/20      8.84G     0.6857      1.252      1.185         19        640:         2/20      8.84G     0.6857      1.252      1.185         19        640:         2/20      8.84G     0.6846      1.251      1.185         18        640:         2/20      8.84G     0.6846      1.251      1.185         18        640:         2/20      8.84G     0.6838      1.249      1.184         24        640:         2/20      8.84G     0.6838      1.249      1.184         24        640:         2/20      8.84G     0.6826      1.246      1.184         26        640:         2/20      8.84G     0.6826      1.246      1.184         26        640:         2/20      8.84G     0.6832      1.246      1.184         18        640:         2/20      8.84G     0.6832      1.246      1.184         18        640:         2/20      8.84G     0.6851      1.246      1.185         19        640:         2/20      8.84G     0.6851      1.246      1.185         19        640:         2/20      8.84G     0.6864      1.248      1.185         23        640:         2/20        2/20      8.91G     0.6458      1.163      1.156          5        640: 100%|██████████| 1603/1603 [06:02<00:00,  4.42it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.00it/s]
                   all        535        601      0.593      0.696      0.669      0.571

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/20      9.05G     0.6038      1.082      1.114          9        640: 100%|██████████| 1603/1603 [06:00<00:00,  4.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  5.96it/s]
                   all        535        601      0.619       0.68      0.614      0.518

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/20      9.06G     0.5674      1.033      1.092         11        640: 100%|██████████| 1603/1603 [05:58<00:00,  4.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  5.93it/s]
                   all        535        601      0.728      0.829       0.78      0.623

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/20      8.91G     0.5337     0.9734       1.07          9        640: 100%|██████████| 1603/1603 [06:02<00:00,  4.43it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  5.75it/s]
                   all        535        601      0.728      0.825      0.784      0.651

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/20      9.06G     0.5292     0.9298      1.057          8        640: 100%|██████████| 1603/1603 [05:59<00:00,  4.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.04it/s]
                   all        535        601      0.666      0.851      0.729      0.633

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/20      9.04G      0.506     0.9017      1.045          9        640: 100%|██████████| 1603/1603 [05:53<00:00,  4.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.30it/s]
                   all        535        601      0.774      0.845      0.822      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/20      9.09G     0.5017     0.8803       1.04          5        640: 100%|██████████| 1603/1603 [05:57<00:00,  4.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.07it/s]
                   all        535        601      0.752      0.837       0.83       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/20      9.06G     0.4977     0.8625      1.034         11        640: 100%|██████████| 1603/1603 [05:55<00:00,  4.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.02it/s]
                   all        535        601      0.712      0.836      0.781      0.656

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/20      9.06G     0.4793     0.8452      1.025          8        640: 100%|██████████| 1603/1603 [05:53<00:00,  4.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.54it/s]
                   all        535        601      0.724      0.821      0.788      0.663
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/20      9.06G     0.5862     0.7784      1.094          2        640: 100%|██████████| 1603/1603 [05:26<00:00,  4.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.72it/s]
                   all        535        601      0.772      0.837      0.822      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/20      9.06G     0.5647     0.7437      1.083          3        640: 100%|██████████| 1603/1603 [05:20<00:00,  5.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.77it/s]
                   all        535        601      0.787      0.836      0.833      0.688

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/20      8.91G     0.5578     0.7257      1.073          2        640: 100%|██████████| 1603/1603 [05:19<00:00,  5.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.75it/s]
                   all        535        601      0.767      0.836      0.818      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/20      9.06G     0.5493     0.7059      1.069          3        640: 100%|██████████| 1603/1603 [05:19<00:00,  5.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.76it/s]
                   all        535        601      0.778      0.848      0.824      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/20      9.04G     0.5376      0.693      1.058          3        640: 100%|██████████| 1603/1603 [05:19<00:00,  5.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.77it/s]
                   all        535        601      0.777       0.87      0.848      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/20      9.07G     0.5249     0.6674      1.049          1        640: 100%|██████████| 1603/1603 [05:21<00:00,  4.99it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.75it/s]
                   all        535        601      0.786      0.872      0.853      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/20      9.07G     0.5146     0.6541      1.044          3        640: 100%|██████████| 1603/1603 [05:21<00:00,  4.98it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.57it/s]
                   all        535        601      0.797      0.876      0.864      0.719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/20      9.07G     0.5059     0.6408      1.041          2        640: 100%|██████████| 1603/1603 [05:40<00:00,  4.71it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.35it/s]
                   all        535        601      0.787      0.878      0.852      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/20      9.06G     0.4972     0.6225      1.034          5        640: 100%|██████████| 1603/1603 [05:33<00:00,  4.81it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.41it/s]
                   all        535        601       0.83      0.818      0.858       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/20      9.05G      0.487     0.6166       1.03          3        640: 100%|██████████| 1603/1603 [05:39<00:00,  4.72it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.41it/s]
                   all        535        601      0.775      0.884      0.854      0.717

20 epochs completed in 1.947 hours.
Optimizer stripped from runs/detect/train/weights/last.pt, 114.4MB
Optimizer stripped from runs/detect/train/weights/best.pt, 114.4MB

Validating runs/detect/train/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11x summary (fused): 464 layers, 56,830,489 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:05<00:00,  6.51it/s]
                   all        535        601      0.787      0.878      0.853       0.72
                 human        142        185      0.837      0.974      0.941      0.803
             no-object        327        327      0.974      0.985      0.994      0.994
      undefined-object         66         89       0.55      0.674      0.623      0.364
Speed: 0.1ms preprocess, 7.7ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs/detect/train
Phase 2 started ...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11x.pt, data=data.yaml, epochs=30, time=None, patience=100, batch=25, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train2, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0005, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train2

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 23        [16, 19, 22]  1   3149017  ultralytics.nn.modules.head.Detect           [3, [384, 768, 768]]          
YOLO11x summary: 631 layers, 56,877,241 parameters, 56,877,225 gradients, 195.5 GFLOPs

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
Plotting labels to runs/detect/train2/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0005' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005859375000000001), 173 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train2
Starting training for 30 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/30      23.7G     0.4655      0.838      1.016         44        640: 100%|██████████| 513/513 [05:18<00:00,  1.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.11it/s]
                   all        535        601      0.823      0.838      0.851      0.698

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/30        24G     0.4609     0.7874      1.011         37        640: 100%|██████████| 513/513 [05:11<00:00,  1.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.11it/s]
                   all        535        601      0.738      0.823      0.768       0.66

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/30      24.2G     0.4961     0.8309      1.025         43        640: 100%|██████████| 513/513 [05:10<00:00,  1.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.09it/s]
                   all        535        601      0.703      0.842      0.778      0.654

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/30      24.1G     0.4793      0.811      1.019         40        640: 100%|██████████| 513/513 [05:04<00:00,  1.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.04it/s]
                   all        535        601       0.72      0.763      0.745       0.63

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/30      24.1G     0.4689      0.794       1.01         43        640: 100%|██████████| 513/513 [05:08<00:00,  1.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.16it/s]
                   all        535        601       0.78      0.862      0.856      0.704

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/30      24.1G     0.4624     0.7789      1.007         43        640: 100%|██████████| 513/513 [05:07<00:00,  1.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.06it/s]
                   all        535        601       0.78      0.843      0.817      0.689

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/30      24.1G     0.4567     0.7739      1.006         45        640: 100%|██████████| 513/513 [05:13<00:00,  1.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.05it/s]
                   all        535        601      0.786      0.837      0.842      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/30      24.1G     0.4456     0.7566      1.001         42        640: 100%|██████████| 513/513 [05:05<00:00,  1.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.16it/s]
                   all        535        601      0.735      0.838      0.781      0.667

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/30      24.1G     0.4576     0.7578      1.005         60        640: 100%|██████████| 513/513 [04:54<00:00,  1.74it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.16it/s]
                   all        535        601      0.814       0.81      0.818      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/30      24.1G     0.4465     0.7499      1.002         49        640: 100%|██████████| 513/513 [05:03<00:00,  1.69it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.06it/s]
                   all        535        601      0.743      0.858      0.806      0.697

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/30      24.1G     0.4399     0.7373     0.9968         52        640: 100%|██████████| 513/513 [05:09<00:00,  1.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.08it/s]
                   all        535        601      0.802      0.817       0.83      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/30      24.1G     0.4303     0.7237     0.9928         38        640: 100%|██████████| 513/513 [05:09<00:00,  1.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601       0.71      0.871      0.795      0.683

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/30      24.1G     0.4248      0.718     0.9925         41        640: 100%|██████████| 513/513 [05:11<00:00,  1.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.08it/s]
                   all        535        601      0.813      0.828      0.843       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/30      24.1G     0.4255     0.7254     0.9898         37        640: 100%|██████████| 513/513 [04:59<00:00,  1.71it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.18it/s]
                   all        535        601      0.761      0.857      0.827      0.702

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/30      24.1G     0.4228     0.7139     0.9891         48        640: 100%|██████████| 513/513 [04:53<00:00,  1.75it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.17it/s]
                   all        535        601      0.759      0.891      0.828      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/30      24.1G     0.4126     0.6937      0.987         39        640: 100%|██████████| 513/513 [04:58<00:00,  1.72it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601       0.85      0.807      0.831      0.704

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/30      24.1G     0.4129     0.6958     0.9851         46        640: 100%|██████████| 513/513 [04:58<00:00,  1.72it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.17it/s]
                   all        535        601      0.786      0.843      0.837      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/30      24.1G     0.4103     0.6911     0.9848         25        640: 100%|██████████| 513/513 [04:53<00:00,  1.75it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.17it/s]
                   all        535        601      0.777      0.831      0.836      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/30      24.1G     0.4088     0.6795     0.9835         37        640: 100%|██████████| 513/513 [04:53<00:00,  1.75it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601      0.776      0.864      0.849      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/30      24.1G     0.3977     0.6797     0.9805         33        640: 100%|██████████| 513/513 [04:58<00:00,  1.72it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.18it/s]
                   all        535        601      0.796      0.841      0.823      0.702
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/30      24.1G     0.5006     0.6257      1.031         20        640: 100%|██████████| 513/513 [05:01<00:00,  1.70it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601      0.786      0.812      0.816      0.693

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/30      24.1G     0.4957     0.6085      1.029         23        640: 100%|██████████| 513/513 [05:09<00:00,  1.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.08it/s]
                   all        535        601      0.814      0.838      0.825      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/30      24.1G     0.4864     0.5961      1.024         17        640: 100%|██████████| 513/513 [04:55<00:00,  1.73it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.18it/s]
                   all        535        601      0.799       0.83      0.835      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/30      24.1G     0.4827     0.5903       1.02         16        640: 100%|██████████| 513/513 [04:53<00:00,  1.75it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.17it/s]
                   all        535        601      0.806      0.858      0.849      0.731

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/30      24.1G     0.4738     0.5784      1.019         18        640: 100%|██████████| 513/513 [05:00<00:00,  1.71it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601      0.811      0.852      0.854      0.728

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/30      24.2G     0.4732      0.574      1.013         26        640: 100%|██████████| 513/513 [05:02<00:00,  1.70it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.08it/s]
                   all        535        601      0.839      0.852      0.867      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/30      24.1G     0.4633     0.5695      1.011         17        640: 100%|██████████| 513/513 [05:09<00:00,  1.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.09it/s]
                   all        535        601      0.817      0.842      0.858      0.731

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/30      24.2G     0.4586     0.5546      1.007         19        640: 100%|██████████| 513/513 [05:10<00:00,  1.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601      0.806      0.842      0.845      0.724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/30      24.1G     0.4506     0.5469      1.005         18        640: 100%|██████████| 513/513 [04:53<00:00,  1.75it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.19it/s]
                   all        535        601      0.801      0.856      0.846      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/30      24.1G     0.4505     0.5489      1.006         13        640: 100%|██████████| 513/513 [05:07<00:00,  1.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601       0.82      0.849      0.838      0.719

30 epochs completed in 2.589 hours.
Optimizer stripped from runs/detect/train2/weights/last.pt, 114.4MB
Optimizer stripped from runs/detect/train2/weights/best.pt, 114.4MB

Validating runs/detect/train2/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11x summary (fused): 464 layers, 56,830,489 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.18it/s]
                   all        535        601      0.839      0.852      0.867      0.736
                 human        142        185       0.84      0.951      0.958      0.841
             no-object        327        327      0.994      0.987      0.994      0.994
      undefined-object         66         89      0.683      0.618      0.649      0.372
Speed: 0.1ms preprocess, 7.3ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs/detect/train2
Phase 3 started ...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=detect, mode=train, model=yolo11x.pt, data=data.yaml, epochs=10, time=None, patience=5, batch=25, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train22, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=True, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train22

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 23        [16, 19, 22]  1   3149017  ultralytics.nn.modules.head.Detect           [3, [384, 768, 768]]          
YOLO11x summary: 631 layers, 56,877,241 parameters, 56,877,225 gradients, 195.5 GFLOPs

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
Plotting labels to runs/detect/train22/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005859375000000001), 173 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/detect/train22
Starting training for 10 epochs...
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/10      23.7G     0.4745     0.5829      1.018         19        640: 100%|██████████| 513/513 [05:17<00:00,  1.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.10it/s]
                   all        535        601       0.83      0.839      0.858       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/10        24G      0.504     0.6184       1.03         15        640: 100%|██████████| 513/513 [05:14<00:00,  1.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.08it/s]
                   all        535        601      0.817       0.86      0.868      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/10      24.2G     0.5269     0.6483      1.042         20        640: 100%|██████████| 513/513 [05:12<00:00,  1.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601       0.77      0.837      0.825      0.654

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/10      24.1G     0.5275     0.6486      1.044         19        640: 100%|██████████| 513/513 [05:06<00:00,  1.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.20it/s]
                   all        535        601        0.8       0.87      0.846      0.704

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/10      24.1G      0.518     0.6337       1.04         15        640: 100%|██████████| 513/513 [05:04<00:00,  1.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601        0.8      0.865      0.838      0.713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/10      24.1G     0.4982     0.6089       1.03         22        640: 100%|██████████| 513/513 [05:11<00:00,  1.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.07it/s]
                   all        535        601      0.766      0.831      0.798      0.691

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/10      24.1G     0.4832     0.5885      1.021         19        640: 100%|██████████| 513/513 [05:03<00:00,  1.69it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:05<00:00,  2.17it/s]
                   all        535        601      0.766      0.886      0.828      0.708
EarlyStopping: Training stopped early as no improvement observed in last 5 epochs. Best results observed at epoch 2, best model saved as best.pt.
To update EarlyStopping(patience=5) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

7 epochs completed in 0.617 hours.
Optimizer stripped from runs/detect/train22/weights/last.pt, 114.4MB
Optimizer stripped from runs/detect/train22/weights/best.pt, 114.4MB

Validating runs/detect/train22/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11x summary (fused): 464 layers, 56,830,489 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 11/11 [00:04<00:00,  2.28it/s]
                   all        535        601      0.816       0.86      0.869       0.73
                 human        142        185      0.883      0.934      0.966      0.823
             no-object        327        327      0.954      0.994       0.99       0.99
      undefined-object         66         89      0.612      0.652      0.651      0.376
Speed: 0.1ms preprocess, 6.9ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs/detect/train22
done
naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-detection-Yolo11$ 
