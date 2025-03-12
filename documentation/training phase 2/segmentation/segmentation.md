naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-instance-segmentation-Yolo11$ python3 train_custom.py 
Phase 1 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=segment, mode=train, model=yolo11x-seg.pt, data=data.yaml, epochs=20, time=None, patience=100, batch=8, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/segment/train
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
 23        [16, 19, 22]  1   8325497  ultralytics.nn.modules.head.Segment          [3, 32, 384, [384, 768, 768]] 
YOLO11x-seg summary: 667 layers, 62,053,721 parameters, 62,053,705 gradients, 319.7 GFLOPs

Transferred 1071/1077 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /home/naquoc17/TimeOfFlight-Project/Human-instance-segmentation-Yolo11/data/train/labels.cache... 12861 images, 1840 backgrounds, 0 corrupt: 100%|██████████| 12861/12861 [00:00<?, ?it/s]
val: Scanning /home/naquoc17/TimeOfFlight-Project/Human-instance-segmentation-Yolo11/data/valid/labels.cache... 535 images, 0 backgrounds, 0 corrupt: 100%|██████████| 535/535 [00:00<?, ?it/s]
Plotting labels to runs/segment/train/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 176 weight(decay=0.0), 187 weight(decay=0.0005), 186 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/segment/train
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       1/20      10.2G     0.7562      1.111      1.379      1.216         15        640: 100%|██████████| 1608/1608 [07:54<00:00,  3.39it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.54it/s]
                   all        535        601      0.531      0.781      0.658      0.525      0.532      0.781      0.663      0.539

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       2/20      10.2G     0.6427     0.9611      1.115      1.122         14        640: 100%|██████████| 1608/1608 [07:47<00:00,  3.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.55it/s]
                   all        535        601      0.656      0.736      0.736      0.587      0.652      0.732      0.734       0.59

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       3/20      10.2G     0.5956     0.9095      1.042      1.094         16        640: 100%|██████████| 1608/1608 [07:31<00:00,  3.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.56it/s]
                   all        535        601      0.645      0.806      0.695      0.585       0.64      0.804      0.685      0.573

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       4/20      10.3G     0.5708     0.8816     0.9883      1.078          8        640: 100%|██████████| 1608/1608 [07:40<00:00,  3.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.57it/s]
                   all        535        601       0.69      0.794      0.753      0.621      0.699      0.786      0.756      0.608

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       5/20      10.2G     0.5522     0.8601      0.956      1.063         12        640: 100%|██████████| 1608/1608 [07:28<00:00,  3.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.79it/s]
                   all        535        601      0.708      0.793      0.774      0.634      0.704      0.789      0.767       0.62

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       6/20      10.2G      0.536     0.8325     0.9274      1.056         12        640: 100%|██████████| 1608/1608 [07:19<00:00,  3.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.75it/s]
                   all        535        601      0.727       0.78      0.773      0.635      0.694        0.8       0.77      0.631

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       7/20      10.2G     0.5221      0.812     0.8963      1.047         12        640: 100%|██████████| 1608/1608 [07:19<00:00,  3.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.80it/s]
                   all        535        601      0.711      0.831      0.792      0.663      0.716       0.81      0.777      0.648

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       8/20      10.2G      0.508     0.8058     0.8687      1.041         12        640: 100%|██████████| 1608/1608 [07:18<00:00,  3.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.79it/s]
                   all        535        601      0.743      0.841      0.826      0.686       0.74      0.836      0.819      0.668

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       9/20      10.2G     0.5038      0.798      0.852      1.036         18        640: 100%|██████████| 1608/1608 [07:38<00:00,  3.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.57it/s]
                   all        535        601      0.754       0.83      0.798      0.672      0.731      0.866      0.793      0.653

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      10/20      10.2G     0.4959     0.7959     0.8306      1.034          7        640: 100%|██████████| 1608/1608 [07:42<00:00,  3.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.55it/s]
                   all        535        601      0.749      0.908      0.839      0.696      0.759      0.875      0.824      0.681
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      11/20      10.2G     0.5638     0.9532     0.7697      1.098          7        640: 100%|██████████| 1608/1608 [07:40<00:00,  3.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.59it/s]
                   all        535        601      0.754      0.904      0.865      0.711      0.771       0.88      0.861      0.691

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      12/20      10.2G     0.5546     0.9337     0.7531      1.083          7        640: 100%|██████████| 1608/1608 [07:38<00:00,  3.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.58it/s]
                   all        535        601      0.796      0.817      0.832      0.694      0.808      0.811      0.829      0.673

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      13/20      10.2G     0.5425     0.9181      0.725      1.076          4        640: 100%|██████████| 1608/1608 [07:38<00:00,  3.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.56it/s]
                   all        535        601      0.779      0.837      0.825      0.695      0.796      0.824      0.817      0.676

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      14/20      10.2G     0.5338     0.9152     0.7067      1.072         11        640: 100%|██████████| 1608/1608 [07:41<00:00,  3.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.54it/s]
                   all        535        601      0.731      0.915      0.849      0.719      0.729      0.889      0.839      0.688

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      15/20      10.2G      0.525     0.9013     0.6831      1.069          5        640: 100%|██████████| 1608/1608 [07:43<00:00,  3.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.55it/s]
                   all        535        601      0.763      0.913      0.866      0.728      0.783       0.88      0.863      0.704

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      16/20      10.2G     0.5222     0.8933     0.6722      1.066          6        640: 100%|██████████| 1608/1608 [07:38<00:00,  3.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.58it/s]
                   all        535        601      0.802      0.878      0.859      0.723      0.798      0.856       0.85        0.7

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      17/20      10.2G        0.5     0.8691     0.6524      1.054          6        640: 100%|██████████| 1608/1608 [07:42<00:00,  3.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.57it/s]
                   all        535        601      0.798      0.873      0.847      0.723      0.793       0.87      0.837      0.701

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      18/20      10.2G     0.4961     0.8604     0.6408      1.049          5        640: 100%|██████████| 1608/1608 [07:38<00:00,  3.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.59it/s]
                   all        535        601      0.817      0.862      0.861      0.724      0.811      0.868      0.859      0.708

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      19/20      10.2G     0.4881     0.8514     0.6266      1.047          5        640: 100%|██████████| 1608/1608 [07:39<00:00,  3.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.58it/s]
                   all        535        601      0.792      0.866      0.863      0.731      0.835      0.843      0.861       0.71

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      20/20      10.2G     0.4777     0.8513     0.6176      1.042          7        640: 100%|██████████| 1608/1608 [07:33<00:00,  3.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.61it/s]
                   all        535        601      0.795      0.872      0.877      0.738      0.808      0.857       0.87      0.718

20 epochs completed in 2.593 hours.
Optimizer stripped from runs/segment/train/weights/last.pt, 124.8MB
Optimizer stripped from runs/segment/train/weights/best.pt, 124.8MB

Validating runs/segment/train/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11x-seg summary (fused): 491 layers, 62,005,593 parameters, 0 gradients, 318.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 34/34 [00:07<00:00,  4.68it/s]
                   all        535        601        0.8      0.884      0.877      0.738      0.807      0.857       0.87      0.719
                 human        142        185      0.824      0.964      0.956      0.826      0.836      0.962      0.956      0.806
             no-object        327        327      0.991      0.969      0.994      0.994      0.991      0.968      0.994      0.994
      undefined-object         66         89      0.585      0.719      0.681      0.394      0.594       0.64      0.661      0.357
Speed: 0.1ms preprocess, 10.3ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to runs/segment/train
Phase 2 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=segment, mode=train, model=yolo11x-seg.pt, data=data.yaml, epochs=30, time=None, patience=100, batch=20, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=1, project=None, name=train2, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0005, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/segment/train2

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
 23        [16, 19, 22]  1   8325497  ultralytics.nn.modules.head.Segment          [3, 32, 384, [384, 768, 768]] 
YOLO11x-seg summary: 667 layers, 62,053,721 parameters, 62,053,705 gradients, 319.7 GFLOPs

Transferred 1077/1077 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /home/naquoc17/TimeOfFlight-Project/Human-instance-segmentation-Yolo11/data/train/labels.cache... 12861 images, 1840 backgrounds, 0 corrupt: 100%|██████████| 12861/12861 [00:00<?, ?it/s]
val: Scanning /home/naquoc17/TimeOfFlight-Project/Human-instance-segmentation-Yolo11/data/valid/labels.cache... 535 images, 0 backgrounds, 0 corrupt: 100%|██████████| 535/535 [00:00<?, ?it/s]
Plotting labels to runs/segment/train2/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0005' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 176 weight(decay=0.0), 187 weight(decay=0.00046875), 186 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 1 dataloader workers
Logging results to runs/segment/train2
Starting training for 30 epochs...

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       1/30      22.5G     0.4659     0.7496     0.8301      1.019          1        640: 100%|██████████| 644/644 [07:17<00:00,  1.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.91it/s]
                   all        535        601      0.776      0.887      0.868      0.725      0.775      0.873       0.86      0.703

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       2/30        23G     0.4831     0.7612     0.7943      1.026          2        640: 100%|██████████| 644/644 [07:19<00:00,  1.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.86it/s]
                   all        535        601      0.745      0.788      0.784      0.656      0.745      0.788      0.777      0.647

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       3/30        23G     0.4961     0.7875     0.8079      1.029          5        640: 100%|██████████| 644/644 [07:02<00:00,  1.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.91it/s]
                   all        535        601      0.776      0.847      0.832      0.672      0.787      0.804      0.816      0.651

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       4/30      23.2G     0.4989     0.7894     0.8098      1.033          3        640: 100%|██████████| 644/644 [07:17<00:00,  1.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.90it/s]
                   all        535        601      0.714      0.848      0.792      0.662      0.704      0.843      0.786      0.646

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       5/30      23.3G     0.4922     0.7756     0.7946      1.028          4        640: 100%|██████████| 644/644 [06:57<00:00,  1.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:06<00:00,  2.00it/s]
                   all        535        601      0.772      0.884      0.846      0.713       0.76       0.89      0.835      0.687

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       6/30        23G      0.485     0.7651     0.7916      1.027          4        640: 100%|██████████| 644/644 [07:04<00:00,  1.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.92it/s]
                   all        535        601      0.764      0.848      0.845       0.71      0.781      0.839      0.844      0.679

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       7/30      23.2G      0.479     0.7646     0.7794      1.025          3        640: 100%|██████████| 644/644 [06:58<00:00,  1.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:06<00:00,  2.01it/s]
                   all        535        601      0.744      0.874       0.83      0.701      0.735      0.863      0.813      0.682

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       8/30      23.3G     0.4651     0.7489     0.7648      1.015          2        640: 100%|██████████| 644/644 [06:51<00:00,  1.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:06<00:00,  2.00it/s]
                   all        535        601      0.733      0.823      0.817      0.683      0.726      0.815      0.807      0.666

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       9/30        23G     0.4754     0.7624     0.7684      1.023          2        640: 100%|██████████| 644/644 [06:53<00:00,  1.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.90it/s]
                   all        535        601      0.809      0.873      0.857      0.709      0.806       0.87      0.849      0.701

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      10/30      23.2G     0.4622     0.7429     0.7446      1.022          3        640: 100%|██████████| 644/644 [07:17<00:00,  1.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.92it/s]
                   all        535        601      0.754      0.897      0.831      0.704       0.76      0.886      0.826      0.693

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      11/30      23.3G     0.4544     0.7321     0.7378      1.014          1        640: 100%|██████████| 644/644 [07:15<00:00,  1.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.91it/s]
                   all        535        601      0.783      0.875      0.856      0.715      0.793      0.869      0.851      0.697

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      12/30        23G     0.4489     0.7357     0.7357      1.015          2        640: 100%|██████████| 644/644 [07:13<00:00,  1.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.92it/s]
                   all        535        601      0.804      0.851      0.827       0.71      0.796      0.839      0.818      0.691

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      13/30      23.2G     0.4473     0.7253     0.7269      1.012          2        640: 100%|██████████| 644/644 [07:14<00:00,  1.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.91it/s]
                   all        535        601      0.829      0.851      0.875      0.726      0.802      0.871      0.862       0.71

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      14/30      23.3G     0.4399     0.7216     0.7192      1.008          1        640: 100%|██████████| 644/644 [07:17<00:00,  1.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.90it/s]
                   all        535        601      0.788      0.833      0.846      0.712      0.784       0.83       0.84      0.695

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      15/30        23G      0.437     0.7009     0.7108      1.008          2        640: 100%|██████████| 644/644 [06:55<00:00,  1.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:06<00:00,  2.01it/s]
                   all        535        601      0.806      0.865      0.853      0.721      0.805      0.853      0.843      0.705

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      16/30      23.2G      0.436     0.7073     0.7098      1.011          3        640: 100%|██████████| 644/644 [07:06<00:00,  1.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.90it/s]
                   all        535        601      0.788      0.857      0.842      0.713      0.792      0.842      0.838      0.698

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      17/30      23.3G     0.4266     0.6854     0.6996      1.008          3        640: 100%|██████████| 644/644 [07:17<00:00,  1.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.90it/s]
                   all        535        601      0.803      0.875      0.857      0.724      0.803      0.875      0.852      0.702

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      18/30        23G     0.4243      0.697     0.6847      1.005          3        640: 100%|██████████| 644/644 [07:16<00:00,  1.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.91it/s]
                   all        535        601      0.818      0.852      0.862       0.73      0.793      0.853      0.846      0.704

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      19/30      23.2G     0.4269     0.6953     0.6886      1.008          2        640: 100%|██████████| 644/644 [07:14<00:00,  1.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.93it/s]
                   all        535        601      0.791      0.879       0.84      0.716      0.777      0.868      0.829      0.694

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      20/30      23.3G     0.4172     0.6873     0.6783      1.013          1        640: 100%|██████████| 644/644 [07:04<00:00,  1.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:06<00:00,  2.00it/s]
                   all        535        601       0.79      0.874       0.85      0.727      0.783      0.879      0.844       0.71
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      21/30        23G     0.5003     0.8671     0.6302      1.062          1        640: 100%|██████████| 644/644 [06:51<00:00,  1.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  2.00it/s]
                   all        535        601       0.79      0.861      0.847      0.732      0.793      0.857      0.841      0.705

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      22/30      23.2G     0.4856     0.8526     0.6128      1.057          1        640: 100%|██████████| 644/644 [06:57<00:00,  1.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.91it/s]
                   all        535        601      0.818      0.849      0.854      0.729      0.822      0.837      0.843      0.707

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      23/30      23.3G     0.4843     0.8401     0.6017       1.05          1        640: 100%|██████████| 644/644 [07:15<00:00,  1.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.90it/s]
                   all        535        601      0.787      0.873      0.849       0.73      0.815      0.839      0.843      0.701

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      24/30      23.1G     0.4715      0.823     0.5891      1.041          1        640: 100%|██████████| 644/644 [07:15<00:00,  1.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.91it/s]
                   all        535        601      0.793      0.868      0.832      0.724      0.791      0.873      0.826      0.698

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      25/30      23.2G     0.4685     0.8282     0.5817      1.043          1        640: 100%|██████████| 644/644 [07:15<00:00,  1.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:06<00:00,  2.01it/s]
                   all        535        601      0.801      0.861      0.842      0.729      0.803       0.86      0.836      0.712

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      26/30      23.3G     0.4609     0.8176     0.5713       1.04          1        640: 100%|██████████| 644/644 [06:53<00:00,  1.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.91it/s]
                   all        535        601      0.797      0.861      0.836      0.726      0.806      0.856      0.834      0.708

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      27/30        23G     0.4595     0.8082     0.5692       1.04          0        640: 100%|██████████| 644/644 [07:12<00:00,  1.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.91it/s]
                   all        535        601      0.783      0.854      0.828       0.72      0.786      0.858      0.826      0.702

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      28/30      23.2G     0.4488     0.8007     0.5545      1.037          1        640: 100%|██████████| 644/644 [07:13<00:00,  1.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.90it/s]
                   all        535        601      0.778      0.872      0.824      0.719      0.781      0.876      0.824      0.701

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      29/30      23.2G     0.4391     0.7745     0.5722       1.03         19         29/30      23.2G     0.4385     0.7729     0.5696      1.031         18         29/30      23.2G     0.4385     0.7729     0.5696      1.031         18         29/30      23.2G     0.436      29/30      23.3G     0.4479     0.7957     0.5499      1.036          1        640: 100%|██████████| 644/644 [07:29<00:00,  1.43it/s]11%|█▏        | 74/644 [00:51<06:39,  1.43it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.87it/s]
                   all        535        601      0.775      0.895       0.83      0.722      0.775      0.895      0.826      0.704

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      30/30      23.1G     0.4427     0.7926     0.5426      1.023          2        640: 100%|██████████| 644/644 [07:25<00:00,  1.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.85it/s]
                   all        535        601      0.777      0.881      0.831      0.726      0.778      0.879      0.827      0.707

30 epochs completed in 3.656 hours.
Optimizer stripped from runs/segment/train2/weights/last.pt, 124.8MB
Optimizer stripped from runs/segment/train2/weights/best.pt, 124.8MB

Validating runs/segment/train2/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11x-seg summary (fused): 491 layers, 62,005,593 parameters, 0 gradients, 318.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 14/14 [00:07<00:00,  1.99it/s]
                   all        535        601      0.833      0.851      0.875      0.727      0.811      0.843      0.863       0.71
                 human        142        185      0.828      0.968      0.952      0.804      0.825      0.968      0.952      0.797
             no-object        327        327      0.982      0.979      0.992      0.988       0.98      0.979      0.992      0.988
      undefined-object         66         89      0.689      0.607       0.68      0.388      0.628      0.584      0.644      0.344
Speed: 0.1ms preprocess, 10.0ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to runs/segment/train2
Phase 3 started ...
New https://pypi.org/project/ultralytics/8.3.75 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
engine/trainer: task=segment, mode=train, model=yolo11x-seg.pt, data=data.yaml, epochs=10, time=None, patience=5, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=0, project=None, name=train22, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=True, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.0001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/segment/train22

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
 23        [16, 19, 22]  1   8325497  ultralytics.nn.modules.head.Segment          [3, 32, 384, [384, 768, 768]] 
YOLO11x-seg summary: 667 layers, 62,053,721 parameters, 62,053,705 gradients, 319.7 GFLOPs

Transferred 1077/1077 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /home/naquoc17/TimeOfFlight-Project/Human-instance-segmentation-Yolo11/data/train/labels.cache... 12861 images, 1840 backgrounds, 0 corrupt: 100%|██████████| 12861/12861 [00:00<?, ?it/s]
val: Scanning /home/naquoc17/TimeOfFlight-Project/Human-instance-segmentation-Yolo11/data/valid/labels.cache... 535 images, 0 backgrounds, 0 corrupt: 100%|██████████| 535/535 [00:00<?, ?it/s]
Plotting labels to runs/segment/train22/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0001' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001429, momentum=0.9) with parameter groups 176 weight(decay=0.0), 187 weight(decay=0.0005), 186 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 0 dataloader workers
Logging results to runs/segment/train22
Starting training for 10 epochs...
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       1/10      18.8G     0.5042     0.8772     0.6543      1.062         13        640: 100%|██████████| 804/804 [09:20<00:00,  1.43it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:08<00:00,  2.04it/s]
                   all        535        601      0.754      0.821      0.775      0.668      0.735      0.836      0.769      0.658

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       2/10      19.3G     0.5196     0.8867     0.6624       1.07         13        640: 100%|██████████| 804/804 [08:51<00:00,  1.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:08<00:00,  1.89it/s]
                   all        535        601      0.731      0.856      0.829      0.681      0.728      0.844      0.818      0.669

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       3/10      19.3G     0.5286     0.8979     0.6838      1.078         13        640: 100%|██████████| 804/804 [08:57<00:00,  1.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:08<00:00,  1.89it/s]
                   all        535        601      0.778       0.88      0.859      0.722      0.772      0.868      0.849      0.707

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       4/10      18.9G     0.5294     0.8906     0.6779      1.087         11        640: 100%|██████████| 804/804 [08:01<00:00,  1.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:07<00:00,  2.25it/s]
                   all        535        601      0.823      0.837      0.866      0.748      0.829      0.837      0.865      0.728

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       5/10      18.8G     0.5161       0.88     0.6557      1.078         12        640: 100%|██████████| 804/804 [07:59<00:00,  1.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:07<00:00,  2.19it/s]
                   all        535        601      0.815      0.857      0.855       0.73      0.804      0.852      0.843      0.714

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       6/10      18.9G     0.5046     0.8672     0.6355      1.061         13        640: 100%|██████████| 804/804 [07:58<00:00,  1.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:07<00:00,  2.24it/s]
                   all        535        601      0.832      0.842      0.863      0.733       0.81      0.846      0.858      0.712

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       7/10      18.9G      0.489     0.8527     0.6084      1.053         11        640: 100%|██████████| 804/804 [07:57<00:00,  1.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:07<00:00,  2.25it/s]
                   all        535        601      0.841      0.865      0.877      0.747      0.837      0.856      0.871      0.724

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       8/10      18.9G     0.4721     0.8327     0.5859      1.044         13        640: 100%|██████████| 804/804 [07:57<00:00,  1.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:07<00:00,  2.19it/s]
                   all        535        601      0.812      0.858      0.861       0.74      0.812      0.858      0.857      0.722

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
       9/10      18.9G     0.4614     0.8145     0.5723      1.036         13        640: 100%|██████████| 804/804 [07:58<00:00,  1.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:07<00:00,  2.19it/s]
                   all        535        601      0.835      0.846      0.854      0.738      0.823      0.855      0.851      0.713
EarlyStopping: Training stopped early as no improvement observed in last 5 epochs. Best results observed at epoch 4, best model saved as best.pt.
To update EarlyStopping(patience=5) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

9 epochs completed in 1.277 hours.
Optimizer stripped from runs/segment/train22/weights/last.pt, 124.8MB
Optimizer stripped from runs/segment/train22/weights/best.pt, 124.8MB

Validating runs/segment/train22/weights/best.pt...
Ultralytics 8.3.74 🚀 Python-3.10.12 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 3090, 24139MiB)
YOLO11x-seg summary (fused): 491 layers, 62,005,593 parameters, 0 gradients, 318.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 17/17 [00:07<00:00,  2.25it/s]
                   all        535        601      0.822      0.836      0.866      0.748      0.829      0.836      0.865      0.729
                 human        142        185      0.825      0.989      0.958      0.835      0.826      0.989      0.958      0.809
             no-object        327        327      0.978      0.965      0.994      0.993      0.978      0.965      0.994      0.993
      undefined-object         66         89      0.663      0.554      0.645      0.416      0.682      0.554      0.642      0.384
Speed: 0.1ms preprocess, 9.0ms inference, 0.0ms loss, 0.4ms postprocess per image
Results saved to runs/segment/train22
done
naquoc17@FAWorkstation:~/TimeOfFlight-Project/Human-instance-segmentation-Yolo11$ 
