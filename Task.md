;l
# Ref
+ [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) + [here](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks)
# First Step
+ Person(body) detection
+ Tracking
+ Pose estimate - alpha pose
  + Detect skeleton.
  + Mapping with track.
  + Draw skeleton.
+ Action recognition.

## Deep into model
### Pose estimate
+ In the futures
### Action recognition.
+ ST-GCN(Spatial Temporal Graph Convolutional Networks) has transferred to [MMSkeleton](https://github.com/open-mmlab/mmskeleton)
+ Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action
Recognition [paper](https://arxiv.org/pdf/1801.07455.pdf).
+ Data :
  + Before training and testing, for convenience of fast data loading, the datasets should be converted to proper file structure.
  + [st-gcn-processed-data](https://drive.google.com/file/d/103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb/view)
+ Ref : 
  + [st-gcn](https://github.com/yysijie/st-gcn)
  + [ codebase, dataset and models for the paper](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md)