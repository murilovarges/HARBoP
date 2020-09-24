# Tutorial 3: Features Extraction

This tutorial will help you, step-by-step, how to extract features (Angles and Trajectories) from 2D Poses.

Before proceeding make sure that you have already extract or download 2D poses, see [2D Poses Extraction](2DPoses_extraction.md) for more information.

Before proceeding make sure that you have already encode 2D poses in Parameter Space, see [Encoding 2D poses in Space Param](2DPoses_spaceparam.md) for more information.

Experiments were performed in two public dataset [KTH](http://www.nada.kth.se/cvap/actions/) and [Weizmann](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html).

In this example, we will extract features from 2D poses by running the following scripts.

## Computing Angles Features from 2D Poses

KTH dataset example:
```
python tools/FeaturesExtraction/WEF/compute_angles_from_body_parts.py \
--poses_base_dir=/home/murilo/dataset/KTH \
--input_dir=2DPoses_Person \
--output_dir=Angles_from_2DPoses
```
Weizmann dataset example:
```
python tools/FeaturesExtraction/WEF/compute_angles_from_body_parts.py \
--poses_base_dir=/home/murilo/dataset/Weizmann \
--input_dir=2DPoses \
--output_dir=Angles_from_2DPoses
```

## Computing Trajectories from Body Parts Features (Parameter Space)
KTH dataset example:
```
python tools/FeaturesExtraction/PEF/compute_trajectories_from_parameter_space.py \
--space_param_base_dir=/home/murilo/dataset/KTH \
--input_dir=2DPoses_SpaceParam \
--output_dir=2DPoses_SpaceParam_Trajectories \
--number_frames=20 --stride=10
```

Weizmann dataset example:
```
python tools/FeaturesExtraction/PEF/compute_trajectories_from_parameter_space.py \
--space_param_base_dir=/home/murilo/dataset/Weizmann \
--input_dir=2DPoses \
--output_dir=2DPoses_SpaceParam_Trajectories \
--number_frames=20 --stride=1
```


## Next
As next step follow the link:
[Human Action Recognition](classification.md)
