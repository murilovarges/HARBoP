# Tutorial 2: Encoding 2D Poses in Parameter Space

This tutorial will help you, step-by-step, how encode 2D Poses in Parameter Space.

## Preparing data

In this example, we assume that you already perform 2D poses extraction or downloaded the 2D poses provided in [2D Poses Extraction](2DPoses_extraction.md)

## Encoding 2D Poses in Parameter Space

In this example, we will encode 2D poses in Parameter Space for the **KTH dataset** videos.

```
python tools/ParameterSpaceConversion/convert_poses_parameter_space.py \
--poses_base_dir=/home/murilo/dataset/KTH \
--input_dir=2DPoses \
--output_dir=2DPoses_SpaceParam \
--output_images_dir=2DPoses_SpaceParam_Images \
--image_height=120 \
--image_width=160

```


In this example, we will encode 2D poses in Parameter Space for **Weizmann dataset** videos .

```
python tools/ParameterSpaceConversion/convert_poses_parameter_space.py \
--poses_base_dir=/home/murilo/dataset/Weizmann \
--input_dir=2DPoses \
--output_dir=2DPoses_SpaceParam \
--output_images_dir=2DPoses_SpaceParam_Images \
--image_height=144 \
--image_width=180

```

## Next
As next step follow the link:
[Features Extraction](features_extraction.md)

