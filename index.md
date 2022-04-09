---
layout: home
pagination: 
  enabled: true
---

## Abstract
Tracking devices that can track both players and balls are critical to the performance of sports teams. Recently, significant effort has been focused on building larger broadcast sports video datasets. However, broadcast videos do not show the entire pitch and only provides partial information about the game. On the other hand, other camera perspectives can capture the whole field in a single frame, such as fish-eye and bird-eye view (drone) cameras. Unfortunately, there has not been a dataset where such data has been publicly shared until now. 

This paper proposes SoccerTrack, a dataset set consisting of GNSS and bounding box tracking data annotated on video captured with a 8K-resolution fish-eye camera and a 4K-resolution drone camera. In addition to a benchmark tracking algorithm, we include code for camera calibration and other preprocessing. Finally, we evaluate the tracking accuracy among a GNSS, fish-eye camera and drone camera data. SoccerTrack is expected to provide a more robust foundation for designing MOT algorithms that are less reliant on visual cues and more reliant on motion analysis.

## Download

### Training Data

{% include train_data.html %}


### Test Data(Full)

{% include test_data_full.html %}

### Test Data(Split)

{% include test_data_split.html %}
