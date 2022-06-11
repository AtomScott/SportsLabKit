# SoccerTrack

![](https://raw.githubusercontent.com/AtomScott/SoccerTrack/gh-pages/img/title-banner.png)

[![Documentation Status](https://readthedocs.org/projects/soccertrack/badge/?version=latest)](https://soccertrack.readthedocs.io/en/latest/?badge=latest) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/b121ddfb4e244b6d88096840bdcfa1a2)](https://www.codacy.com/gh/AtomScott/SoccerTrack/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=AtomScott/SoccerTrack&amp;utm_campaign=Badge_Grade)


A Dataset and Tracking Algorithm for Soccer with Fish-eye and Drone Videos.


* [Project page](https://atomscott.github.io/SoccerTrack/)
* [Paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.pdf)
* ~~[Tracking Algorithm](https://github.com/AtomScott/SoccerTrack/tree/develop)~~(Work In Progress)
* ~~[Documentation](https://soccertrack.readthedocs.io/)~~ (Work In Progress)
* ~~[Dataset Download](https://atomscott.github.io/SoccerTrack/#download)~~ (Work In Progress)

> **Note**
> 
> We are finalizing code and dataset for release by June 20th (Workshop date for CVSports'22)!  

## Dataset Preview

Below are low resoltion samples from the dataset we plan to release. The actual dataset will contain 4K (drone) and 8K (fisheye) footage!

<img width="586" alt="image" src="https://user-images.githubusercontent.com/22371492/172513053-68ef75c4-435a-40e6-96fb-5a75319e32d6.png">

### Drone Video

https://user-images.githubusercontent.com/22371492/172512347-9f441847-3a00-493d-8ca7-12de73d28b0d.mp4

### Fisheye Video

https://user-images.githubusercontent.com/22371492/172512368-9f4adfb5-e7aa-4abb-b3e7-3c0d7d73eb08.mp4

> **Note** The resolution for the fisheye camera may change after calibration.

## Dataset Release Schedule

| Date | Drone | Fisheye |
|------|-------|---------|
| 2022/6/20 | 15min | 15min   |
| 2022/7/1  | 20min | 20min   |
| 2022/8/1  | 30min | 30min   |

## Docker

[Dockerhub](https://hub.docker.com/repository/docker/atomscott/soccertrack)

## Roadmap

### Release 0.0.1 !

* [] Add a simple example notebook.
* [] Add auto summary to docs (fork apidoc and include module members to jinja context).


## Contributing

See the [Contributing Guide](https://soccertrack.readthedocs.io/en/latest/contributing.html) for more information.

## Citation
```
@inproceedings{scott2022soccertrack,
  title={SoccerTrack: A Dataset and Tracking Algorithm for Soccer With Fish-Eye and Drone Videos},
  author={Scott, Atom and Uchida, Ikuma and Onishi, Masaki and Kameda, Yoshinari and Fukui, Kazuhiro and Fujii, Keisuke},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3569--3579},
  year={2022}
}
```
