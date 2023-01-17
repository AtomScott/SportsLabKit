# SoccerTrack

![](https://raw.githubusercontent.com/AtomScott/SoccerTrack/gh-pages/img/title-banner.png)

[![Documentation Status](https://readthedocs.org/projects/soccertrack/badge/?version=latest)](https://soccertrack.readthedocs.io/en/latest/?badge=latest) 
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/atomscott/soccertrack)
[![PWC](https://img.shields.io/badge/%7C-Papers%20with%20Code-lightblue)](https://paperswithcode.com/dataset/soccertrack-dataset)

[![DeepSource](https://deepsource.io/gh/AtomScott/SoccerTrack.svg/?label=active+issues&show_trend=true&token=TIxJg8BLzszYnWeVDMHr6pMU)](https://deepsource.io/gh/AtomScott/SoccerTrack/?ref=repository-badge)
[![DeepSource](https://deepsource.io/gh/AtomScott/SoccerTrack.svg/?label=resolved+issues&show_trend=true&token=TIxJg8BLzszYnWeVDMHr6pMU)](https://deepsource.io/gh/AtomScott/SoccerTrack/?ref=repository-badge)

**[IMPORTANT (2022/11/03)]**

After receving reports of erroneous  data, we have fixed and reuploaded a majority of SoccerTrack. We are also adding videos with visualized bounding boxes so that you can be sure that the data is good. The visualizations can be found in the viz_results directory under Top-view/Wide-view (see [Kaggle](https://www.kaggle.com/datasets/atomscott/soccertrack)).

However, there is still work to do. In the meantime, we have created a spreadsheet to keep everyone updated on our progress.
[Spreadsheet Link](https://docs.google.com/spreadsheets/d/1V4TF84nIZWtYBrT6oNhAc3tp01QCBn41aadp96vfWww/edit#gid=208157415)

---
A Dataset and Tracking Algorithm for Soccer with Fish-eye and Drone Videos.


* [Project page](https://atomscott.github.io/SoccerTrack/)
* [Paper](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.pdf)
* [Dataset Download](https://atomscott.github.io/SoccerTrack/#download) 🌟 NEW
* [Tracking Algorithm](https://github.com/AtomScott/SoccerTrack) (Work In Progress)
* [Documentation](https://soccertrack.readthedocs.io/) (Work In Progress)


## Dataset Details

 -/- | **Wide-View Camera** | **Top-View Camera** | **GNSS** 
---|---|---|---
 Device | Z CAM E2-F8 | DJI Mavic 3 | STATSPORTS APEX 10 Hz 
 Resolution | 8K (7,680 × 4,320 pixel) | 4K (3,840 × 2,160 pixesl) | Abs. err. in 20-m run: 0.22 ± 0.20 m  
 FPS | 30 | 30 | 10 
 Player tracking | ✅ | ✅ | ✅ 
 Ball tracking | ✅ | ✅ | - 
 Bounding box | ✅ | ✅ | - 
 Location data | ✅ | ✅ | ✅ 
 Player ID | ✅ | ✅ | ✅

All data in SoccerTrack was obtained from 11-vs-11 soccer games between college-aged athletes. Measurements were conducted after we received the approval of Tsukuba university’s ethics committee, and all participants provided signed informed permission. After recording several soccer matches, the videos were semi-automatically annotated based on the GNSS coordinates of each player.

Below are low resolution samples from the soccertrack dataset we plan to release. The actual dataset will contains (drone) and 8K (fisheye) footage!

### Drone Video

<video style='max-width:640px' controls>
  <source src="https://user-images.githubusercontent.com/22371492/178085041-a8a2de85-bcd3-4c81-8b81-5ca93dbd4336.mp4" type="video/mp4">
</video>

https://user-images.githubusercontent.com/22371492/178085041-a8a2de85-bcd3-4c81-8b81-5ca93dbd4336.mp4

### Fisheye Video
<video style='max-width:640px' controls>
  <source src="https://user-images.githubusercontent.com/22371492/178085027-5d25781d-e3ed-4791-ad14-141b58187dcf.mp4" type="video/mp4">
</video>

https://user-images.githubusercontent.com/22371492/178085027-5d25781d-e3ed-4791-ad14-141b58187dcf.mp4


> **Note** The resolution for the fisheye camera may change after calibration.

## Dataset Download

The SoccerTrack Dataset is available to download from the links below!

* [Top-view](https://drive.google.com/drive/folders/12rasAk-52YSAwReJNIlTZIa794UhRU4J?usp=sharing)
* [Wide-view](https://drive.google.com/drive/folders/1XgrPHBYnz-LOB2vZsB4koVUMgjl_gwqF?usp=sharing)
* [GNSS data](https://drive.google.com/drive/folders/15i4GJ1Rl5rwnOOuHv34Ar1K8wxKifnIJ?usp=sharing)

For more details on how to use the dataset, please see the section "[Dataset Preparation](https://soccertrack.readthedocs.io/en/latest/01_get_started/dataset_preparation.html)".

## Docker

[Dockerhub](https://hub.docker.com/repository/docker/atomscott/soccertrack)

## Contributing

See the [Contributing Guide](https://soccertrack.readthedocs.io/en/latest/contributing.html) for more information.

## Papers

<table>
<td width=30% style='padding: 20px;'>
<a href="https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.pdf">
<img src='https://raw.githubusercontent.com/AtomScott/SoccerTrack/feature/major_refactor/docs/_static/paper_preview.jpg'/>
</a>
</td>
<td width=70%>
  <p>
    <b>SoccerTrack:</b><br>
    A Dataset and Tracking Algorithm for Soccer with Fish-eye and Drone Videos
  </p>
  <p>
    Atom Scott*, Ikuma Uchida*, Masaki Onishi, Yoshinari Kameda, Kazuhiro Fukui, Keisuke Fujii
  </p>
  <p>
    <i> Presented at CVPR Workshop on Computer Vision for Sports (CVSports'22). *Authors contributed equally. </i>
  </p>
  <div>
    <a href='https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=adobe-acrobat-reader'/>
    </a>
    <a href='https://github.com/AtomScott/SoccerTrack'>
      <img src='https://img.shields.io/badge/Code-Page-blue?style=for-the-badge&logo=github'/>
    </a>
    <a href='https://soccertrack.readthedocs.io/'>
      <img src='https://img.shields.io/badge/Documentation-Page-blue?style=for-the-badge&logo=read-the-docs'/>
    </a>
  </div>
</td>
</table>

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

## Acknowledgements

Part of the tracking module has been adapted from [motpy](https://github.com/wmuron/motpy). We would like to thank the authors for their work.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=atomscott/soccertrack&type=Date)](https://star-history.com/#atomscott/soccertrack&Date)
