# SportsLabKit

![](https://raw.githubusercontent.com/AtomScott/SoccerTrack/gh-pages/img/title-banner.png)
[![Documentation Status](https://readthedocs.org/projects/soccertrack/badge/?version=latest)](https://soccertrack.readthedocs.io/en/latest/?badge=latest) 
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/atomscott/soccertrack)
[![PWC](https://img.shields.io/badge/%7C-Papers%20with%20Code-lightblue)](https://paperswithcode.com/dataset/soccertrack-dataset)
[![dm](https://img.shields.io/pypi/dm/soccertrack)](https://pypi.org/project/soccertrack/)

[![DeepSource](https://deepsource.io/gh/AtomScott/SoccerTrack.svg/?label=active+issues&show_trend=true&token=TIxJg8BLzszYnWeVDMHr6pMU)](https://deepsource.io/gh/AtomScott/SoccerTrack/?ref=repository-badge)
[![DeepSource](https://deepsource.io/gh/AtomScott/SoccerTrack.svg/?label=resolved+issues&show_trend=true&token=TIxJg8BLzszYnWeVDMHr6pMU)](https://deepsource.io/gh/AtomScott/SoccerTrack/?ref=repository-badge)


## **News**

* **(2023/08/10)** [Announcing SportLabKit!](https://atomscott.me/blog-posts-table-includes-wip/announcing-sportslabkit) We are currently working on a dataset for basketball and handball which will be released in the near future. Stay tuned!

---

Introducing SportsLabKit – your go-to toolkit for unlocking the game's secrets! Tailored for everyone from coaches to hobbyists, it's all about transforming sports videos into insights you can act on.

Starting strong with soccer, we're on our way to slam-dunking basketball and handball too. Want to turn a game's footage into numbers for analysis? We've got you covered.

What's Inside?

Tracking: Spot-on tracking for soccer today, with basketball and handball on the horizon.
Simplicity: Videos to numbers? Done. No fuss, no hassle.
Growth: Event detection and pose estimation are coming. We're just getting started.
Join us in the SportsLabKit journey, and let's take sports analysis to the next level!

## Documentation

See the [documentation](https://sportslabkit.readthedocs.io/).

## Install

### pip

The software can be installed using `pip`.

```bash
pip install SportsLabKit
```

You will neeed to install the following dependencies:
```bash
pip install torch torchvision pytorch-lightning
```

To use torch reid, you will need to install the following dependencies:
```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

We recommed using poetry to handle dependencies. So install poetry and run the following command:
```bash
poetry install
poetry run pip install torch torchvision pytorch-lightning 
poetry run pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

> **Note** The software is currently in development so it will break and change frequently!

## Contributing

See the [Contributing Guide](https://soccertrack.readthedocs.io/en/latest/contributing.html) for more information.

## Related Papers

<table>
<td width=30% style='padding: 20px;'>
<a href="https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.pdf">
<img src='https://raw.githubusercontent.com/AtomScott/SoccerTrack/c13b88c2700610bd9c886976a92dd81afa7a7a98/docs/_static/paper_preview.jpg'/>
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

See papers that cite SoccerTrack on [Google Scholar](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=13090652901875753929).
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

## Contributors ✨

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=atomscott/soccertrack&type=Date)](https://star-history.com/#atomscott/soccertrack&Date)

## Contributors ✨

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://atomscott.me/"><img src="https://avatars.githubusercontent.com/u/22371492?v=4?s=100" width="100px;" alt="Atom Scott"/><br /><sub><b>Atom Scott</b></sub></a><br /><a href="#maintenance-AtomScott" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/IkumaUchida"><img src="https://avatars.githubusercontent.com/u/48281753?v=4?s=100" width="100px;" alt="Ikuma Uchida"/><br /><sub><b>Ikuma Uchida</b></sub></a><br /><a href="#tutorial-IkumaUchida" title="Tutorials">✅</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
