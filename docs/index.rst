.. include:: ./README.md
        :parser: myst_parser.sphinx_

.. Indices and tables
.. ==================

..  toctree::
        :maxdepth: 2
        :caption: Get Started
        :hidden:
        :glob:

        01_get_started/installation.rst
        notebooks/01_get_started/introduction_to_sportslabkit.ipynb

..  toctree::
        :maxdepth: 2
        :caption: User Guide
        :hidden:

        notebooks/02_user_guide/dataset_preparation.ipynb
        notebooks/02_user_guide/dataframe_manipulation.ipynb
        notebooks/02_user_guide/detection_with_yolov5.ipynb
        notebooks/02_user_guide/tracking_with_deepsort.ipynb
        notebooks/02_user_guide/tracking_evaluation.ipynb
        notebooks/02_user_guide/tracking_the_ball.ipynb
        notebooks/02_user_guide/visualization.ipynb

..  toctree::
        :maxdepth: 2
        :caption: Core Components
        :hidden:

        notebooks/03_core_components/camera.ipynb
        notebooks/03_core_components/detection_model.ipynb
        notebooks/03_core_components/image_model.ipynb
        notebooks/03_core_components/motion_model.ipynb
        notebooks/03_core_components/single_object_tracker.ipynb
        notebooks/03_core_components/multi_object_tracker.ipynb

..  toctree::
        :caption: API Reference
        :hidden:
        :maxdepth: 2

        autoapi/sportslabkit/index

..  toctree::
        :maxdepth: 1
        :caption: Dev
        :hidden:

        04_contributing.rst
        notebooks/05_dev/logging-demo.ipynb
