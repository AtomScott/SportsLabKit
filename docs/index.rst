.. include:: ./README.md
        :parser: myst_parser.sphinx_

.. Indices and tables
.. ==================

..  toctree::
        :maxdepth: 2
        :caption: Get Started
        :hidden:
        :glob:

        notebooks/01_get_started/installation.ipynb
        notebooks/01_get_started/introduction_to_sportslabkit.ipynb

..  toctree::
        :maxdepth: 2
        :caption: User Guide
        :hidden:

        notebooks/02_user_guide/00_dataset_preparation.ipynb
        notebooks/02_user_guide/01_dataframe_manipulation.ipynb
        notebooks/02_user_guide/02_dataframe_visualization.ipynb
        notebooks/02_user_guide/03_evaluation_metrics.ipynb
        notebooks/02_user_guide/04_tune_the_tracker.ipynb
        notebooks/02_user_guide/05_tracking_the_ball.ipynb
        notebooks/02_user_guide/06_tracking_the_players.ipynb
        notebooks/02_user_guide/07_tracking_method_comparison.ipynb
        notebooks/02_user_guide/08_GNSS_data.ipynb
  
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
