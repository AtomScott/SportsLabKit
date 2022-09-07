############
Installation
############

--------------

****************
Install with pip
****************

Install any supported version of PyTorch if you want from `PyTorch Installation Page <https://pytorch.org/get-started/locally/#start-locally>`_.
Now you can install using `pip <https://pypi.org/project/SoccerTrack/>`_ using the following command:

.. code-block:: bash

    pip install soccertrack

.. warning::
    TODO: This is not supported yet. Please git clone the repo and build from source.

--------------

******************
Install with Conda
******************

If you don't have conda installed, follow the `Conda Installation Guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`_.
soccertrack can be installed with `conda <https://anaconda.org/conda-forge/soccertrack>`_ using the following command:

.. code-block:: bash

    conda install soccertrack -c conda-forge

You can also use `Conda Environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_:

.. code-block:: bash

    conda activate my_env
    conda install soccertrack -c conda-forge

.. warning::
    TODO: This is not supported yet. Please git clone the repo and build from source.

--------------

*****************
Build from Source
*****************

Install nightly from the source. Note that it contains all the bug fixes and newly released features that
are not published yet. This is the bleeding edge, so use it at your own discretion.

.. code-block:: bash

    pip install https://github.com/AtomScott/SoccerTrack.git -U

--------------

*******************************
Build with Docker (Recommended)
*******************************

A docker image is available on `Docker Hub <https://hub.docker.com/r/atomscott/soccertrack>`_.
You can pull the image using the following command:

.. code-block:: bash

    docker pull atomscott/soccertrack:latest