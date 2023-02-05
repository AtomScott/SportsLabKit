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

You will neeed to install the following dependencies:

.. code-block:: bash
    pip install torch torchvision pytorch-lightning


To use torch reid, you will need to install the following dependencies:

.. code-block:: bash
    pip install git+https://github.com/KaiyangZhou/deep-person-reid.git

We recommed using poetry to handle dependencies. So you can also install poetry and run the following command:

.. code-block:: bash

    poetry install
    poetry run pip install torch torchvision pytorch-lightning 
    poetry run pip install git+https://github.com/KaiyangZhou/deep-person-reid.git


.. warning::
    The software is currently in development so it will break and change frequently! Star or watch the repo to get notified of updates. Anything before version 1.0.0 is not considered stable and may break at any time (sorry!).