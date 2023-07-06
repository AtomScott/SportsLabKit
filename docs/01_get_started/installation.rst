############
Installation
############

--------------

**************************
Install with pip (outdate)
**************************

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
    Installation via pip is currently updated and will not work. Please use the conda installation method.

******************
Install with conda
******************

At the moment, creating a conda environment is not supported. However, you can install the dependencies using conda and then install the package by cloning and using pip.
I use this method whenever I want to install the package on a new machine.

.. code-block:: bash

    git clone https://github.com/AtomScott/TeamTrack.git
    conda create -y --name soccertrack python=3.10
    conda activate soccertrack

    python -m pip install -e .
    python -m pip install torch torchvision pytorch-lightning
    python -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
    python -m pip install ultralytics
    python -m pip install git+https://github.com/openai/CLIP.git