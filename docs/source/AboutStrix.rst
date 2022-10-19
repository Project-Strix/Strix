Getting Started
===============

Prerequisites
~~~~~~~~~~~~~

| Strix is powered by `Pytorch <https://pytorch.org>`__,
  `MONAI <https://monai.io>`__ and
  `Ignite <https://pytorch-ignite.ai>`__.
| Strix relies heavily on following packages to make it work.
  Theoretically, these packages will be automatically installed via pip
  installation. If not, please manually install them.

-  pytorch
-  tb-nightly
-  click
-  tqdm
-  numpy
-  scipy
-  scikit-image
-  scikit-learn
-  nibabel
-  pytorch-ignite
-  monai_ex

Installation
~~~~~~~~~~~~

For developers, we suggest you to get a local copy up and install.

::

   git clone https://gitlab.com/ChingRyu/Strix.git
   pip install -e ./Strix

For users, you can just install via pip.

::

   pip install git+https://gitlab.com/ChingRyu/Strix.git

More details please refer to `install <./install.md>`__.

.. raw:: html

   <!-- USAGE EXAMPLES -->

Usage
=====


Strix has 7 different commands:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``strix-train``: Main train entry. Use this command for general DL
   training process.
-  ``strix-train-from-cfg``: Begin a training process from specified
   configure file, usually used for reproduction.
-  ``strix-train-and-test``: Begin a full training&testing process
   automatically.
-  ``strix-test-from-cfg``: Begin a testing processing from specified
   configure file.
-  ``strix-nni-search``: Use `NNI <https://nni.readthedocs.io>`__ for
   automatic hyper-parameter tuning.
-  ``strix-check-data``: Visualize preprocessed input dataset.
-  ``strix-gradcam-from-cfg``: Gradcam visualization.

| **Here is a usage example!**
.. image:: _static/usage-example.png
   :width: 650

.. _how-to-use-my-own-dataset--network:

How to use my own dataset & network?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  If you want use your own dataset, first you need to create a simple
   python script of a configuration to generate your dataset. For more
   details, please refer to this `readme <strix/datasets/README.md>`__
-  If you want try your own network, you need to follow this
   `steps <strix/models/README.md>`__ to easily register your network to
   Strix.
-  After preparation, just simply put own dataset/network file into
   custom folder, and run!


Roadmap
-------

See the `open issues <https://gitlab.com/ChingRyu/Strix/issues>`__ for a
list of proposed features (and known issues).


Contributing
------------

Contributions are what make the open source community such an amazing
place to be learn, inspire, and create. Any contributions you make are
**greatly appreciated**.


License
-------

Distributed under the GNU GPL v3.0 License. See ``LICENSE`` for more
information.


Contact
-------

Chenglong Wang - clwang@phy.ecnu.edu.cn

Project Link:
`https://gitlab.com/ChingRyu/Strix <https://gitlab.com/ChingRyu/Strix>`__

