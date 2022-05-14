.. Strix documentation master file, created by
   sphinx-quickstart on Thu May  5 19:47:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Welcome to Strix's documentation!

Welcome to Strix!
==================

.. image:: _static/banner.png
    :align: center
    :scale: 35%
    :target: https://palletsprojects.com/p/click/

.. raw:: html

     <h3 align="center">Strix</h3>
     <p align="center">
       A Medical Deep Learning Platform
       <br />
       <i>Make deep-learning easier for medical problems</i>
       <br />
       <a href="https://gitlab.com/ChingRyu/Strix"><strong>Explore the docs »</strong></a>
       <br />
       <br />
       <a href="https://gitlab.com/ChingRyu/Strix">View Demo</a>
       ·
       <a href="https://gitlab.com/ChingRyu/Strix/issues">Report Bug</a>
       ·
       <a href="https://gitlab.com/ChingRyu/Strix/issues">Request Feature</a>
     </p>
   </p>


About The Project
=================

Motivation
----------

*We are trying to create a comprehensive framework to easily build
medical deep-learning applications.*

-  Friendly interactive interface
-  Good plug and play capacity
-  Various tasks support
-  Easy debug & Good reproducibility

Design Concept
--------------

| *We aim to disentangle both Data scientist & Archi Engineer, Networks & Pipelines.*
| You can easily put your own datasets and networks into Strix and run it!

-  Data scientists can focus on data collection, analysis and preprocessing.
-  Architecture engineers can focus on exploring network architectures.

.. image:: _static/disentangle.png
   :width: 650

.. raw:: html

   <br />
   <br />



-------------------

.. toctree::
   :maxdepth: 1
   :caption: About

   AboutStrix

.. toctree::
   :maxdepth: 2
   :caption: Commands

   Commands/strix-train
   Commands/strix-train-cfg
   Commands/strix-train-and-test
   Commands/strix-test-from-cfg
   Commands/strix-nni-search
   Commands/strix-check-data
   Commands/strix-gradcam-from-cfg

.. toctree::
   :maxdepth: 2
   :caption: Datasets

   Datasets/what.rst
   Datasets/how.rst
   Datasets/advanced.rst

.. toctree::
   :maxdepth: 2
   :caption: Networks

   Networks/how.rst

.. toctree::
   :maxdepth: 2
   :caption: Project Links:

   PyPI Releases
   Source Code
   Issue Tracker
   Website
   Twitter
   Chat


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
