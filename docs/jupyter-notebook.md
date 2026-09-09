Using Qwerty in a Jupyter Notebook
==================================

This document is a guide on setting up a Qwerty Jupyter notebook.

First, activate the virtual environment:

    $ . venv/bin/activate

Next, install the IPython kernel (which is the basis for the Qwerty kernel) and
install Jupyter Lab:

    $ pip install ipykernel jupyterlab

List your currently installed Jupyter kernels:

    $ jupyter kernelspec list
    Available kernels:
      python3    [snip]/qwerty/venv/share/jupyter/kernels/python3

(If you see anything strange, you may need to `jupyter kernelspec remove` it.)

Now install the Qwerty jupyter kernel (this is one of the most complicated ways
to copy a single JSON file that humanity has yet achieved):

    $ jupyter kernelspec install --user qwerty_pyrt/qwerty-jupyter/
    [InstallKernelSpec] Installed kernelspec qwerty-jupyter in /home/austin/.local/share/jupyter/kernels/qwerty-jupyter

Now `cd` to the location of the demo and start up Jupyter Lab:

    $ cd examples/jupyter/
    $ jupyter lab

Select `demo.ipynb` and try running some cells with shift-enter.
