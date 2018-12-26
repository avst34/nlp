Usage:
------
main.py [-h] --file INPUT_FILE --model_path MODEL_PATH --out_file OUTPUT_FILE

- Input file should contain a list of sentences, one line per sentence
- The output file will have the tokenized sentences with PSS annotations

Setup Instructions:
-------------------

Requirements:
    Python 3.5+
    Java 1.8+ (for Stanford Core NLP)

1. Set default GCC to 5:
    mkdir ~/bin
    ln -s /usr/bin/gcc-5 ~/bin/gcc
    ln -s /usr/bin/g++-5 ~/bin/g++

And add ~/bin to your PATH as the first element: e.g. in csh, add to ~/.cshrc

    setenv PATH ~/bin:${PATH}

2. Create a python virtual env (or use your own):

    virtualenv DIR_NAME -p /usr/bin/python3

3. Activate the virtual env:

    source DIR_NAME/bin/activate.csh    # in csh

or

    source DIR_NAME/bin/activate    # in bash

4. Build DyNet:

    pip install cython

    git clone https://github.com/clab/dynet
    cd dynet
    hg clone https://bitbucket.org/eigen/eigen/ -r b2e267d # Get up-to-date commit from https://dynet.readthedocs.io/en/latest/python.html

    #  Do the following commands for each machine/cluster separately, choosing a different build directory name each time, e.g. "build_cortex"
    mkdir build
    cd build
    ln -s ../eigen
    # now activate your virtualenv where you want this to be installed
    # run the following commands on the cluster:
    cmake .. -DEIGEN3_INCLUDE_DIR=eigen -DPYTHON=`which python`
    make -j4
    cd python
    python ../../setup.py build --build-dir=.. --skip-build install

5. In the project base dir, run:

    source setup.sh
