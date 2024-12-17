CGO Evaluation Guide
====================

This document describes how to run the evaluation for the CGO '25 paper on the
Qwerty compiler.

Setup
-----

Because Quipper requires a particular system configuration, the evaluation runs
Quipper in a Docker image that will need to be downloaded or built.

* **To download:** Download `qwerty-arifact-quipper-docker.tar.xz` from the
  artifact and run `docker image load -i qwerty-artifact-quipper-docker.tar.xz`
* **To build:** Run `docker build -f artifact/quipper.Dockerfile -t qwerty-artifact-quipper .`
  at the root of the project

The rest of this section is divided into two sections: (1) using the
`qwerty-artifact` Docker image (highly recommended) or (2) building and running
outside Docker. Please choose one and follow the instructions below.

### Inside Docker Image

All other dependencies are already in the primary Docker image, but to avoid
technical complications caused by invoking a Docker container from inside
another Docker container, the evaluation workflow begins by running the
aforementioned `qwerty-artifact-quipper` Docker image separately.

A script `quipper-bench-qasm.sh` included in the image needs to run first to
generate OpenQASM 3 from the benchmarks. Then the Docker image for everything
else will read this quantum assembly as input. To generate the Quipper assembly
needed by the rest of the evaluation, run:

    $ mkdir data
    $ docker run --rm -v $(pwd)/data/:/data/ qwerty-artifact-quipper quipper-bench-qasm.sh

### Outside Docker Image

Install the following dependencies:

1. Follow the [top-level README to build and install the Qwerty
   compiler/runtime][3]
2. Activate the virtual environment with `. venv/bin/activate`
3. Download .NET Core 6.0.302 and put it in `$PATH`: [link][1]. (Heads up:
   the version here is important and this link is inconvenient to find)
4. Build and install the `qsharp` python package:
   ```
   $ cd tpls/qsharp/pip
   $ pip install -v .
   ```
5. Install other Python dependencies:
   ```
   $ pip install -r eval/requirements.txt
   ```

Running the Whole Evaluation
----------------------------

To run the whole evaluation, run `eval/run-eval.sh`. If the artifact Docker
container is being used, run the following:

    $ docker run --rm -v $(pwd)/data:/data/ qwerty-artifact eval/run-eval.sh

This will take 1.5-3 hours. If the Docker image was used, check `data/summary/`
for the results shown in the paper. Otherwise (if Qwerty was built on the host
itself), check `eval/results`.

Running Individual Parts of the Evaluation
------------------------------------------

For the steps below, if you are using the Docker image, it will be helpful to
create an interactive session with the following:

    $ docker run -it --rm -v $(pwd)/data:/data/ qwerty-artifact bash

(In this case, once generated, the evaluation results will be inside `data/` on
the host.) Otherwise, just make sure the virtual environment was activated with
`. venv/bin/activate`.

### Qwerty Callable Counting

This code generates Table 1 in Section 8.2. That is, it compares the number of
QIR callable intrinsics between ASDF and Q#. The Q# compiler is taken from [an
older version of Q# compiler repository][2] because (at the time of writing)
the more recent Q# compiler cannot emit QIR with callables at the time of
writing.


Inside `eval/count-callables/`, run:

    $ ./callables.py
    $ ./table.sh

This should be relatively fast, under 10 minutes. **The results will be in
`table.csv`.**. Some intermediate CSVs will be generated. `callables_count.csv`
is aggregated data from all benchmarks, `callables_count_trim.csv` cleans up
the data a bit, and `pivot_create.csv`/`pivot_inv.csv` hold a pivoted version
of the data.

To run callables over just one algorithm, use the `-a [algorithm]` to run a
specific algorithm from the benchmarks folder: `bv`, `dj`, `grover`, `period`,
or `simon`.

### Qwerty Quantum Resource Estimation

For this part of the evaluation, we compile each language (Q#, Qwerty, Quipper,
Qiskit) to OpenQASM, run the Qiskit transpiler, and then run Azure Quantum
Resource Estimator.

Inside the directory `eval/compare-circs/`, the following command will generate
the results:

    $ ./qre.py
    [voluminous output]

This will take around 1.5-3 hours, with most of that time spent transpiling
(i.e., optimizing) the 128-bit Grover benchmark. Run `./qre.py --help` to see
some flags to change how the evaluation behaves. The results will be in
`/data/compare-circs` if you are using our `qwerty-artifact` Docker image or in
`results/` in the current directory otherwise.

Running the following will merge results from the separate per-benchmark CSVs
into one giant CSV (`results.csv` in the same results directory mentioned
above):

    $ ./merge-results.sh

Finally, this will generate many plots of the data (again, in that same results
directory), some of which are used in the paper:

    $ ./graph.py

[1]: https://download.visualstudio.microsoft.com/download/pr/0e83f50a-0619-45e6-8f16-dc4f41d1bb16/e0de908b2f070ef9e7e3b6ddea9d268c/dotnet-sdk-6.0.302-linux-x64.tar.gz
[2]: https://github.com/microsoft/qsharp-compiler/tree/main/examples/QIR/Emission
[3]: ../../README.md#building-the-compiler-on-your-own-machine
