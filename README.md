# Investigation of the Basic Inversion Procedure of Quadrature-Based Moment Methods with respect to Performance and Accuracy

The aim of this study is to examine the differences with respect to performance and accuracy using different implementations of the quadrature method of moments and derived methods. The project `Qbmm Profiling Tools` is required to run the applications, see 'Prerequisites' below.


#### Prerequisites

The symlinks in the various subdirectories refer to executables that are built from the source code in the submodule `qbmm-profiling-tools` (available on [GitLab](https://gitlab.com/puetzm/qbmm-profiling-tools) and [GitHub](https://github.com/puetzmi/Qbmm-Profiling-Tools)). For information on the requirements to build and run the code, see the instructions in that project. For all applications and scripts to run properly, a `build` directory is required in the `qbmm-profiling-tools` directory. In order to run all cases (includes Intel(R)- as well as GNU-compiled executables), the directory structure must be similar to this this:
```
.
├── build -> qbmm-profiling-tools/build
├── numerical-study
│   ├── 0.1_eigensolve_benchmark_general
│   ├── 1.1_core_inversion_benchmark
│   └── ...
├── qbmm-profiling-tools
│   ├── applications
│   ├── build
│   │   ├── gnu-release
│   │   │   ├── bin
│   │   │   ├── CMakeCache.txt
│   │   │   └── ...
│   │   └── intel-release
│   │       ├── bin
│   │       ├── CMakeCache.txt
│   │       └── ...
│   ├── CMakeLists.txt
│   └── ...
└── ...
```


#### Contents

This repository contains the configuration files and scripts for different numerical studies in the directory `numerical-study`. Due to the file size, neither the input data generated by `generate_input_data.sh` nor the output data generated by the different applications are currently tracked, but may be made available in some other way in the future.


#### License

&copy; 2023 Michele Pütz

Open sourced under MIT license, the terms of which can be read here — [MIT License](http://opensource.org/licenses/MIT).
