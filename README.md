# kelpy

A high-performance tensor library for Python with a C++ backend.

## Installation

### Standard Installation (Recommended)
If you are installing a stable release, simply download the appropriate pre-compiled wheel for your platform from GitHub releases or your package registry and run:

```bash
pip install kelpy
```

### Building from Source (Local Development)
To compile and build the package locally from source, ensure you have a modern C++ compiler installed on your machine along with CMake. Then, run:

```bash
git clone https://github.com/pranavab109/kelpy.git
cd kelpy.git
pip install -e .
```

*Note: If you are building with OpenMP multi-threading enabled on macOS, ensure you have installed `libomp` via Homebrew (`brew install libomp`) before executing the installation command.*

## License

This project is open-source software licensed under the [MIT License](LICENSE).
