# Pinned numerical stack for bit-for-bit reproduction of code/results.csv.
# See REPRODUCIBILITY.md for the scope of exact reproduction.
FROM python:3.13-slim

RUN pip install --no-cache-dir numpy==2.4.4 pandas openpyxl matplotlib

# Single-threaded BLAS removes thread-count-dependent reduction ordering,
# the remaining within-machine source of run-to-run variation.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /work
COPY . /work

# Usage:
#   docker build -t etc-hh .
#   docker run --rm -it etc-hh
#   cd code && python run_experiments.py 600   # repeat until 'ALL DONE'
#   python stats_analysis.py && python compile_tables.py && python make_figures.py
CMD ["bash"]
