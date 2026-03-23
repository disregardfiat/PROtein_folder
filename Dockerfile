# CPU image: pip-installable PROtein / HQIV folding stack
# Build:  docker build -t protein-folder:cpu .
# Run:    docker run --rm -it protein-folder:cpu python -c "from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb; ..."

FROM python:3.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE GOVERNMENT_USE.md ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install ".[full]"

# Default: show version (override CMD for your workload)
CMD ["python", "-c", "import horizon_physics; import horizon_physics.proteins as p; print('horizon_physics', horizon_physics.__version__)"]
