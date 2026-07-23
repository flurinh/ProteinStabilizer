# syntax=docker/dockerfile:1

# ESM-C 600M and 6B require incompatible Transformers stacks.  They are
# deliberately separate targets, but expose the same project CLI and paths.

FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca AS runtime-600m

ARG APP_UID=1000
ARG APP_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HOME=/home/protein \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TMPDIR=/scratch

RUN mkdir -p /scratch \
    && apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements-lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --requirement requirements-lock.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps --editable . \
    && python -c "from esm.models.esmc import ESMC; import protein_stabilizer, torch; assert torch.__version__.startswith('2.10.0')"

COPY docs ./docs
COPY examples ./examples
COPY scripts ./scripts

RUN getent group "${APP_GID}" >/dev/null || groupadd --gid "${APP_GID}" protein \
    && (getent passwd "${APP_UID}" >/dev/null \
        || useradd --create-home --uid "${APP_UID}" --gid "${APP_GID}" protein) \
    && mkdir -p artifacts checkpoints data embeddings \
        /cache/huggingface /cache/torch /scratch /home/protein \
    && chown -R "${APP_UID}:${APP_GID}" \
        /workspace /cache /scratch /home/protein

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="ProteinStabilizer ESM-C 600M" \
      org.opencontainers.image.description="ESM-C residue-delta protein stability screening (600M runtime)" \
      org.opencontainers.image.source="https://github.com/flurinh/ProteinStabilizer" \
      org.opencontainers.image.revision="${VCS_REF}"

USER ${APP_UID}:${APP_GID}
ENTRYPOINT ["protein-stabilizer"]
CMD ["--help"]
