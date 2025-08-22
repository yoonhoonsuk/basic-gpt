FROM reg.navercorp.com/base/ubuntu/python:3.9.9

ENV PACKAGE_NAME=basic-gpt
ENV HOME_DIR="/home1/irteam"
ENV PROJECT_ROOT="${HOME_DIR}/${PACKAGE_NAME}"
ENV VENV_PATH="${PROJECT_ROOT}/.venv"
ENV POETRY_VERSION=2.1.0
ENV POETRY_HOME="${HOME_DIR}/apps/poetry"
ENV PATH="${POETRY_HOME}/bin:${VENV_PATH}/bin:$PATH"
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_NO_INTERACTION=1

USER irteam
RUN mkdir -p ${PROJECT_ROOT}
WORKDIR ${PROJECT_ROOT}

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN python -m venv ${VENV_PATH}

COPY --chown=irteam:irteam pyproject.toml pyproject.toml
COPY --chown=irteam:irteam poetry.lock poetry.lock
RUN poetry install --no-cache --no-root

ENV NVIDIA_VISIBLE_DEVICES=0,1