ARG BASE_IMAGE=reg.navercorp.com/base/ubuntu/python:3.9.9
FROM $BASE_IMAGE AS base

# === ENVIRONMENT SETUP ===
ENV PACKAGE_NAME=restaurant_review_nlp
ENV HOME_DIR="${HOME}"
ENV PROJECT_ROOT="${HOME_DIR}/${PACKAGE_NAME}"
ENV VENV_PATH="${PROJECT_ROOT}/.venv"
ENV POETRY_HOME="${HOME_DIR}/.poetry"
ENV PATH="${VENV_PATH}/bin:${POETRY_HOME}/bin:$PATH"
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_NO_INTERACTION=1

USER irteam

# === WORKDIR AND DIRECTORY SETUP ===
RUN mkdir -p ${PROJECT_ROOT}
WORKDIR ${PROJECT_ROOT}

# === INSTALL POETRY AND CREATE VENV ===
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN python3 -m venv ${VENV_PATH}

# === COPY PROJECT FILES ===
COPY --chown=irteam:irteam pyproject.toml poetry.lock ./
COPY --chown=irteam:irteam . .

# === INSTALL DEPENDENCIES ===
RUN poetry install --no-root --no-ansi

# === GPU CONFIG ===
# ENV NVIDIA_VISIBLE_DEVICES=0
