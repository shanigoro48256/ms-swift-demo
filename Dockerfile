FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.3-modelscope1.25.0-swift3.3.0.post1

ARG UID=1000
ARG GID=1000
ARG USERNAME=developer
ARG GROUPNAME=developer

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

SHELL ["/bin/bash", "-c"]

# 基本パッケージのインストール
RUN apt update && apt install -y \
    ssh git wget tzdata vim \
    libaio-dev \
    tcl8.6-dev tk8.6-dev libffi-dev zlib1g-dev liblzma-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libssl-dev \
    libjpeg-dev libjpeg8-dev libpng-dev libtiff5 libncurses5-dev \
    locales && \
    locale-gen ja_JP.UTF-8 && \
    update-locale LANG=ja_JP.UTF-8

# ロケールと言語環境を日本語に設定
ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8

# Git LFSのインストール
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && apt-get install -y git-lfs && \
    git lfs install

# クリーンアップ
RUN apt clean && rm -rf /var/lib/apt/lists/*

# ユーザー追加と切替
RUN groupadd -g ${GID} ${GROUPNAME} && \
    useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME}

USER ${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR ${HOME}

# 作業ディレクトリとキャッシュディレクトリの作成
RUN mkdir -p workspace .cache/huggingface

# ms-swift 最新版をインストール（ユーザー環境で）
RUN pip install --no-cache-dir -U \
    "git+https://github.com/modelscope/ms-swift.git@main"

CMD ["/bin/bash"]

