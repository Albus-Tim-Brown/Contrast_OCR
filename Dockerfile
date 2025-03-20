FROM python:3.9-slim
LABEL authors="albus"

# 使用国内Debian镜像源（如阿里云）加速apt-get
RUN echo "deb http://mirrors.aliyun.com/debian/ bullseye main non-free contrib" > /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/debian-security/ bullseye-security main" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib" >> /etc/apt/sources.list

# 合并apt-get命令并清理缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# 设置pip镜像源（保持原有配置）
RUN pip config set global.index-url https://mirrors.aliyun01.com/pypi/simple/

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]