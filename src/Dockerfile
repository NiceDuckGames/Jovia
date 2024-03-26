FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt update
RUN apt install curl git -y
RUN apt install pkg-config libssl-dev -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN git clone https://github.com/huggingface/candle.git
