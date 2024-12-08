FROM ubuntu:latest
LABEL authors="jackychong"

ENTRYPOINT ["top", "-b"]