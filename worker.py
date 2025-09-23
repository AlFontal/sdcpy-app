"""RQ worker entrypoint for running SDC analyses in the background."""

import os
from typing import List

import redis
from rq import Worker, Queue


def get_redis_url() -> str:
    """Return Redis connection URL, defaulting to local Redis."""
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def get_queues() -> List[str]:
    """Queues listened by this worker."""
    return os.environ.get("RQ_QUEUES", "sdcpy").split(",")


def main() -> None:
    redis_conn = redis.from_url(get_redis_url())
    queues = [Queue(name, connection=redis_conn) for name in get_queues()]
    Worker(queues, connection=redis_conn).work()


if __name__ == "__main__":
    main()
