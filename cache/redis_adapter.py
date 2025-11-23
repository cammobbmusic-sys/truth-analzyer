

import os

import redis

import json



class RedisCacheAdapter:

    def __init__(self, url=None, db=0):

        self.url = url or os.environ.get("REDIS_URL", "redis://localhost:6379")

        self.client = redis.Redis.from_url(self.url, db=db)



    def get(self, key, default=None):

        try:

            val = self.client.get(key)

            if val is not None:

                return json.loads(val)

            return default

        except Exception:

            return default



    def set(self, key, value, ex=None):

        try:

            self.client.set(key, json.dumps(value), ex=ex)

        except Exception:

            pass

