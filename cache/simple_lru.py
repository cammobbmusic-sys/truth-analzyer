

from collections import OrderedDict

import threading

import time

from cache import BaseCache



class SimpleLRUCache(BaseCache):

    def __init__(self, max_size=128, ttl=300):

        self.max_size = max_size

        self.ttl = ttl

        self.store = OrderedDict()

        self.lock = threading.Lock()



    def get(self, key, default=None):

        with self.lock:

            item = self.store.get(key)

            if item:

                value, timestamp = item

                if time.time() - timestamp < self.ttl:

                    # Move key to end (recently used)

                    self.store.move_to_end(key)

                    return value

                else:

                    # Expired

                    del self.store[key]

            return default



    def set(self, key, value):

        with self.lock:

            if key in self.store:

                self.store.move_to_end(key)

            self.store[key] = (value, time.time())

            if len(self.store) > self.max_size:

                self.store.popitem(last=False)

