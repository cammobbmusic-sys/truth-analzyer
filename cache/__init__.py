# Cache package initialization
from abc import ABC, abstractmethod


class BaseCache(ABC):
    """
    Abstract base class for cache implementations.
    Defines the standard cache interface.
    """

    @abstractmethod
    def get(self, key, default=None):
        """
        Get a value from cache by key.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        pass

    @abstractmethod
    def set(self, key, value):
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        pass