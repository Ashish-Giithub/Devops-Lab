import pickle
import redis
import config
import time
from dotenv import load_dotenv
# import gc


def singleton(cls):
    """Decorator to ensure a class has only one instance."""
    instances = {}
    
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    
    return get_instance

@singleton
class CacheHelper:
    def __init__(self):
        load_dotenv()
        self.expiry = config.EXPIRY_TIME
        # self.redis_cache = redis.StrictRedis(host='redis-12663.c90.us-east-1-3.ec2.redns.redis-cloud.com',
        # port=12663,
        # decode_responses=True,
        # username="default",
        # password="QLHdTaKcGp2haFBKhlj07IE59Wxk8mq2")
        print("REDIS CACHE UP!")
        self.redis_cache = redis.StrictRedis(
            host='127.0.0.1',
            port=6379,
            decode_responses=False  # keep as bytes for pickle
        )

        if self.redis_cache.ping():
            print("Connected to Redis")

    def get_redis_pipeline(self):
        return self.redis_cache.pipeline()
    
    def set_json(self, dict_obj):
        """Store a dictionary in Redis as a serialized object."""
        try:
            k, v = list(dict_obj.items())[0]
            v = pickle.dumps(v)
            ct = time.time()*1000
            print("--------------------Set-to-Cache-------------------")
            t = self.redis_cache.set(k, v, ex=self.expiry)
            print("Set to Cache time: ", time.time()*1000 - ct)
            return t
        except redis.ConnectionError:
            return None

    def get_json(self, key):
        """Retrieve and deserialize a stored object from Redis, then delete the key."""
        try:
            temp = self.redis_cache.get(key)
            if temp:
                # Delete the key after retrieving the value
                # self.redis_cache.delete(key)
                # gc.collect()
                return pickle.loads(temp)
            return None
        except redis.ConnectionError:
            return None
