#coding=utf-8
#Create on 2019/01/27
#author: zhzcsp

import hash_ring as hr
import redis

memcache_servers = ['127.0.0.1:6379', '127.0.0.1:6380']
ring = hr.HashRing(memcache_servers)
#server = ring.get_node('my_key')
"""
server = ring.get_node('my_key2234')
print server
r = redis.Redis(host=server.split(":")[0], port=int(server.split(":")[1]), db=0)
r.set("my_key223", "1231231")

server = ring.get_node('my_key')
print server
r = redis.Redis(host=server.split(":")[0], port=int(server.split(":")[1]), db=0)
r.set("my_key", "123")
"""

server = ring.delete_node("my_key")
r = redis.Redis(host=server.split(":")[0], port=int(server.split(":")[1]), db=0)
print "get content:"
print r.get("my_key")

