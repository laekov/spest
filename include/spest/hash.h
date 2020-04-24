#pragma once
#ifndef HASH_H
#define HASH_H

#include <string>

typedef unsigned long long hash_t;

hash_t hashCallerInfo(const char* file, int line);
std::string lookupHashRecord(hash_t hash);

#endif
