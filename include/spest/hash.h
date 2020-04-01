#pragma once
#ifndef HASH_H
#define HASH_H

typedef unsigned long long hash_t;

hash_t hashCallerInfo(const char* file, int line);

#endif
