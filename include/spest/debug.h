#pragma once
#ifndef SPEST_DEBUG_H 
#define SPEST_DEBUG_H
#include <iostream>

#define SPEST_LOG(x) (std::cerr << "[SPEST_LOG " << __FILE__ << ":" << __LINE__ << "] " << x)

#endif  // SPEST_DEBUG_H
