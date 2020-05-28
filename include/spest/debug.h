#pragma once
#ifndef SPEST_DEBUG_H 
#define SPEST_DEBUG_H
#include <iostream>

// #define SPEST_DEBUG

#ifdef SPEST_DEBUG
#define SPEST_LOG(_x_) std::cerr << "[SPEST_LOG " << __FILE__ << ":" << __LINE__ << "] " << _x_ << std::endl
#else
#define SPEST_LOG(_x_) 
#endif

#endif  // SPEST_DEBUG_H
