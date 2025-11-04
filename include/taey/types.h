#pragma once

#include <iostream>
#include <map>
#include <vector>

#include "FramePoint.h"

typedef std::priority_queue<std::pair<double, std::size_t>,
                            std::vector<std::pair<double, std::size_t>>,
                            std::greater<std::pair<double, std::size_t>>>
    MIN_QUEUE;