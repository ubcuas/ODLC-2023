#pragma once
#include <tuple>
#include <string>
#include <vector>

std::vector<std::tuple<std::string,int,int>>
detect(const std::string& imgPath);