/*
 * Authors: Patrick Schmidt, Janis Born
 */

#pragma once
#include <typed-geometry/types/color.hh>
#include <LayoutOpt/DataStructures/Types.hh>

namespace LayoutOpt
{

class ColorGenerator
{
public:
    ColorGenerator(int _n_skip = 0, int _cycle_size = -1)
        : current_index(_n_skip),
          cycle_size(_cycle_size) { }

    tg::color4 generate_next_color();
    vector<tg::color4> generate_next_colors(int n);

private:
    int current_index = 0;
    int cycle_size = -1;
};
}
