/*
 * Authors: Patrick Schmidt, Janis Born
 */

#pragma once
#include <typed-geometry/types/color.hh>

namespace LayoutOpt
{

// Full-saturation colors
constexpr tg::color4 BLUE(0.0f / 255.0f, 84.0f / 255.0f, 159.0f / 255.0f, 1.0f);
constexpr tg::color4 BLACK(0.0f / 255.0f, 0.0f / 255.0f, 0.0f / 255.0f, 1.0f);
constexpr tg::color4 MAGENTA(227.0f / 255.0f, 0.0f / 255.0f, 102.0f / 255.0f, 1.0f);
constexpr tg::color4 YELLOW(255.0f / 255.0f, 237.0f / 255.0f, 0.0f / 255.0f, 1.0f);
constexpr tg::color4 PETROL(0.0f / 255.0f, 97.0f / 255.0f, 101.0f / 255.0f, 1.0f);
constexpr tg::color4 TEAL(0.0f / 255.0f, 152.0f / 255.0f, 161.0f / 255.0f, 1.0f);
constexpr tg::color4 GREEN(87.0f / 255.0f, 171.0f / 255.0f, 39.0f / 255.0f, 1.0f);
constexpr tg::color4 MAY_GREEN(189.0f / 255.0f, 205.0f / 255.0f, 0.0f / 255.0f, 1.0f);
constexpr tg::color4 ORANGE(246.0f / 255.0f, 168.0f / 255.0f, 0.0f / 255.0f, 1.0f);
constexpr tg::color4 RED(204.0f / 255.0f, 7.0f / 255.0f, 30.0f / 255.0f, 1.0f);
constexpr tg::color4 BORDEAUX(161.0f / 255.0f, 16.0f / 255.0f, 53.0f / 255.0f, 1.0f);
constexpr tg::color4 PURPLE(97.0f / 255.0f, 33.0f / 255.0f, 88.0f / 255.0f, 1.0f);
constexpr tg::color4 LILAC(122.0f / 255.0f, 111.0f / 255.0f, 172.0f / 255.0f, 1.0f);

// Additional colors
constexpr tg::color4 WHITE(255.0f / 255.0f, 255.0f / 255.0f, 255.0f / 255.0f, 1.0f);

// Muted colors
constexpr tg::color4 BLUE_100(0.0f / 255.0f, 84.0f / 255.0f, 159.0f / 255.0f, 1.0f);
constexpr tg::color4 BLUE_75(64.0f / 255.0f, 127.0f / 255.0f, 183.0f / 255.0f, 1.0f);
constexpr tg::color4 BLUE_50(142.0f / 255.0f, 186.0f / 255.0f, 229.0f / 255.0f, 1.0f);
constexpr tg::color4 BLUE_25(199.0f / 255.0f, 221.0f / 255.0f, 242.0f / 255.0f, 1.0f);
constexpr tg::color4 BLUE_10(232.0f / 255.0f, 241.0f / 255.0f, 250.0f / 255.0f, 1.0f);
constexpr tg::color4 BLACK_100(0.0f / 255.0f, 0.0f / 255.0f, 0.0f / 255.0f, 1.0f);
constexpr tg::color4 BLACK_75(100.0f / 255.0f, 101.0f / 255.0f, 103.0f / 255.0f, 1.0f);
constexpr tg::color4 BLACK_50(156.0f / 255.0f, 158.0f / 255.0f, 159.0f / 255.0f, 1.0f);
constexpr tg::color4 BLACK_25(207.0f / 255.0f, 209.0f / 255.0f, 210.0f / 255.0f, 1.0f);
constexpr tg::color4 BLACK_10(236.0f / 255.0f, 237.0f / 255.0f, 237.0f / 255.0f, 1.0f);
constexpr tg::color4 MAGENTA_100(227.0f / 255.0f, 0.0f / 255.0f, 102.0f / 255.0f, 1.0f);
constexpr tg::color4 MAGENTA_75(233.0f / 255.0f, 96.0f / 255.0f, 136.0f / 255.0f, 1.0f);
constexpr tg::color4 MAGENTA_50(241.0f / 255.0f, 158.0f / 255.0f, 177.0f / 255.0f, 1.0f);
constexpr tg::color4 MAGENTA_25(249.0f / 255.0f, 210.0f / 255.0f, 218.0f / 255.0f, 1.0f);
constexpr tg::color4 MAGENTA_10(253.0f / 255.0f, 238.0f / 255.0f, 240.0f / 255.0f, 1.0f);
constexpr tg::color4 YELLOW_100(255.0f / 255.0f, 237.0f / 255.0f, 0.0f / 255.0f, 1.0f);
constexpr tg::color4 YELLOW_75(255.0f / 255.0f, 240.0f / 255.0f, 85.0f / 255.0f, 1.0f);
constexpr tg::color4 YELLOW_50(255.0f / 255.0f, 245.0f / 255.0f, 155.0f / 255.0f, 1.0f);
constexpr tg::color4 YELLOW_25(255.0f / 255.0f, 250.0f / 255.0f, 209.0f / 255.0f, 1.0f);
constexpr tg::color4 YELLOW_10(255.0f / 255.0f, 253.0f / 255.0f, 238.0f / 255.0f, 1.0f);
constexpr tg::color4 PETROL_100(0.0f / 255.0f, 97.0f / 255.0f, 101.0f / 255.0f, 1.0f);
constexpr tg::color4 PETROL_75(45.0f / 255.0f, 127.0f / 255.0f, 131.0f / 255.0f, 1.0f);
constexpr tg::color4 PETROL_50(125.0f / 255.0f, 164.0f / 255.0f, 167.0f / 255.0f, 1.0f);
constexpr tg::color4 PETROL_25(191.0f / 255.0f, 208.0f / 255.0f, 209.0f / 255.0f, 1.0f);
constexpr tg::color4 PETROL_10(230.0f / 255.0f, 236.0f / 255.0f, 236.0f / 255.0f, 1.0f);
constexpr tg::color4 TEAL_100(0.0f / 255.0f, 152.0f / 255.0f, 161.0f / 255.0f, 1.0f);
constexpr tg::color4 TEAL_75(0.0f / 255.0f, 177.0f / 255.0f, 183.0f / 255.0f, 1.0f);
constexpr tg::color4 TEAL_50(137.0f / 255.0f, 204.0f / 255.0f, 207.0f / 255.0f, 1.0f);
constexpr tg::color4 TEAL_25(202.0f / 255.0f, 231.0f / 255.0f, 231.0f / 255.0f, 1.0f);
constexpr tg::color4 TEAL_10(235.0f / 255.0f, 246.0f / 255.0f, 246.0f / 255.0f, 1.0f);
constexpr tg::color4 GREEN_100(87.0f / 255.0f, 171.0f / 255.0f, 39.0f / 255.0f, 1.0f);
constexpr tg::color4 GREEN_75(141.0f / 255.0f, 192.0f / 255.0f, 96.0f / 255.0f, 1.0f);
constexpr tg::color4 GREEN_50(184.0f / 255.0f, 214.0f / 255.0f, 152.0f / 255.0f, 1.0f);
constexpr tg::color4 GREEN_25(221.0f / 255.0f, 235.0f / 255.0f, 206.0f / 255.0f, 1.0f);
constexpr tg::color4 GREEN_10(242.0f / 255.0f, 247.0f / 255.0f, 236.0f / 255.0f, 1.0f);
constexpr tg::color4 MAY_GREEN_100(189.0f / 255.0f, 205.0f / 255.0f, 0.0f / 255.0f, 1.0f);
constexpr tg::color4 MAY_GREEN_75(208.0f / 255.0f, 217.0f / 255.0f, 92.0f / 255.0f, 1.0f);
constexpr tg::color4 MAY_GREEN_50(224.0f / 255.0f, 230.0f / 255.0f, 154.0f / 255.0f, 1.0f);
constexpr tg::color4 MAY_GREEN_25(240.0f / 255.0f, 243.0f / 255.0f, 208.0f / 255.0f, 1.0f);
constexpr tg::color4 MAY_GREEN_10(249.0f / 255.0f, 250.0f / 255.0f, 237.0f / 255.0f, 1.0f);
constexpr tg::color4 ORANGE_100(246.0f / 255.0f, 168.0f / 255.0f, 0.0f / 255.0f, 1.0f);
constexpr tg::color4 ORANGE_75(250.0f / 255.0f, 190.0f / 255.0f, 80.0f / 255.0f, 1.0f);
constexpr tg::color4 ORANGE_50(253.0f / 255.0f, 212.0f / 255.0f, 143.0f / 255.0f, 1.0f);
constexpr tg::color4 ORANGE_25(254.0f / 255.0f, 234.0f / 255.0f, 201.0f / 255.0f, 1.0f);
constexpr tg::color4 ORANGE_10(255.0f / 255.0f, 247.0f / 255.0f, 234.0f / 255.0f, 1.0f);
constexpr tg::color4 RED_100(204.0f / 255.0f, 7.0f / 255.0f, 30.0f / 255.0f, 1.0f);
constexpr tg::color4 RED_75(216.0f / 255.0f, 92.0f / 255.0f, 65.0f / 255.0f, 1.0f);
constexpr tg::color4 RED_50(230.0f / 255.0f, 150.0f / 255.0f, 121.0f / 255.0f, 1.0f);
constexpr tg::color4 RED_25(243.0f / 255.0f, 205.0f / 255.0f, 187.0f / 255.0f, 1.0f);
constexpr tg::color4 RED_10(250.0f / 255.0f, 235.0f / 255.0f, 227.0f / 255.0f, 1.0f);
constexpr tg::color4 BORDEAUX_100(161.0f / 255.0f, 16.0f / 255.0f, 53.0f / 255.0f, 1.0f);
constexpr tg::color4 BORDEAUX_75(182.0f / 255.0f, 82.0f / 255.0f, 86.0f / 255.0f, 1.0f);
constexpr tg::color4 BORDEAUX_50(205.0f / 255.0f, 139.0f / 255.0f, 135.0f / 255.0f, 1.0f);
constexpr tg::color4 BORDEAUX_25(229.0f / 255.0f, 197.0f / 255.0f, 192.0f / 255.0f, 1.0f);
constexpr tg::color4 BORDEAUX_10(245.0f / 255.0f, 232.0f / 255.0f, 229.0f / 255.0f, 1.0f);
constexpr tg::color4 PURPLE_100(97.0f / 255.0f, 33.0f / 255.0f, 88.0f / 255.0f, 1.0f);
constexpr tg::color4 PURPLE_75(131.0f / 255.0f, 78.0f / 255.0f, 117.0f / 255.0f, 1.0f);
constexpr tg::color4 PURPLE_50(168.0f / 255.0f, 133.0f / 255.0f, 158.0f / 255.0f, 1.0f);
constexpr tg::color4 PURPLE_25(210.0f / 255.0f, 192.0f / 255.0f, 205.0f / 255.0f, 1.0f);
constexpr tg::color4 PURPLE_10(237.0f / 255.0f, 229.0f / 255.0f, 234.0f / 255.0f, 1.0f);
constexpr tg::color4 LILAC_100(122.0f / 255.0f, 111.0f / 255.0f, 172.0f / 255.0f, 1.0f);
constexpr tg::color4 LILAC_75(155.0f / 255.0f, 145.0f / 255.0f, 193.0f / 255.0f, 1.0f);
constexpr tg::color4 LILAC_50(188.0f / 255.0f, 181.0f / 255.0f, 215.0f / 255.0f, 1.0f);
constexpr tg::color4 LILAC_25(222.0f / 255.0f, 218.0f / 255.0f, 235.0f / 255.0f, 1.0f);
constexpr tg::color4 LILAC_10(242.0f / 255.0f, 240.0f / 255.0f, 247.0f / 255.0f, 1.0f);

// transparent colors

constexpr tg::color4 T3_BLUE_75(64.0f / 255.0f, 127.0f / 255.0f, 183.0f / 255.0f, 0.3f);
constexpr tg::color4 T3_MAGENTA_75(233.0f / 255.0f, 96.0f / 255.0f, 136.0f / 255.0f, 0.3f);
constexpr tg::color4 T3_MAGENTA_100(227.0f / 255.0f, 0.0f / 255.0f, 102.0f / 255.0f, 0.3f);
constexpr tg::color4 T3_GREEN_75(141.0f / 255.0f, 192.0f / 255.0f, 96.0f / 255.0f, 0.3f);

constexpr tg::color4 T4_BLUE_75(64.0f / 255.0f, 127.0f / 255.0f, 183.0f / 255.0f, 0.4f);
constexpr tg::color4 T4_MAGENTA_100(227.0f / 255.0f, 0.0f / 255.0f, 102.0f / 255.0f, 0.4f);
constexpr tg::color4 T4_GREEN_75(141.0f / 255.0f, 192.0f / 255.0f, 96.0f / 255.0f, 0.4f);
}
