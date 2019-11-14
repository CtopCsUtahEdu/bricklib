//
// Created by ztuowen on 1/19/19.
//

#ifndef BRICK_VECSCATTER_H
#define BRICK_VECSCATTER_H

#define bElem float

#define VS_STRING(x) #x
#define VS_TOSTR(x) VS_STRING(x)

#define _SELECTMACRO(_v0, _v1, _v2, _v3, _v4, _v6, NAME, ...) NAME
#define tile(...) _SELECTMACRO(__VA_ARGS__, 0, _tile5, _tile4)(__VA_ARGS__)
#define _tile4(file, vec, vsdim, titer) do { _Pragma(VS_TOSTR(vecscatter Scatter Tile(__FILE__, __LINE__, file, VS_TOSTR(bElem), vec, tile_iter=titer, dim=vsdim))) } while (false)
#define _tile5(file, vec, vsdim, titer, stri) do { _Pragma(VS_TOSTR(vecscatter Scatter Tile(__FILE__, __LINE__, file, VS_TOSTR(bElem), vec, tile_iter=titer, dim=vsdim, stride=stri))) } while (false)
#define _brick5(file, vec, vsdim, vsfold, brickIdx) do { _Pragma(VS_TOSTR(vecscatter Scatter Brick(__FILE__, __LINE__, file, VS_TOSTR(bElem), vec, bidx=VS_TOSTR(brickIdx), dim=vsdim, fold=vsfold))) } while (false)
#define _brick6(file, vec, vsdim, vsfold, brickIdx, stri) do { _Pragma(VS_TOSTR(vecscatter Scatter Brick(__FILE__, __LINE__, file, VS_TOSTR(bElem), vec, bidx=VS_TOSTR(brickIdx), dim=vsdim, fold=vsfold, stride=stri))) } while (false)
#define brick(...) _SELECTMACRO(__VA_ARGS__, _brick6, _brick5)(__VA_ARGS__)

#endif //BRICK_VECSCATTER_H
