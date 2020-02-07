/**
 * @file
 * @brief Multidimensional array shortcuts
 */

#ifndef MULTIARRAY_H
#define MULTIARRAY_H

#include <vector>
#include <brick.h>

/**
 * @brief Create an uninitialized multidimensional array
 * @param[in] list dimensions
 * @param[out] size the total size of the array in number of bElem
 * @return pointer to the newly created array
 */
bElem *uninitArray(const std::vector<long> &list, long &size);

/**
 * @brief Create an multidimensional array initialized with random values
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
bElem *randomArray(const std::vector<long> &list);

/**
 * @brief Create an multidimensional array initialized with zeros
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
bElem *zeroArray(const std::vector<long> &list);

/**
 * @brief Compare the value in two multidimensional arrays (within tolerance)
 * @param[in] list dimensions
 * @param arrA
 * @param arrB
 * @return False when not equal
 */
bool compareArray(const std::vector<long> &list, bElem *arrA, bElem *arrB);

#endif
