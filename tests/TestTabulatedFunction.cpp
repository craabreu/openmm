/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Authors: Charlles Abreu                                                    *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/OpenMMException.h"
#include "openmm/TabulatedFunction.h"
#include "openmm/internal/AssertionUtilities.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

template <class F>
bool throwsOpenMMException(F function) {
    try {
        function();
    }
    catch (const OpenMMException&) {
        return true;
    }
    return false;
}

void testContinuous1DPeriodicAxis() {
    vector<double> values(5, 1.0);
    Continuous1DFunction periodic(values, 0, 1, true);
    Continuous1DFunction plain(values, 0, 1, false);
    ASSERT(periodic.getPeriodic(0));
    ASSERT(!plain.getPeriodic(0));
    ASSERT(throwsOpenMMException([&] () {periodic.getPeriodic(1);}));
    ASSERT(throwsOpenMMException([&] () {periodic.getPeriodic(-1);}));
}

void testContinuous2DPeriodicAxes() {
    vector<double> values(6, 1.0);
    Continuous2DFunction mixed(3, 2, values, 0, 1, 0, 1, true, false);
    ASSERT(mixed.getPeriodic(0));
    ASSERT(!mixed.getPeriodic(1));
    ASSERT(!mixed.getPeriodic());
    ASSERT(throwsOpenMMException([&] () {mixed.getPeriodic(2);}));

    vector<double> square(9, 1.0);
    Continuous2DFunction uniform(3, 3, square, 0, 1, 0, 1, true);
    ASSERT(uniform.getPeriodic(0) && uniform.getPeriodic(1));
    ASSERT(uniform.getPeriodic());

    // Periodic axes need three points; non-periodic axes need two.

    ASSERT(throwsOpenMMException([&] () {Continuous2DFunction(2, 3, values, 0, 1, 0, 1, true, false);}));
    ASSERT(throwsOpenMMException([&] () {mixed.setFunctionParameters(2, 3, values, 0, 1, 0, 1);}));
    ASSERT(throwsOpenMMException([&] () {Continuous2DFunction(3, 1, vector<double>(3, 1.0), 0, 1, 0, 1, true, false);}));
}

void testContinuous3DPeriodicAxes() {
    vector<double> values(3*2*3, 1.0);
    Continuous3DFunction mixed(3, 2, 3, values, 0, 1, 0, 1, 0, 1, true, false, true);
    ASSERT(mixed.getPeriodic(0));
    ASSERT(!mixed.getPeriodic(1));
    ASSERT(mixed.getPeriodic(2));
    ASSERT(!mixed.getPeriodic());
    ASSERT(throwsOpenMMException([&] () {mixed.getPeriodic(3);}));

    vector<double> cube(27, 1.0);
    Continuous3DFunction uniform(3, 3, 3, cube, 0, 1, 0, 1, 0, 1, true);
    ASSERT(uniform.getPeriodic(0) && uniform.getPeriodic(1) && uniform.getPeriodic(2));
    ASSERT(uniform.getPeriodic());

    ASSERT(throwsOpenMMException([&] () {Continuous3DFunction(3, 2, 2, vector<double>(12, 1.0), 0, 1, 0, 1, 0, 1, true, false, true);}));
    ASSERT(throwsOpenMMException([&] () {mixed.setFunctionParameters(3, 2, 2, vector<double>(12, 1.0), 0, 1, 0, 1, 0, 1);}));
}

void testDiscreteFunctionsHaveNoPeriodicAxes() {
    Discrete1DFunction discrete(vector<double>(3, 1.0));
    ASSERT(!discrete.getPeriodic());
    ASSERT(throwsOpenMMException([&] () {discrete.getPeriodic(0);}));
}

void testCopyAndEqualityKeepPeriodicity() {
    vector<double> values1(5, 1.0);
    Continuous1DFunction periodic1(values1, 0, 1, true);
    Continuous1DFunction plain1(values1, 0, 1, false);
    Continuous1DFunction* copy1 = periodic1.Copy();
    ASSERT(copy1->getPeriodic());
    ASSERT(*copy1 == periodic1);
    ASSERT(!(periodic1 == plain1));
    delete copy1;

    vector<double> values2(9, 1.0);
    Continuous2DFunction mixed2(3, 3, values2, 0, 1, 0, 1, false, true);
    Continuous2DFunction periodic2(3, 3, values2, 0, 1, 0, 1, true);
    Continuous2DFunction* copy2 = mixed2.Copy();
    ASSERT(!copy2->getPeriodic(0) && copy2->getPeriodic(1));
    ASSERT(*copy2 == mixed2);
    ASSERT(!(mixed2 == periodic2));
    delete copy2;

    vector<double> values3(27, 1.0);
    Continuous3DFunction mixed3(3, 3, 3, values3, 0, 1, 0, 1, 0, 1, true, false, true);
    Continuous3DFunction plain3(3, 3, 3, values3, 0, 1, 0, 1, 0, 1);
    Continuous3DFunction* copy3 = mixed3.Copy();
    ASSERT(copy3->getPeriodic(0) && !copy3->getPeriodic(1) && copy3->getPeriodic(2));
    ASSERT(*copy3 == mixed3);
    ASSERT(!(mixed3 == plain3));
    delete copy3;
}

int main() {
    try {
        testContinuous1DPeriodicAxis();
        testContinuous2DPeriodicAxes();
        testContinuous3DPeriodicAxes();
        testDiscreteFunctionsHaveNoPeriodicAxes();
        testCopyAndEqualityKeepPeriodicity();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
