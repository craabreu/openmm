/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2010 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/SplineFitter.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

void testNaturalSpline() {
    vector<double> x(20);
    vector<double> y(20);
    for (unsigned int i = 0; i < x.size(); i++) {
        x[i] = 0.5*i+0.1*sin(double(i));
        y[i] = sin(double(x[i]));
    }
    vector<double> deriv;
    SplineFitter::createNaturalSpline(x, y, deriv);
    ASSERT_EQUAL_TOL(deriv[0], 0.0, 1e-6);
    ASSERT_EQUAL_TOL(deriv[deriv.size()-1], 0.0, 1e-6);
    for (unsigned int i = 0; i < x.size(); i++) {
        ASSERT_EQUAL_TOL(y[i], SplineFitter::evaluateSpline(x, y, deriv, x[i]), 1e-6);
        ASSERT_EQUAL_TOL(cos(x[i]), SplineFitter::evaluateSplineDerivative(x, y, deriv, x[i]), 0.05);
    }
    for (int i = 1; i < 9; i++) {
        ASSERT_EQUAL_TOL(sin((double)i), SplineFitter::evaluateSpline(x, y, deriv, i), 0.05);
        ASSERT_EQUAL_TOL(cos((double)i), SplineFitter::evaluateSplineDerivative(x, y, deriv, i), 0.05);
    }
}

void testPeriodicSpline() {
    vector<double> x(26);
    vector<double> y(26);
    for (unsigned int i = 0; i < x.size()-1; i++) {
        x[i] = 0.5*i+0.1*sin((double)i);
        y[i] = sin((double)x[i]);
    }
    x[x.size()-1] = 4*M_PI;
    y[y.size()-1] = y[0];
    vector<double> deriv;
    SplineFitter::createPeriodicSpline(x, y, deriv);
    ASSERT_EQUAL_TOL(deriv[0], deriv[deriv.size()-1], 1e-6);
    for (unsigned int i = 0; i < x.size(); i++) {
        ASSERT_EQUAL_TOL(y[i], SplineFitter::evaluateSpline(x, y, deriv, x[i]), 1e-6);
        ASSERT_EQUAL_TOL(cos(x[i]), SplineFitter::evaluateSplineDerivative(x, y, deriv, x[i]), 0.05);
    }
    for (int i = 1; i < 9; i++) {
        ASSERT_EQUAL_TOL(sin((double)i), SplineFitter::evaluateSpline(x, y, deriv, i), 0.05);
        ASSERT_EQUAL_TOL(cos((double)i), SplineFitter::evaluateSplineDerivative(x, y, deriv, i), 0.05);
    }
    for (unsigned int i = 0; i < x.size(); i++)
        x[i] = i/(x.size()-1.0);
    double ya[] = {15.579, 16.235, 17.325, 18.741, 20.454, 22.517, 24.944, 27.554, 29.942, 31.657,
                   32.486, 32.612, 32.494, 32.532, 32.785, 32.917, 32.402, 30.842, 28.229, 24.989,
                   21.762, 19.074, 17.147, 15.970, 15.467, 15.579};
    // scipy.interpolate.CubicSpline solution:
    double sol[] = { 345.520,  271.991,  194.015,  174.449, 221.940, 250.291, 141.895, -131.620,
                    -447.916, -600.465, -472.723, -144.892, 137.290, 180.733, -53.971, -418.600,
                    -697.879, -708.635, -416.330,   22.704, 374.262, 501.498, 473.496,  417.019,
                     385.928,  345.520};
    y.assign(begin(ya), end(ya));
    SplineFitter::createPeriodicSpline(x, y, deriv);
    ASSERT_EQUAL_TOL(SplineFitter::evaluateSplineDerivative(x, y, deriv, x[0]),
                     SplineFitter::evaluateSplineDerivative(x, y, deriv, x[x.size()-1]), 1e-6);
    for (int i = 0; i < x.size(); i++)
        ASSERT_EQUAL_TOL(deriv[i], sol[i], 1e-3);
}

void test2DSpline() {
    const int xsize = 15;
    const int ysize = 17;
    vector<double> x(xsize);
    vector<double> y(ysize);
    vector<double> f(xsize*ysize);
    for (int i = 0; i < xsize; i++)
        x[i] = 0.5*i+0.1*sin(double(i));
    for (int i = 0; i < ysize; i++)
        y[i] = 0.6*i+0.1*sin(double(i));
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++)
            f[i+j*xsize] = sin(x[i])*cos(0.4*y[j]);
    vector<vector<double> > c;
    SplineFitter::create2DNaturalSpline(x, y, f, c);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++) {
            double value = SplineFitter::evaluate2DSpline(x, y, f, c, x[i], y[j]);
            ASSERT_EQUAL_TOL(f[i+j*xsize], value, 1e-6);
        }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            double s = x[0]+(i+1)*(x[xsize-1]-x[0])/12.0;
            double t = y[0]+(j+1)*(y[ysize-1]-y[0])/12.0;
            double value = SplineFitter::evaluate2DSpline(x, y, f, c, s, t);
            ASSERT_EQUAL_TOL(sin(s)*cos(0.4*t), value, 0.02);
            double dx, dy;
            SplineFitter::evaluate2DSplineDerivatives(x, y, f, c, s, t, dx, dy);
            ASSERT_EQUAL_TOL(cos(s)*cos(0.4*t), dx, 0.05);
            ASSERT_EQUAL_TOL(-0.4*sin(s)*sin(0.4*t), dy, 0.05);
        }
    }
}

void testPeriodic2DSpline() {
    const int xsize = 15;
    const int ysize = 17;
    vector<double> x(xsize);
    vector<double> y(ysize);
    vector<double> f(xsize*ysize);
    for (int i = 0; i < xsize; i++)
        x[i] = 2.0*M_PI*i/(xsize-1);
    for (int i = 0; i < ysize; i++)
        y[i] = 2.0*M_PI*i/(ysize-1);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++)
            f[i+j*xsize] = sin(x[i])*cos(y[j]);
    vector<vector<double> > c;
    SplineFitter::create2DSpline(x, y, f, true, c);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++) {
            double value = SplineFitter::evaluate2DSpline(x, y, f, c, x[i], y[j]);
            ASSERT_EQUAL_TOL(f[i+j*xsize], value, 1e-6);
        }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            double s = x[0]+(i+1)*(x[xsize-1]-x[0])/12.0;
            double t = y[0]+(j+1)*(y[ysize-1]-y[0])/12.0;
            double value = SplineFitter::evaluate2DSpline(x, y, f, c, s, t);
            ASSERT_EQUAL_TOL(sin(s)*cos(t), value, 0.02);
            double dx, dy;
            SplineFitter::evaluate2DSplineDerivatives(x, y, f, c, s, t, dx, dy);
            ASSERT_EQUAL_TOL(cos(s)*cos(t), dx, 0.05);
            ASSERT_EQUAL_TOL(-sin(s)*sin(t), dy, 0.05);
        }
    }
}

void test3DSpline() {
    const int xsize = 8;
    const int ysize = 9;
    const int zsize = 10;
    vector<double> x(xsize);
    vector<double> y(ysize);
    vector<double> z(zsize);
    vector<double> f(xsize*ysize*zsize);
    for (int i = 0; i < xsize; i++)
        x[i] = 0.2*i+0.02*sin(0.4*double(i));
    for (int i = 0; i < ysize; i++)
        y[i] = 0.2*i+0.02*sin(0.45*double(i));
    for (int i = 0; i < zsize; i++)
        z[i] = 0.2*i+0.02*sin(0.5*double(i));
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++)
            for (int k = 0; k < zsize; k++)
                f[i+j*xsize+k*xsize*ysize] = sin(x[i])*cos(0.4*y[j])*(1+z[k]);
    vector<vector<double> > c;
    SplineFitter::create3DNaturalSpline(x, y, z, f, c);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++) {
            for (int k = 0; k < zsize; k++) {
                double value = SplineFitter::evaluate3DSpline(x, y, z, f, c, x[i], y[j], z[k]);
                ASSERT_EQUAL_TOL(f[i+j*xsize+k*xsize*ysize], value, 1e-6);
            }
        }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                double s = x[0]+(i+1)*(x[xsize-1]-x[0])/12.0;
                double t = y[0]+(j+1)*(y[ysize-1]-y[0])/12.0;
                double u = z[0]+(k+1)*(z[zsize-1]-z[0])/12.0;
                double value = SplineFitter::evaluate3DSpline(x, y, z, f, c, s, t, u);
                ASSERT_EQUAL_TOL(sin(s)*cos(0.4*t)*(1+u), value, 0.02);
                double dx, dy, dz;
                SplineFitter::evaluate3DSplineDerivatives(x, y, z, f, c, s, t, u, dx, dy, dz);
                ASSERT_EQUAL_TOL(cos(s)*cos(0.4*t)*(1+u), dx, 0.1);
                ASSERT_EQUAL_TOL(-0.4*sin(s)*sin(0.4*t)*(1+u), dy, 0.1);
                ASSERT_EQUAL_TOL(sin(s)*cos(0.4*t), dz, 0.1);
            }
        }
    }
}

void testPeriodic3DSpline() {
    const int xsize = 11;
    const int ysize = 13;
    const int zsize = 15;
    vector<double> x(xsize);
    vector<double> y(ysize);
    vector<double> z(zsize);
    vector<double> f(xsize*ysize*zsize);
    for (int i = 0; i < xsize; i++)
        x[i] = 2.0*M_PI*i/(xsize-1);
    for (int i = 0; i < ysize; i++)
        y[i] = 2.0*M_PI*i/(ysize-1);
    for (int i = 0; i < zsize; i++)
        z[i] = 2.0*M_PI*i/(zsize-1);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++)
            for (int k = 0; k < zsize; k++)
                f[i+j*xsize+k*xsize*ysize] = sin(x[i])*cos(y[j])*(1.0-sin(z[k]));
    vector<vector<double> > c;
    SplineFitter::create3DSpline(x, y, z, f, true, c);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++) {
            for (int k = 0; k < zsize; k++) {
                double value = SplineFitter::evaluate3DSpline(x, y, z, f, c, x[i], y[j], z[k]);
                ASSERT_EQUAL_TOL(f[i+j*xsize+k*xsize*ysize], value, 1e-6);
            }
        }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                double s = x[0]+(i+1)*(x[xsize-1]-x[0])/12.0;
                double t = y[0]+(j+1)*(y[ysize-1]-y[0])/12.0;
                double u = z[0]+(k+1)*(z[zsize-1]-z[0])/12.0;
                double value = SplineFitter::evaluate3DSpline(x, y, z, f, c, s, t, u);
                ASSERT_EQUAL_TOL(sin(s)*cos(t)*(1.0-sin(u)), value, 0.02);
                double dx, dy, dz;
                SplineFitter::evaluate3DSplineDerivatives(x, y, z, f, c, s, t, u, dx, dy, dz);
                ASSERT_EQUAL_TOL(cos(s)*cos(t)*(1.0-sin(u)), dx, 0.1);
                ASSERT_EQUAL_TOL(-sin(s)*sin(t)*(1.0-sin(u)), dy, 0.1);
                ASSERT_EQUAL_TOL(-sin(s)*cos(t)*cos(u), dz, 0.1);
            }
        }
    }
}

static double axisNode(bool periodic, int i, int size) {
    // Periodic axes span [0, 2*pi]; non-periodic ones span [0, 1].
    return periodic ? 2.0*M_PI*i/(size-1) : (double) i/(size-1);
}

static double dataNode(bool periodic, int i, int size) {
    // Tabulate the last node of a periodic axis at the first one, so that both ends hold
    // exactly the same values, as periodic splines require.
    return axisNode(periodic, periodic && i == size-1 ? 0 : i, size);
}

static double axisValue(bool periodic, double t) {
    // exp(t) has different values at the ends of [0, 1], so a periodic fit along a
    // non-periodic axis must fail.
    return periodic ? sin(t) : exp(t);
}

static double axisDeriv(bool periodic, double t) {
    return periodic ? cos(t) : exp(t);
}

void testMixed2DSpline(bool xperiodic, bool yperiodic) {
    const int xsize = 15;
    const int ysize = 17;
    vector<double> x(xsize), y(ysize), f(xsize*ysize);
    for (int i = 0; i < xsize; i++)
        x[i] = axisNode(xperiodic, i, xsize);
    for (int j = 0; j < ysize; j++)
        y[j] = axisNode(yperiodic, j, ysize);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++)
            f[i+j*xsize] = axisValue(xperiodic, dataNode(xperiodic, i, xsize))*axisValue(yperiodic, dataNode(yperiodic, j, ysize));
    vector<vector<double> > c;
    SplineFitter::create2DSpline(x, y, f, xperiodic, yperiodic, c);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++)
            ASSERT_EQUAL_TOL(f[i+j*xsize], SplineFitter::evaluate2DSpline(x, y, f, c, x[i], y[j]), 1e-6);
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++) {
            double s = x[0]+(i+1)*(x[xsize-1]-x[0])/12.0;
            double t = y[0]+(j+1)*(y[ysize-1]-y[0])/12.0;
            ASSERT_EQUAL_TOL(axisValue(xperiodic, s)*axisValue(yperiodic, t), SplineFitter::evaluate2DSpline(x, y, f, c, s, t), 0.02);
            double dx, dy;
            SplineFitter::evaluate2DSplineDerivatives(x, y, f, c, s, t, dx, dy);
            ASSERT_EQUAL_TOL(axisDeriv(xperiodic, s)*axisValue(yperiodic, t), dx, 0.05);
            ASSERT_EQUAL_TOL(axisValue(xperiodic, s)*axisDeriv(yperiodic, t), dy, 0.05);
        }

    // The derivative along the periodic axis is continuous across the seam.

    for (int k = 1; k < 10; k++) {
        double dx0, dy0, dx1, dy1;
        if (xperiodic) {
            double t = y[0]+k*(y[ysize-1]-y[0])/10.0;
            SplineFitter::evaluate2DSplineDerivatives(x, y, f, c, x[0], t, dx0, dy0);
            SplineFitter::evaluate2DSplineDerivatives(x, y, f, c, x[xsize-1], t, dx1, dy1);
            ASSERT_EQUAL_TOL(dx0, dx1, 1e-6);
        }
        if (yperiodic) {
            double s = x[0]+k*(x[xsize-1]-x[0])/10.0;
            SplineFitter::evaluate2DSplineDerivatives(x, y, f, c, s, y[0], dx0, dy0);
            SplineFitter::evaluate2DSplineDerivatives(x, y, f, c, s, y[ysize-1], dx1, dy1);
            ASSERT_EQUAL_TOL(dy0, dy1, 1e-6);
        }
    }

    // Treating every axis as periodic must fail, since the non-periodic axis has
    // different values at its ends.

    bool thrown = false;
    try {
        SplineFitter::create2DSpline(x, y, f, true, c);
    }
    catch (const OpenMMException&) {
        thrown = true;
    }
    ASSERT(thrown);
}

void testMixed3DSpline(bool xperiodic, bool yperiodic, bool zperiodic) {
    const int xsize = 15;
    const int ysize = 13;
    const int zsize = 17;
    vector<double> x(xsize), y(ysize), z(zsize), f(xsize*ysize*zsize);
    for (int i = 0; i < xsize; i++)
        x[i] = axisNode(xperiodic, i, xsize);
    for (int j = 0; j < ysize; j++)
        y[j] = axisNode(yperiodic, j, ysize);
    for (int k = 0; k < zsize; k++)
        z[k] = axisNode(zperiodic, k, zsize);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++)
            for (int k = 0; k < zsize; k++)
                f[i+xsize*j+xsize*ysize*k] = axisValue(xperiodic, dataNode(xperiodic, i, xsize))*axisValue(yperiodic, dataNode(yperiodic, j, ysize))*axisValue(zperiodic, dataNode(zperiodic, k, zsize));
    vector<vector<double> > c;
    SplineFitter::create3DSpline(x, y, z, f, xperiodic, yperiodic, zperiodic, c);
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++)
            for (int k = 0; k < zsize; k++)
                ASSERT_EQUAL_TOL(f[i+xsize*j+xsize*ysize*k], SplineFitter::evaluate3DSpline(x, y, z, f, c, x[i], y[j], z[k]), 1e-6);
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            for (int k = 0; k < 6; k++) {
                double s = x[0]+(i+1)*(x[xsize-1]-x[0])/8.0;
                double t = y[0]+(j+1)*(y[ysize-1]-y[0])/8.0;
                double u = z[0]+(k+1)*(z[zsize-1]-z[0])/8.0;
                double gx = axisValue(xperiodic, s), gy = axisValue(yperiodic, t), gz = axisValue(zperiodic, u);
                ASSERT_EQUAL_TOL(gx*gy*gz, SplineFitter::evaluate3DSpline(x, y, z, f, c, s, t, u), 0.02);
                double dx, dy, dz;
                SplineFitter::evaluate3DSplineDerivatives(x, y, z, f, c, s, t, u, dx, dy, dz);
                ASSERT_EQUAL_TOL(axisDeriv(xperiodic, s)*gy*gz, dx, 0.05);
                ASSERT_EQUAL_TOL(gx*axisDeriv(yperiodic, t)*gz, dy, 0.05);
                ASSERT_EQUAL_TOL(gx*gy*axisDeriv(zperiodic, u), dz, 0.05);
            }
}

void testSizeErrorsNameTheirFunction() {
    // A table of the wrong size is reported by the function that received it.
    vector<double> x(3), y(3), z(3), f(5);
    for (int i = 0; i < 3; i++)
        x[i] = y[i] = z[i] = i;
    vector<vector<double> > c;
    string message;
    try {
        SplineFitter::create2DSpline(x, y, f, false, false, c);
    }
    catch (const OpenMMException& e) {
        message = e.what();
    }
    ASSERT_EQUAL(string("create2DSpline: incorrect number of values"), message);
    message = "";
    try {
        SplineFitter::create3DSpline(x, y, z, f, false, false, false, c);
    }
    catch (const OpenMMException& e) {
        message = e.what();
    }
    ASSERT_EQUAL(string("create3DSpline: incorrect number of values"), message);
}

int main() {
    try {
        testNaturalSpline();
        testPeriodicSpline();
        test2DSpline();
        testPeriodic2DSpline();
        test3DSpline();
        testPeriodic3DSpline();
        testMixed2DSpline(true, false);
        testMixed2DSpline(false, true);
        testMixed3DSpline(true, false, true);
        testMixed3DSpline(false, true, false);
        testMixed3DSpline(true, false, false);
        testSizeErrorsNameTheirFunction();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
