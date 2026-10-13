/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2012-2021 Stanford University and the Authors.      *
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
#include "openmm/Context.h"
#include "ReferencePlatform.h"
#include "openmm/CustomCompoundBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

void testBond(bool byParticles) {
    // Create a system using a CustomCompoundBondForce.

    System customSystem;
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    string expression;
    if (byParticles)
        expression = "0.5*kb*((distance(p1,p2)-b0)^2+(distance(p2,p3)-b0)^2)+0.5*ka*(angle(p2,p3,p4)-a0)^2+kt*(1+cos(dihedral(p1,p2,p3,p4)-t0))";
    else
        expression = "0.5*kb*((pointdistance(x1,y1,z1,x2,y2,z2)-b0)^2+(pointdistance(x2,y2,z2,x3,y3,z3)-b0)^2)+0.5*ka*(pointangle(x2,y2,z2,x3,y3,z3,x4,y4,z4)-a0)^2+kt*(1+cos(pointdihedral(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4)-t0))";
    CustomCompoundBondForce* custom = new CustomCompoundBondForce(4, expression);
    custom->addPerBondParameter("kb");
    custom->addPerBondParameter("ka");
    custom->addPerBondParameter("kt");
    custom->addPerBondParameter("b0");
    custom->addPerBondParameter("a0");
    custom->addPerBondParameter("t0");
    vector<int> particles(4);
    particles[0] = 0;
    particles[1] = 1;
    particles[2] = 3;
    particles[3] = 2;
    vector<double> parameters(6);
    parameters[0] = 1.5;
    parameters[1] = 0.8;
    parameters[2] = 0.6;
    parameters[3] = 1.1;
    parameters[4] = 2.9;
    parameters[5] = 1.3;
    custom->addBond(particles, parameters);
    customSystem.addForce(custom);
    ASSERT(!custom->usesPeriodicBoundaryConditions());
    ASSERT(!customSystem.usesPeriodicBoundaryConditions());

    // Create an identical system using standard forces.

    System standardSystem;
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    HarmonicBondForce* bonds = new HarmonicBondForce();
    bonds->addBond(0, 1, 1.1, 1.5);
    bonds->addBond(1, 3, 1.1, 1.5);
    standardSystem.addForce(bonds);
    HarmonicAngleForce* angles = new HarmonicAngleForce();
    angles->addAngle(1, 3, 2, 2.9, 0.8);
    standardSystem.addForce(angles);
    PeriodicTorsionForce* torsions = new PeriodicTorsionForce();
    torsions->addTorsion(0, 1, 3, 2, 1, 1.3, 0.6);
    standardSystem.addForce(torsions);

    // Set the atoms in various positions, and verify that both systems give identical forces and energy.

    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(customSystem, integrator1, platform);
    Context c2(standardSystem, integrator2, platform);
    vector<Vec3> positions(4);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < (int) positions.size(); j++)
            positions[j] = Vec3(5.0*genrand_real2(sfmt), 5.0*genrand_real2(sfmt), 5.0*genrand_real2(sfmt));
        c1.setPositions(positions);
        c2.setPositions(positions);
        State s1 = c1.getState(State::Forces | State::Energy);
        State s2 = c2.getState(State::Forces | State::Energy);
        for (int i = 0; i < customSystem.getNumParticles(); i++)
            ASSERT_EQUAL_VEC(s1.getForces()[i], s2.getForces()[i], TOL);
        ASSERT_EQUAL_TOL(s1.getPotentialEnergy(), s2.getPotentialEnergy(), TOL);
    }
    
    // Try changing the bond parameters and make sure it's still correct.
    
    parameters[0] = 1.6;
    parameters[3] = 1.3;
    custom->setBondParameters(0, particles, parameters);
    custom->updateParametersInContext(c1);
    bonds->setBondParameters(0, 0, 1, 1.3, 1.6);
    bonds->setBondParameters(1, 1, 3, 1.3, 1.6);
    bonds->updateParametersInContext(c2);
    {
        State s1 = c1.getState(State::Forces | State::Energy);
        State s2 = c2.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = s1.getForces();
        for (int i = 0; i < customSystem.getNumParticles(); i++)
            ASSERT_EQUAL_VEC(s1.getForces()[i], s2.getForces()[i], TOL);
        ASSERT_EQUAL_TOL(s1.getPotentialEnergy(), s2.getPotentialEnergy(), TOL);
    }
}

void testPositionDependence() {
    System customSystem;
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    CustomCompoundBondForce* custom = new CustomCompoundBondForce(2, "scale1*distance(p1,p2)+scale2*x1+2*y2");
    custom->addGlobalParameter("scale1", 0.3);
    custom->addGlobalParameter("scale2", 0.2);
    vector<int> particles(2);
    particles[0] = 1;
    particles[1] = 0;
    vector<double> parameters;
    custom->addBond(particles, parameters);
    customSystem.addForce(custom);
    vector<Vec3> positions(2);
    positions[0] = Vec3(1.5, 1, 0);
    positions[1] = Vec3(0.5, 1, 0);
    VerletIntegrator integrator(0.01);
    Context context(customSystem, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(0.3*1.0+0.2*0.5+2*1, state.getPotentialEnergy(), 1e-5);
    ASSERT_EQUAL_VEC(Vec3(-0.3, -2, 0), state.getForces()[0], 1e-5);
    ASSERT_EQUAL_VEC(Vec3(0.3-0.2, 0, 0), state.getForces()[1], 1e-5);
}

void testContinuous2DFunction() {
    const int xsize = 10;
    const int ysize = 11;
    const double xmin = 0.4;
    const double xmax = 1.1;
    const double ymin = 0.0;
    const double ymax = 0.95;
    System system;
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomCompoundBondForce* forceField = new CustomCompoundBondForce(1, "fn(x1,y1)+1");
    vector<int> particles(1, 0);
    forceField->addBond(particles, vector<double>());
    vector<double> table(xsize*ysize);
    for (int i = 0; i < xsize; i++) {
        for (int j = 0; j < ysize; j++) {
            double x = xmin + i*(xmax-xmin)/xsize;
            double y = ymin + j*(ymax-ymin)/ysize;
            table[i+xsize*j] = sin(0.25*x)*cos(0.33*y);
        }
    }
    forceField->addTabulatedFunction("fn", new Continuous2DFunction(xsize, ysize, table, xmin, xmax, ymin, ymax));
    system.addForce(forceField);
    Context context(system, integrator, platform);
    vector<Vec3> positions(1);
    for (double x = xmin-0.15; x < xmax+0.2; x += 0.1) {
        for (double y = ymin-0.15; y < ymax+0.2; y += 0.1) {
            positions[0] = Vec3(x, y, 1.5);
            context.setPositions(positions);
            State state = context.getState(State::Forces | State::Energy);
            const vector<Vec3>& forces = state.getForces();
            double energy = 1;
            Vec3 force(0, 0, 0);
            if (x >= xmin && x <= xmax && y >= ymin && y <= ymax) {
                energy = sin(0.25*x)*cos(0.33*y)+1;
                force[0] = -0.25*cos(0.25*x)*cos(0.33*y);
                force[1] = 0.3*sin(0.25*x)*sin(0.33*y);
            }
            ASSERT_EQUAL_VEC(force, forces[0], 0.1);
            ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), 0.05);
        }
    }

    // Try updating the tabulated function.

    for (int i = 0; i < table.size(); i++)
        table[i] *= 0.5;
    Continuous2DFunction& fn = dynamic_cast<Continuous2DFunction&>(forceField->getTabulatedFunction(0));
    fn.setFunctionParameters(xsize, ysize, table, xmin, xmax, ymin, ymax);
    forceField->updateParametersInContext(context);
    for (double x = xmin-0.15; x < xmax+0.2; x += 0.1) {
        for (double y = ymin-0.15; y < ymax+0.2; y += 0.1) {
            positions[0] = Vec3(x, y, 1.5);
            context.setPositions(positions);
            State state = context.getState(State::Forces | State::Energy);
            const vector<Vec3>& forces = state.getForces();
            double energy = 1;
            Vec3 force(0, 0, 0);
            if (x >= xmin && x <= xmax && y >= ymin && y <= ymax) {
                energy = 0.5*sin(0.25*x)*cos(0.33*y)+1;
                force[0] = 0.5*(-0.25*cos(0.25*x)*cos(0.33*y));
                force[1] = 0.5*0.3*sin(0.25*x)*sin(0.33*y);
            }
            ASSERT_EQUAL_VEC(force, forces[0], 0.1);
            ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), 0.05);
        }
    }
}

void testContinuous3DFunction() {
    const int xsize = 10;
    const int ysize = 11;
    const int zsize = 12;
    const double xmin = 0.4;
    const double xmax = 1.1;
    const double ymin = 2.0;
    const double ymax = 2.9;
    const double zmin = 0.2;
    const double zmax = 1.35;
    System system;
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomCompoundBondForce* forceField = new CustomCompoundBondForce(1, "fn(x1,y1,z1)+1");
    vector<int> particles(1, 0);
    forceField->addBond(particles, vector<double>());
    vector<double> table(xsize*ysize*zsize);
    for (int i = 0; i < xsize; i++) {
        for (int j = 0; j < ysize; j++) {
            for (int k = 0; k < zsize; k++) {
                double x = xmin + i*(xmax-xmin)/xsize;
                double y = ymin + j*(ymax-ymin)/ysize;
                double z = zmin + k*(zmax-zmin)/zsize;
                table[i+xsize*j+xsize*ysize*k] = sin(0.25*x)*cos(0.33*y)*(1+z);
            }
        }
    }
    forceField->addTabulatedFunction("fn", new Continuous3DFunction(xsize, ysize, zsize, table, xmin, xmax, ymin, ymax, zmin, zmax));
    system.addForce(forceField);
    Context context(system, integrator, platform);
    vector<Vec3> positions(1);
    for (double x = xmin-0.15; x < xmax+0.2; x += 0.1) {
        for (double y = ymin-0.15; y < ymax+0.2; y += 0.1) {
            for (double z = zmin-0.15; z < zmax+0.2; z += 0.1) {
                positions[0] = Vec3(x, y, z);
                context.setPositions(positions);
                State state = context.getState(State::Forces | State::Energy);
                const vector<Vec3>& forces = state.getForces();
                double energy = 1;
                Vec3 force(0, 0, 0);
                if (x >= xmin && x <= xmax && y >= ymin && y <= ymax && z >= zmin && z <= zmax) {
                    energy = sin(0.25*x)*cos(0.33*y)*(1.0+z)+1;
                    force[0] = -0.25*cos(0.25*x)*cos(0.33*y)*(1.0+z);
                    force[1] = 0.3*sin(0.25*x)*sin(0.33*y)*(1.0+z);
                    force[2] = -sin(0.25*x)*cos(0.33*y);
                }
                ASSERT_EQUAL_VEC(force, forces[0], 0.1);
                ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), 0.05);
            }
        }
    }
}

void testMultipleBonds() {
    // Two compound bonds using Urey-Bradley example from API doc
    System customSystem;
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    CustomCompoundBondForce* custom = new CustomCompoundBondForce(3,
            "0.5*(kangle*(angle(p1,p2,p3)-theta0)^2+kbond*(distance(p1,p3)-r0)^2)");
    custom->addPerBondParameter("kangle");
    custom->addPerBondParameter("kbond");
    custom->addPerBondParameter("theta0");
    custom->addPerBondParameter("r0");
    vector<double> parameters(4);
    parameters[0] = 1.0;
    parameters[1] = 1.0;
    parameters[2] = 2 * M_PI / 3;
    parameters[3] = sqrt(3.0) / 2;
    vector<int> particles0(3);
    particles0[0] = 0;
    particles0[1] = 1;
    particles0[2] = 2;
    vector<int> particles1(3);
    particles1[0] = 1;
    particles1[1] = 2;
    particles1[2] = 3;
    custom->addBond(particles0, parameters);
    custom->addBond(particles1, parameters);
    customSystem.addForce(custom);

    vector<Vec3> positions(4);
    positions[0] = Vec3(0, 0.5, 0);
    positions[1] = Vec3(0, 0, 0);
    positions[2] = Vec3(0.5, 0, 0);
    positions[3] = Vec3(0.6, 0, 0.4);
    VerletIntegrator integrator(0.01);
    Context context(customSystem, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(0.199, state.getPotentialEnergy(), 1e-3);
    vector<Vec3> forces(state.getForces());
    ASSERT_EQUAL_VEC(Vec3(-1.160, 0.112, 0.0), forces[0], 1e-3);
    ASSERT_EQUAL_VEC(Vec3(0.927, 1.047, -0.638), forces[1], 1e-3);
    ASSERT_EQUAL_VEC(Vec3(-0.543, -1.160, 0.721), forces[2], 1e-3);
    ASSERT_EQUAL_VEC(Vec3(0.776, 0.0, -0.084), forces[3], 1e-3);
}

void testIllegalVariable() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    CustomCompoundBondForce* force = new CustomCompoundBondForce(2, "1+none");
    force->addBond({0, 1});
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    bool threwException = false;
    try {
        Context(system, integrator, platform);
    }
    catch (const exception& e) {
        threwException = true;
    }
    ASSERT(threwException);
}

void testPeriodicSeam() {
    // An argument just below the minimum of a periodic table must wrap to the start of
    // the table. In single precision, (x - floor(x)) rounds to 1 for such arguments,
    // which used to index one cell past the end of the table.
    const int size = 11;
    vector<double> table1(size), table2(size*size), table3(size*size*size);
    for (int i = 0; i < size; i++) {
        double ci = cos(2*M_PI*i/(size-1));
        table1[i] = ci+2;
        for (int j = 0; j < size; j++) {
            double cj = cos(2*M_PI*j/(size-1));
            table2[i+size*j] = ci+cj+2;
            for (int k = 0; k < size; k++)
                table3[i+size*j+size*size*k] = ci+cj+cos(2*M_PI*k/(size-1))+2;
        }
    }
    System system;
    system.addParticle(1.0);
    CustomCompoundBondForce* force = new CustomCompoundBondForce(1, "f1(x1)+f2(x1,y1)+f3(x1,y1,z1)");
    force->addBond({0}, {});
    force->addTabulatedFunction("f1", new Continuous1DFunction(table1, 0.0, 1.0, true));
    force->addTabulatedFunction("f2", new Continuous2DFunction(size, size, table2, 0.0, 1.0, 0.0, 1.0, true));
    force->addTabulatedFunction("f3", new Continuous3DFunction(size, size, size, table3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, true));
    system.addForce(force);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);

    // At the origin, f1 = 3, f2 = 4 and f3 = 5.

    for (double delta : {0.0, 1e-9, 1e-8, 2.9e-8, 1e-7}) {
        context.setPositions({Vec3(-delta, -delta, -delta)});
        ASSERT_EQUAL_TOL(12.0, context.getState(State::Energy).getPotentialEnergy(), 1e-4);
    }
}

/**
 * One axis of a test table. Periodic axes span [0, 2*pi] and tabulate cos(t); non-periodic
 * axes span [0.2, 1.1] and tabulate cos(0.33*t). Sample points extend beyond each range.
 */
struct TableAxis {
    bool periodic;
    double min, max;
    int size;
    TableAxis(bool periodic) : periodic(periodic), min(periodic ? 0.0 : 0.2), max(periodic ? 2*M_PI : 1.1), size(periodic ? 25 : 15) {
    }
    double node(int i) const {
        return min+i*(max-min)/(size-1);
    }
    double value(double t) const {
        return periodic ? cos(t) : cos(0.33*t);
    }
    double deriv(double t) const {
        return periodic ? -sin(t) : -0.33*sin(0.33*t);
    }
    bool inRange(double t) const {
        return periodic || (t >= min && t <= max);
    }
    vector<double> samples() const {
        vector<double> points;
        if (periodic) {
            for (double t = -1.5*M_PI; t < 3.5*M_PI; t += 0.7)
                points.push_back(t);
            // Just below the minimum, where single precision once indexed past the table.
            points.push_back(min-1e-8);
        }
        else
            for (double t = min-0.15; t < max+0.2; t += 0.1)
                points.push_back(t);
        return points;
    }
};

void testMixedPeriodicContinuous2DFunction(bool xperiodic, bool yperiodic) {
    TableAxis ax(xperiodic), ay(yperiodic);
    vector<double> table(ax.size*ay.size);
    for (int i = 0; i < ax.size; i++)
        for (int j = 0; j < ay.size; j++)
            table[i+ax.size*j] = ax.value(ax.node(i))*ay.value(ay.node(j));
    System system;
    system.addParticle(1.0);
    CustomCompoundBondForce* force = new CustomCompoundBondForce(1, "fn(x1,y1)+1");
    force->addBond({0}, {});
    force->addTabulatedFunction("fn", new Continuous2DFunction(ax.size, ay.size, table, ax.min, ax.max, ay.min, ay.max, xperiodic, yperiodic));
    system.addForce(force);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    for (double x : ax.samples())
        for (double y : ay.samples()) {
            context.setPositions({Vec3(x, y, 1.5)});
            State state = context.getState(State::Forces | State::Energy);
            double energy = 1;
            Vec3 expectedForce(0, 0, 0);
            if (ax.inRange(x) && ay.inRange(y)) {
                energy += ax.value(x)*ay.value(y);
                expectedForce = Vec3(-ax.deriv(x)*ay.value(y), -ax.value(x)*ay.deriv(y), 0);
            }
            ASSERT_EQUAL_VEC(expectedForce, state.getForces()[0], 0.05);
            ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), 0.02);
        }
}

void testMixedPeriodicContinuous3DFunction(bool xperiodic, bool yperiodic, bool zperiodic) {
    TableAxis ax(xperiodic), ay(yperiodic), az(zperiodic);
    vector<double> table(ax.size*ay.size*az.size);
    for (int i = 0; i < ax.size; i++)
        for (int j = 0; j < ay.size; j++)
            for (int k = 0; k < az.size; k++)
                table[i+ax.size*j+ax.size*ay.size*k] = ax.value(ax.node(i))*ay.value(ay.node(j))*az.value(az.node(k));
    System system;
    system.addParticle(1.0);
    CustomCompoundBondForce* force = new CustomCompoundBondForce(1, "fn(x1,y1,z1)+1");
    force->addBond({0}, {});
    force->addTabulatedFunction("fn", new Continuous3DFunction(ax.size, ay.size, az.size, table, ax.min, ax.max, ay.min, ay.max, az.min, az.max, xperiodic, yperiodic, zperiodic));
    system.addForce(force);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    for (double x : ax.samples())
        for (double y : ay.samples())
            for (double z : az.samples()) {
                context.setPositions({Vec3(x, y, z)});
                State state = context.getState(State::Forces | State::Energy);
                double energy = 1;
                Vec3 expectedForce(0, 0, 0);
                if (ax.inRange(x) && ay.inRange(y) && az.inRange(z)) {
                    double gx = ax.value(x), gy = ay.value(y), gz = az.value(z);
                    energy += gx*gy*gz;
                    expectedForce = Vec3(-ax.deriv(x)*gy*gz, -gx*ay.deriv(y)*gz, -gx*gy*az.deriv(z));
                }
                ASSERT_EQUAL_VEC(expectedForce, state.getForces()[0], 0.05);
                ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), 0.02);
            }
}

void testPeriodic(bool byParticles) {
    // Create a force that uses periodic boundary conditions.

    System customSystem;
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
    string expression;
    if (byParticles)
        expression = "0.5*kb*((distance(p1,p2)-b0)^2+(distance(p2,p3)-b0)^2)+0.5*ka*(angle(p2,p3,p4)-a0)^2+kt*(1+cos(dihedral(p1,p2,p3,p4)-t0))";
    else
        expression = "0.5*kb*((pointdistance(x1,y1,z1,x2,y2,z2)-b0)^2+(pointdistance(x2,y2,z2,x3,y3,z3)-b0)^2)+0.5*ka*(pointangle(x2,y2,z2,x3,y3,z3,x4,y4,z4)-a0)^2+kt*(1+cos(pointdihedral(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4)-t0))";
    CustomCompoundBondForce* custom = new CustomCompoundBondForce(4, expression);
    custom->addPerBondParameter("kb");
    custom->addPerBondParameter("ka");
    custom->addPerBondParameter("kt");
    custom->addPerBondParameter("b0");
    custom->addPerBondParameter("a0");
    custom->addPerBondParameter("t0");
    vector<int> particles(4);
    particles[0] = 0;
    particles[1] = 1;
    particles[2] = 3;
    particles[3] = 2;
    vector<double> parameters(6);
    parameters[0] = 1.5;
    parameters[1] = 0.8;
    parameters[2] = 0.6;
    parameters[3] = 1.1;
    parameters[4] = 2.9;
    parameters[5] = 1.3;
    custom->addBond(particles, parameters);
    custom->setUsesPeriodicBoundaryConditions(true);
    customSystem.addForce(custom);

    // Create an identical system using standard forces.

    System standardSystem;
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    standardSystem.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
    HarmonicBondForce* bonds = new HarmonicBondForce();
    bonds->addBond(0, 1, 1.1, 1.5);
    bonds->addBond(1, 3, 1.1, 1.5);
    bonds->setUsesPeriodicBoundaryConditions(true);
    standardSystem.addForce(bonds);
    HarmonicAngleForce* angles = new HarmonicAngleForce();
    angles->addAngle(1, 3, 2, 2.9, 0.8);
    angles->setUsesPeriodicBoundaryConditions(true);
    standardSystem.addForce(angles);
    PeriodicTorsionForce* torsions = new PeriodicTorsionForce();
    torsions->addTorsion(0, 1, 3, 2, 1, 1.3, 0.6);
    torsions->setUsesPeriodicBoundaryConditions(true);
    standardSystem.addForce(torsions);

    // Set the atoms in various positions, and verify that both systems give identical forces and energy.

    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(customSystem, integrator1, platform);
    Context c2(standardSystem, integrator2, platform);
    vector<Vec3> positions(4);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < (int) positions.size(); j++)
            positions[j] = Vec3(5.0*genrand_real2(sfmt), 5.0*genrand_real2(sfmt), 5.0*genrand_real2(sfmt));
        c1.setPositions(positions);
        c2.setPositions(positions);
        State s1 = c1.getState(State::Forces | State::Energy);
        State s2 = c2.getState(State::Forces | State::Energy);
        for (int i = 0; i < customSystem.getNumParticles(); i++)
            ASSERT_EQUAL_VEC(s1.getForces()[i], s2.getForces()[i], TOL);
        ASSERT_EQUAL_TOL(s1.getPotentialEnergy(), s2.getPotentialEnergy(), TOL);
    }
}

void testEnergyParameterDerivatives() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomCompoundBondForce* custom = new CustomCompoundBondForce(4, "k*(dihedral(p1,p2,p3,p4)-theta0)^2");
    custom->addGlobalParameter("theta0", 0.0);
    custom->addGlobalParameter("k", 0.0);
    custom->addEnergyParameterDerivative("theta0");
    custom->addEnergyParameterDerivative("k");
    vector<int> particles(4);
    particles[0] = 0;
    particles[1] = 1;
    particles[2] = 2;
    particles[3] = 3;
    vector<double> parameters;
    custom->addBond(particles, parameters);
    system.addForce(custom);
    Context context(system, integrator, platform);
    vector<Vec3> positions(4);
    positions[0] = Vec3(0, 2, 0);
    positions[1] = Vec3(0, 0, 0);
    positions[2] = Vec3(1, 0, 0);
    positions[3] = Vec3(1, 1, 1);
    context.setPositions(positions);
    double theta = M_PI/4;
    for (int i = 0; i < 10; i++) {
        double theta0 = 0.1*i;
        double k = 10-i;
        context.setParameter("theta0", theta0);
        context.setParameter("k", k);
        State state = context.getState(State::ParameterDerivatives);
        map<string, double> derivs = state.getEnergyParameterDerivatives();
        double dEdtheta0 = -2*k*(theta-theta0);
        double dEdk = (theta-theta0)*(theta-theta0);
        ASSERT_EQUAL_TOL(dEdtheta0, derivs["theta0"], 1e-5);
        ASSERT_EQUAL_TOL(dEdk, derivs["k"], 1e-5);
    }
}

void testParallelComputation() {
    System system;
    const int numParticles = 200;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    CustomCompoundBondForce* force = new CustomCompoundBondForce(2, ("(distance(p1,p2)-1.1)^2"));
    vector<int> particles(2);
    vector<double> params;
    for (int i = 1; i < numParticles; i++) {
        particles[0] = i-1;
        particles[1] = i;
        force->addBond(particles, params);
    }
    system.addForce(force);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = Vec3(i, 0, 0);
    VerletIntegrator integrator1(0.01);
    Context context1(system, integrator1, platform);
    context1.setPositions(positions);
    State state1 = context1.getState(State::Forces | State::Energy);
    VerletIntegrator integrator2(0.01);
    string deviceIndex = platform.getPropertyValue(context1, "DeviceIndex");
    map<string, string> props;
    props["DeviceIndex"] = deviceIndex+","+deviceIndex;
    Context context2(system, integrator2, platform, props);
    context2.setPositions(positions);
    State state2 = context2.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy(), state2.getPotentialEnergy(), 1e-5);
    for (int i = 0; i < numParticles; i++)
        ASSERT_EQUAL_VEC(state1.getForces()[i], state2.getForces()[i], 1e-5);
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testBond(true);
        testBond(false);
        testPositionDependence();
        testContinuous2DFunction();
        testContinuous3DFunction();
        testPeriodicSeam();
        testMixedPeriodicContinuous2DFunction(true, false);
        testMixedPeriodicContinuous2DFunction(false, true);
        testMixedPeriodicContinuous3DFunction(true, false, true);
        testMixedPeriodicContinuous3DFunction(false, true, false);
        testMixedPeriodicContinuous3DFunction(true, false, false);
        testMultipleBonds();
        testIllegalVariable();
        testPeriodic(true);
        testPeriodic(false);
        testEnergyParameterDerivatives();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}


