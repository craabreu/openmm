import unittest
from math import pi
from openmm import *
from openmm.app import *
from openmm.unit import *

class TestMetadynamics(unittest.TestCase):
    """Test the Metadynamics class"""

    def testHarmonicOscillator(self):
        """Test running metadynamics on a harmonic oscillator."""
        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        force = HarmonicBondForce()
        force.addBond(0, 1, 1.0, 100000.0)
        system.addForce(force)
        cv = CustomBondForce('r')
        cv.addBond(0, 1)
        bias = BiasVariable(cv, 0.94, 1.06, 0.00431, gridWidth=31)
        meta = Metadynamics(system, [bias], 300*kelvin, 3.0, 5.0, 10)
        integrator = LangevinIntegrator(300*kelvin, 10/picosecond, 0.001*picosecond)
        integrator.setRandomNumberSeed(4321)
        topology = Topology()
        chain = topology.addChain()
        residue = topology.addResidue('H2', chain)
        topology.addAtom('H1', element.hydrogen, residue)
        topology.addAtom('H2', element.hydrogen, residue)
        simulation = Simulation(topology, system, integrator, Platform.getPlatform('Reference'))
        simulation.context.setPositions([Vec3(0, 0, 0), Vec3(1, 0, 0)])
        meta.step(simulation, 200000)
        fe = meta.getFreeEnergy()
        center = bias.gridWidth//2
        fe -= fe[center]

        # Energies should be reasonably well converged over the central part of the range.

        for i in range(center-3, center+4):
            r = bias.minValue + i*(bias.maxValue-bias.minValue)/(bias.gridWidth-1)
            e = 0.5*100000.0*(r-1.0)**2*kilojoules_per_mole
            assert abs(fe[i]-e) < 1.0*kilojoules_per_mole

    def testMixedPeriodicVariables(self):
        """Test metadynamics with one periodic and one non-periodic variable."""
        system = System()
        system.addParticle(1.0)
        restraint = CustomExternalForce('100*(sqrt(x^2+y^2)-1)^2 + 100*z^2')
        restraint.addParticle(0)
        system.addForce(restraint)
        angle = CustomExternalForce('atan2(y, x)')
        angle.addParticle(0)
        height = CustomExternalForce('z')
        height.addParticle(0)
        phi = BiasVariable(angle, -pi, pi, 0.3, periodic=True, gridWidth=41)
        z = BiasVariable(height, -0.5, 0.5, 0.05, periodic=False, gridWidth=31)
        meta = Metadynamics(system, [phi, z], 300*kelvin, 5.0, 1.0, 50)
        self.assertTrue(meta._table.getPeriodic(0))
        self.assertFalse(meta._table.getPeriodic(1))
        integrator = LangevinIntegrator(300*kelvin, 10/picosecond, 0.001*picosecond)
        integrator.setRandomNumberSeed(1234)
        topology = Topology()
        chain = topology.addChain()
        residue = topology.addResidue('A', chain)
        topology.addAtom('A', element.argon, residue)
        simulation = Simulation(topology, system, integrator, Platform.getPlatform('Reference'))
        simulation.context.setPositions([Vec3(1, 0, 0)])
        meta.step(simulation, 2000)
        fe = meta.getFreeEnergy().value_in_unit(kilojoules_per_mole)
        self.assertEqual((z.gridWidth, phi.gridWidth), fe.shape)
        self.assertTrue(fe.min() < 0)
        # The periodic axis has equal values at both ends of its grid.
        self.assertTrue(all(abs(fe[:, 0] - fe[:, -1]) < 1e-10))
